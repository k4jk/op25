// Gardner clock recovery + 4-level C4FM slicer (Phase 1) and
// complex Gardner + H-DQPSK differential decode (Phase 2)
//
// One thread per channel; the timing loop is sequential within each thread
// (Gardner feedback is causal).  Both Phase 1 and Phase 2 channels are
// handled in the same kernel, selected per-channel via d_channel_mode.
//
// Grid: ceil(M/512)   Block: 512
//
// --- Phase 1 (C4FM, FDMA) ---
// Input:  d_fm  [in_steps * M], step-major — FM demod output (radians/sample)
// Why Gardner over M&M:
//   Gardner error = (x[n-1] - x[n+1]) × x_mid[n]
//   The DC carrier offset cancels in the difference (x[n-1] - x[n+1]) because
//   both symbol samples carry the same DC component.  M&M requires slicer
//   decisions for its error term; wrong decisions (from unremoved DC offset)
//   drive the timing loop in the wrong direction — a vicious cycle during
//   re-acquisition after a control-channel timeout.
//
// Cross-block state (Phase 1):
//   d_mm_mu          — fractional timing carry-over (may be in (-1, sps))
//   d_mm_omega       — per-channel samples-per-symbol estimate
//   d_mm_fm_last     — last 3 FM samples of the previous block:
//                        [k]=fm(-1), [M+k]=fm(-2), [2M+k]=fm(-3)
//   d_mm_last_interp — previous raw symbol sample (Gardner error numerator)
//   d_mm_dc_est      — IIR DC carrier estimate (for slicer only, not timing)
//   d_mm_fast_ctr    — fast-gain countdown (symbols remaining in acquisition)
//
// Why 3 samples of FM history:
//   After a 30-symbol batch the carry-out = carry_in + 30*sps - in_steps.
//   With carry_in ≈ 0.874 (the 30/31 boundary) and sps=2.6042:
//     carry_out = 0.874 + 78.126 - 80 = -1.000
//   The midpoint probe at the first strobe of the next batch:
//     mu_mid = carry_out - sps/2 ≈ -1.000 - 1.302 = -2.302 → im0 = -3
//   Without fm(-3), that Gardner error must be zeroed, biasing omega low over
//   many batches and producing a systematic +1 dibit/LDU excess.
//
// --- Phase 2 (H-DQPSK, TDMA) ---
// Input:  d_s2  [in_steps * M], step-major — complex IQ from stage-2 output
//
// Gardner timing recovery on complex IQ samples, then differential decode at
// the symbol level (multiply consecutive complex symbols: diff = curr×conj(prev))
// → take arg → ±π/4 or ±3π/4.  Rescale by 4/π → ±1 or ±3 → dibit (same
// encoding as Phase 1, matching fsk4_slicer_fb in the non-CUDA cqpsk path).
//
// Cross-block state (Phase 2):
//   d_mm_mu, d_mm_omega, d_mm_fast_ctr — shared with Phase 1 (per-channel arrays)
//   d_mm_dc_est      — IIR DC estimate of decoded phase (removes carrier offset bias)
//   d_p2_iq_last     — last 3 complex IQ samples [3*M] for cross-block interpolation
//   d_p2_last_sym    — previous decoded complex symbol [M] for differential decode
//
// P25 C4FM deviation (Phase 1):
//   Outer symbols ±3: ±1800 Hz → ±1800·2π/12500 = ±0.9048 rad/samp
//   Inner symbols ±1:  ±600 Hz →  ±600·2π/12500 = ±0.3016 rad/samp
//   Decision threshold (midpoint):  (0.9048+0.3016)/2 = 0.6032 rad/samp
//
// P25 H-DQPSK symbol transitions (Phase 2, after differential decode):
//   Outer: ±3π/4 (±2.3562 rad) → rescaled to ±3.0
//   Inner: ±π/4  (±0.7854 rad) → rescaled to ±1.0
//   Decision threshold: ±π/2 (±1.5708 rad) → ±2.0 after rescale
//
// Dibit encoding (P25 standard, both phases):
//   Symbol +3 → dibit 01 (1)
//   Symbol +1 → dibit 00 (0)
//   Symbol -1 → dibit 10 (2)
//   Symbol -3 → dibit 11 (3)

#include "symbol_recovery.h"
#include <cstdio>
#include <cstring>

#define CUDA_CHECK_MM(call)                                                   \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            return false;                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// 4-level slicer for P25 Phase 1 C4FM (FM-demod output in rad/samp).
static __device__ float decide_c4fm(float y)
{
    const float T = 0.6032f;
    if (y >  T) return +3.0f;
    if (y > 0.0f) return +1.0f;
    if (y > -T) return -1.0f;
    return -3.0f;
}

// Map symbol value (±1.0 or ±3.0) to dibit (0-3).
static __device__ int8_t symbol_to_dibit(float d)
{
    if (d > +2.0f) return 1;   // +3 → 01
    if (d > 0.0f)  return 0;   // +1 → 00
    if (d > -2.0f) return 2;   // -1 → 10
    return 3;                   // -3 → 11
}

// 4-level slicer for P25 Phase 2 H-DQPSK.
// Input y = differential phase (radians), DC-corrected.
// Rescales by 4/π so ±π/4 → ±1.0 and ±3π/4 → ±3.0, then applies the same
// symbol_to_dibit thresholds — matching the non-CUDA cqpsk path:
//   diff_phasor_cc → complex_to_arg → ×(4/π) → fsk4_slicer_fb([-2,0,2,4])
static __device__ int8_t dqpsk_to_dibit(float y)
{
    float v = y * (4.0f / 3.14159265358979f);
    if (v > 2.0f)  return 1;   // +3π/4 → dibit 01
    if (v > 0.0f)  return 0;   // +π/4  → dibit 00
    if (v > -2.0f) return 2;   // -π/4  → dibit 10
    return 3;                   // -3π/4 → dibit 11
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void mm_recovery_kernel(
    const float*        __restrict__ d_fm,        // [in_steps * M] Phase 1 FM demod, step-major
    const cufftComplex* __restrict__ d_s2,        // [in_steps * M] Phase 2 complex IQ, step-major
    float*   d_mu,                                // [M] fractional timing phase (in/out)
    float*   d_omega,                             // [M] samples-per-symbol estimate (in/out)
    float*   d_fm_last,                           // [3*M] Phase 1: last 3 FM samples (in/out)
    cufftComplex* d_iq_last,                      // [3*M] Phase 2: last 3 complex IQ samples (in/out)
    float*   d_last_sym,                          // [M] Phase 1: previous raw symbol value (in/out)
    cufftComplex* d_last_sym_c,                   // [M] Phase 2: previous decoded complex symbol (in/out)
    float*   d_dc_est,                            // [M] IIR DC estimate (in/out)
    const int8_t* __restrict__ d_channel_mode,   // [M]: 1=Phase1, 2=Phase2
    float*   d_symbols,                           // [MM_MAX_SYM * M] output symbol values (Phase 1 debug)
    int8_t*  d_dibits,                            // [MM_MAX_SYM * M] output dibits
    int32_t* d_sym_count,                         // [M] output: valid symbols per channel
    int32_t* d_fast_ctr,                          // [M] fast-gain countdown (in/out)
    int M, int in_steps,
    float sps_p1, float sps_p2)                   // nominal sps for each mode
{
    const int k = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                + static_cast<int>(threadIdx.x);
    if (k >= M) return;

    const int8_t mode = d_channel_mode[k];

    float mu        = d_mu[k];
    float omega     = d_omega[k];
    float dc_est    = d_dc_est[k];
    int32_t fast_ctr = d_fast_ctr[k];

    // Per-channel omega clamp (±0.2% of nominal sps, matches gardner_cc d_omega_rel=0.002)
    const float sps_nom = (mode == 2) ? sps_p2 : sps_p1;
    const float sps_min = sps_nom * 0.998f;
    const float sps_max = sps_nom * 1.002f;

    // Two-phase gains: fast during initial acquisition, slow for steady-state jitter.
    const float gain_mu    = (fast_ctr > 0) ? 0.05f    : 0.025f;
    const float gain_omega = (fast_ctr > 0) ? 0.00025f : 0.0000625f;
    const float dc_alpha   = (fast_ctr > 0) ? 0.9f     : 0.995f;

    int n_sym = 0;

    // -----------------------------------------------------------------------
    // Phase 1: real Gardner on FM demod output
    // -----------------------------------------------------------------------
    if (mode != 2) {
        float fm_last  = d_fm_last[k];
        float fm_last2 = d_fm_last[M + k];
        float fm_last3 = d_fm_last[2*M + k];
        float last_sym = d_last_sym[k];

        // Fetch FM sample at position i within this block.
        // Negative indices address the cross-block history (down to -3).
        auto get_fm = [&](int i) -> float {
            if (i >= 0)  return d_fm[i * M + k];
            if (i == -1) return fm_last;
            if (i == -2) return fm_last2;
            return fm_last3;
        };

        while (n_sym < MM_MAX_SYM) {
            int i0 = static_cast<int>(floorf(mu));
            int i1 = i0 + 1;
            if (i1 >= in_steps) break;

            float frac     = mu - static_cast<float>(i0);
            float curr_sym = get_fm(i0) + frac * (get_fm(i1) - get_fm(i0));

            float mu_mid = mu - omega * 0.5f;
            int   im0    = static_cast<int>(floorf(mu_mid));
            int   im1    = im0 + 1;
            float frac_m = mu_mid - static_cast<float>(im0);
            float mid_samp = get_fm(im0) + frac_m * (get_fm(im1) - get_fm(im0));

            float e = (last_sym - curr_sym) * mid_samp;
            if (e >  1.0f) e =  1.0f;
            if (e < -1.0f) e = -1.0f;

            omega += gain_omega * e;
            if (omega < sps_min) omega = sps_min;
            if (omega > sps_max) omega = sps_max;

            mu      += omega + gain_mu * e;
            last_sym = curr_sym;

            dc_est += (1.0f - dc_alpha) * (curr_sym - dc_est);
            float y_dc = curr_sym - dc_est;

            float d = decide_c4fm(y_dc);
            d_symbols[n_sym * M + k] = y_dc;
            d_dibits [n_sym * M + k] = symbol_to_dibit(d);
            n_sym++;
        }

        d_fm_last[k]       = get_fm(in_steps - 1);
        d_fm_last[M + k]   = (in_steps >= 2) ? get_fm(in_steps - 2) : fm_last;
        d_fm_last[2*M + k] = (in_steps >= 3) ? get_fm(in_steps - 3) : fm_last2;
        d_last_sym[k] = last_sym;

    // -----------------------------------------------------------------------
    // Phase 2: complex Gardner on IQ, differential decode at symbol level
    // -----------------------------------------------------------------------
    } else {
        cufftComplex iq_m1 = d_iq_last[k];           // IQ at sample -1
        cufftComplex iq_m2 = d_iq_last[M + k];       // IQ at sample -2
        cufftComplex iq_m3 = d_iq_last[2*M + k];     // IQ at sample -3
        cufftComplex last_sym_c = d_last_sym_c[k];   // previous decoded complex symbol

        // Fetch complex IQ at position i (negative indices → cross-block history).
        auto get_iq = [&](int i) -> cufftComplex {
            if (i >= 0) return d_s2[i * M + k];
            if (i == -1) return iq_m1;
            if (i == -2) return iq_m2;
            return iq_m3;
        };

        // Bilinear interpolation of complex sample at fractional index mu.
        auto interp_iq = [&](float fmu) -> cufftComplex {
            int   i0   = static_cast<int>(floorf(fmu));
            float frac = fmu - static_cast<float>(i0);
            cufftComplex a = get_iq(i0);
            cufftComplex b = get_iq(i0 + 1);
            return { a.x + frac * (b.x - a.x), a.y + frac * (b.y - a.y) };
        };

        while (n_sym < MM_MAX_SYM) {
            int i1 = static_cast<int>(floorf(mu)) + 1;
            if (i1 >= in_steps) break;

            cufftComplex curr = interp_iq(mu);
            cufftComplex mid  = interp_iq(mu - omega * 0.5f);

            // Complex Gardner error: Re((last_sym_c - curr) × conj(mid))
            // Sign matches GNU Radio gardner_cc: e = (last - curr) * conj(mid).
            // Normalize by |mid|² to make loop gain independent of IQ amplitude.
            // The non-CUDA cqpsk path runs AGC before gardner_costas_cc; we
            // don't have AGC, so normalization is essential for stable timing.
            float diff_re = last_sym_c.x - curr.x;
            float diff_im = last_sym_c.y - curr.y;
            float e = diff_re * mid.x + diff_im * mid.y;
            float mid_pow = mid.x * mid.x + mid.y * mid.y;
            if (mid_pow > 1e-10f) e /= mid_pow;
            if (e >  1.0f) e =  1.0f;
            if (e < -1.0f) e = -1.0f;

            omega += gain_omega * e;
            if (omega < sps_min) omega = sps_min;
            if (omega > sps_max) omega = sps_max;
            mu += omega + gain_mu * e;

            // Differential decode: curr × conj(last_sym_c) → phase transition
            float d_re = curr.x * last_sym_c.x + curr.y * last_sym_c.y;
            float d_im = curr.y * last_sym_c.x - curr.x * last_sym_c.y;
            float phase = atan2f(d_im, d_re);   // ≈ ±π/4 or ±3π/4

            last_sym_c = curr;

            // DC removal: tracks carrier offset (2π×Δf/6000 rad/symbol bias)
            dc_est += (1.0f - dc_alpha) * (phase - dc_est);
            float phase_dc = phase - dc_est;

            d_symbols[n_sym * M + k] = phase_dc;
            d_dibits [n_sym * M + k] = dqpsk_to_dibit(phase_dc);
            n_sym++;
        }

        d_iq_last[k]       = get_iq(in_steps - 1);
        d_iq_last[M + k]   = (in_steps >= 2) ? get_iq(in_steps - 2) : iq_m1;
        d_iq_last[2*M + k] = (in_steps >= 3) ? get_iq(in_steps - 3) : iq_m2;
        d_last_sym_c[k] = last_sym_c;
    }

    d_mu[k]          = mu - static_cast<float>(in_steps);
    d_omega[k]       = omega;
    d_dc_est[k]      = dc_est;
    d_sym_count[k]   = n_sym;
    d_fast_ctr[k]    = (fast_ctr > n_sym) ? fast_ctr - n_sym : 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool mm_alloc(ChannelizerState& state)
{
    const int M = state.config.num_phases;

    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_mu,
                             static_cast<size_t>(M) * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_fm_last,
                             static_cast<size_t>(3 * M) * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_last_interp,
                             static_cast<size_t>(M) * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_last_dec,
                             static_cast<size_t>(M) * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_omega,
                             static_cast<size_t>(M) * sizeof(float)));

    cudaMemset(state.d_mm_mu,          0, static_cast<size_t>(M) * sizeof(float));
    cudaMemset(state.d_mm_fm_last,     0, static_cast<size_t>(3 * M) * sizeof(float));
    cudaMemset(state.d_mm_last_interp, 0, static_cast<size_t>(M) * sizeof(float));
    cudaMemset(state.d_mm_last_dec,    0, static_cast<size_t>(M) * sizeof(float));
    cudaMemset(state.d_mm_omega,       0, static_cast<size_t>(M) * sizeof(float));

    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_symbols,
                             static_cast<size_t>(MM_MAX_SYM) * M * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_dibits,
                             static_cast<size_t>(MM_MAX_SYM) * M * sizeof(int8_t)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_sym_count,
                             static_cast<size_t>(M) * sizeof(int32_t)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_dc_est,
                             static_cast<size_t>(M) * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_fast_ctr,
                             static_cast<size_t>(M) * sizeof(int32_t)));

    cudaMemset(state.d_mm_sym_count, 0, static_cast<size_t>(M) * sizeof(int32_t));
    cudaMemset(state.d_mm_dc_est,    0, static_cast<size_t>(M) * sizeof(float));
    cudaMemset(state.d_mm_fast_ctr,  0, static_cast<size_t>(M) * sizeof(int32_t));

    // Phase 2 state
    CUDA_CHECK_MM(cudaMalloc(&state.d_channel_mode,
                             static_cast<size_t>(M) * sizeof(int8_t)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_p2_iq_last,
                             static_cast<size_t>(3 * M) * sizeof(cufftComplex)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_p2_last_sym,
                             static_cast<size_t>(M) * sizeof(cufftComplex)));

    cudaMemset(state.d_channel_mode, 1, static_cast<size_t>(M) * sizeof(int8_t));  // default Phase1
    cudaMemset(state.d_p2_iq_last,   0, static_cast<size_t>(3 * M) * sizeof(cufftComplex));
    cudaMemset(state.d_p2_last_sym,  0, static_cast<size_t>(M) * sizeof(cufftComplex));

    return true;
}

void mm_process(ChannelizerState& state, int in_steps)
{
    const int M = state.config.num_phases;

    const float sps_p1 = state.config.channel_bw_hz / 4800.0f;
    const float sps_p2 = state.config.channel_bw_hz / 6000.0f;

    constexpr int CHAN_W = 512;
    mm_recovery_kernel<<<(M + CHAN_W - 1) / CHAN_W, CHAN_W>>>(
        state.d_fm_filtered,
        state.d_s2_output,
        state.d_mm_mu,
        state.d_mm_omega,
        state.d_mm_fm_last,
        state.d_p2_iq_last,
        state.d_mm_last_interp,   // repurposed as "last raw symbol" for Phase 1 Gardner
        state.d_p2_last_sym,
        state.d_mm_dc_est,
        state.d_channel_mode,
        state.d_mm_symbols,
        state.d_mm_dibits,
        state.d_mm_sym_count,
        state.d_mm_fast_ctr,
        M, in_steps, sps_p1, sps_p2);
}

void mm_free(ChannelizerState& state)
{
    if (state.d_mm_mu)          { cudaFree(state.d_mm_mu);          state.d_mm_mu          = nullptr; }
    if (state.d_mm_fm_last)     { cudaFree(state.d_mm_fm_last);     state.d_mm_fm_last     = nullptr; }
    if (state.d_mm_last_interp) { cudaFree(state.d_mm_last_interp); state.d_mm_last_interp = nullptr; }
    if (state.d_mm_last_dec)    { cudaFree(state.d_mm_last_dec);    state.d_mm_last_dec    = nullptr; }
    if (state.d_mm_omega)       { cudaFree(state.d_mm_omega);       state.d_mm_omega       = nullptr; }
    if (state.d_mm_symbols)     { cudaFree(state.d_mm_symbols);     state.d_mm_symbols     = nullptr; }
    if (state.d_mm_dibits)      { cudaFree(state.d_mm_dibits);      state.d_mm_dibits      = nullptr; }
    if (state.d_mm_sym_count)   { cudaFree(state.d_mm_sym_count);   state.d_mm_sym_count   = nullptr; }
    if (state.d_mm_dc_est)      { cudaFree(state.d_mm_dc_est);      state.d_mm_dc_est      = nullptr; }
    if (state.d_mm_fast_ctr)    { cudaFree(state.d_mm_fast_ctr);    state.d_mm_fast_ctr    = nullptr; }
    if (state.d_channel_mode)   { cudaFree(state.d_channel_mode);   state.d_channel_mode   = nullptr; }
    if (state.d_p2_iq_last)     { cudaFree(state.d_p2_iq_last);     state.d_p2_iq_last     = nullptr; }
    if (state.d_p2_last_sym)    { cudaFree(state.d_p2_last_sym);    state.d_p2_last_sym    = nullptr; }
}
