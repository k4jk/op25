// Mueller-Müller clock recovery + 4-level C4FM slicer — Milestone 3
//
// Processes FM-demodulated output for all M channels in parallel.
// One thread per channel; the symbol timing loop within each thread is sequential
// (M&M feedback is inherently causal).
//
// Grid: (1,)   Block: (M,)
//
// Cross-block continuity:
//   d_mm_mu          — fractional timing carry-over (may be negative, up to −(sps−1))
//   d_mm_fm_last     — last FM sample of the previous block, for interpolation when
//                      mu < 0 at block start
//   d_mm_last_interp — previous interpolated symbol value (M&M error numerator)
//   d_mm_last_dec    — previous slicer decision value (M&M error denominator)
//
// P25 C4FM deviation (Phase 1):
//   Outer symbols ±3: ±1800 Hz → ±1800·2π/12500 = ±0.9048 rad/samp
//   Inner symbols ±1:  ±600 Hz →  ±600·2π/12500 = ±0.3016 rad/samp
//   Decision threshold (midpoint):  (0.9048+0.3016)/2 = 0.6032 rad/samp
//
// Dibit encoding (P25 standard, NID/voice superframe order):
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

// Map symbol value to dibit (0-3).
static __device__ int8_t symbol_to_dibit(float d)
{
    if (d > +2.0f) return 1;   // +3 → 01
    if (d > 0.0f)  return 0;   // +1 → 00
    if (d > -2.0f) return 2;   // -1 → 10
    return 3;                   // -3 → 11
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void mm_recovery_kernel(
    const float* __restrict__ d_fm,    // [in_steps * M] FM demod output, step-major
    float*   d_mu,                     // [M] fractional timing state (in/out)
    float*   d_fm_last,                // [M] last FM sample from previous block (in/out)
    float*   d_last_interp,            // [M] previous interpolated symbol (in/out)
    float*   d_last_dec,               // [M] previous slicer decision (in/out)
    float*   d_symbols,                // [MM_MAX_SYM * M] output symbol values
    int8_t*  d_dibits,                 // [MM_MAX_SYM * M] output dibits
    int32_t* d_sym_count,              // [M] output: valid symbols per channel
    int M, int in_steps,
    float sps,           // nominal samples-per-symbol
    float mm_gain)       // M&M loop gain
{
    const int k = static_cast<int>(threadIdx.x);
    if (k >= M) return;

    float mu       = d_mu[k];
    float fm_last  = d_fm_last[k];
    float prev_y   = d_last_interp[k];
    float prev_d   = d_last_dec[k];
    int   n_sym    = 0;

    // Fetch FM sample at absolute position i within this block.
    // i == -1 uses the saved last sample from the previous block.
    auto get_fm = [&](int i) -> float {
        if (i < 0) return fm_last;
        return d_fm[i * M + k];
    };

    // mu is the absolute (float) position in the current block at which the
    // next symbol should be sampled.  It may start negative (carry-over from
    // the previous block) and advances by sps (+/- M&M correction) per symbol.
    while (n_sym < MM_MAX_SYM) {
        int i0 = static_cast<int>(floorf(mu));
        int i1 = i0 + 1;

        // Stop when we can't form a complete interpolation within this block.
        // The un-consumed portion of mu is carried into the next block.
        if (i1 >= in_steps) break;

        float frac = mu - static_cast<float>(i0);
        float s0   = get_fm(i0);
        float s1   = get_fm(i1);
        float y    = s0 + frac * (s1 - s0);   // linear interpolation
        float d    = decide_c4fm(y);

        // Mueller-Müller timing error: e = y[k-1]·d[k] - y[k]·d[k-1]
        float e = prev_y * d - y * prev_d;
        // Clamp to prevent loop divergence during initial acquisition
        if (e >  1.0f) e =  1.0f;
        if (e < -1.0f) e = -1.0f;

        d_symbols[n_sym * M + k] = y;
        d_dibits [n_sym * M + k] = symbol_to_dibit(d);
        n_sym++;

        prev_y  = y;
        prev_d  = d;
        mu     += sps + mm_gain * e;
    }

    // Carry timing state forward to the next block.
    // d_mm_mu holds the position relative to the START of the NEXT block,
    // which may be negative (we still owe some samples from the current block).
    d_mu[k]          = mu - static_cast<float>(in_steps);
    d_fm_last[k]     = get_fm(in_steps - 1);
    d_last_interp[k] = prev_y;
    d_last_dec[k]    = prev_d;
    d_sym_count[k]   = n_sym;
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
                             static_cast<size_t>(M) * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_last_interp,
                             static_cast<size_t>(M) * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_last_dec,
                             static_cast<size_t>(M) * sizeof(float)));

    // Start mu at sps/2 so the first strobe lands at the centre of the first symbol.
    // We don't know sps yet at allocation time; initialise to 0 and let mm_process
    // set it on the first call via cudaMemcpy.
    cudaMemset(state.d_mm_mu,          0, static_cast<size_t>(M) * sizeof(float));
    cudaMemset(state.d_mm_fm_last,     0, static_cast<size_t>(M) * sizeof(float));
    cudaMemset(state.d_mm_last_interp, 0, static_cast<size_t>(M) * sizeof(float));
    cudaMemset(state.d_mm_last_dec,    0, static_cast<size_t>(M) * sizeof(float));

    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_symbols,
                             static_cast<size_t>(MM_MAX_SYM) * M * sizeof(float)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_dibits,
                             static_cast<size_t>(MM_MAX_SYM) * M * sizeof(int8_t)));
    CUDA_CHECK_MM(cudaMalloc(&state.d_mm_sym_count,
                             static_cast<size_t>(M) * sizeof(int32_t)));

    cudaMemset(state.d_mm_sym_count, 0, static_cast<size_t>(M) * sizeof(int32_t));

    return true;
}

void mm_process(ChannelizerState& state, int in_steps, P25Mode mode)
{
    const int M = state.config.num_phases;

    // Samples-per-symbol at the channel output rate (12.5 kHz).
    //   Phase 1: 12500 / 4800 = 2.6042  (C4FM, 4-level FM)
    //   Phase 2: 12500 / 6000 = 2.0833  (TDMA H-DQPSK — timing only here)
    const float sps = (mode == P25Mode::PHASE2)
                      ? (state.config.channel_bw_hz / 6000.0f)
                      : (state.config.channel_bw_hz / 4800.0f);

    // M&M loop gain.  A small gain keeps jitter low once locked;
    // larger values acquire faster but add steady-state jitter.
    const float mm_gain = 0.01f;

    mm_recovery_kernel<<<1, M>>>(
        state.d_fm_output,
        state.d_mm_mu,
        state.d_mm_fm_last,
        state.d_mm_last_interp,
        state.d_mm_last_dec,
        state.d_mm_symbols,
        state.d_mm_dibits,
        state.d_mm_sym_count,
        M, in_steps, sps, mm_gain);
}

void mm_free(ChannelizerState& state)
{
    if (state.d_mm_mu)          { cudaFree(state.d_mm_mu);          state.d_mm_mu          = nullptr; }
    if (state.d_mm_fm_last)     { cudaFree(state.d_mm_fm_last);     state.d_mm_fm_last     = nullptr; }
    if (state.d_mm_last_interp) { cudaFree(state.d_mm_last_interp); state.d_mm_last_interp = nullptr; }
    if (state.d_mm_last_dec)    { cudaFree(state.d_mm_last_dec);    state.d_mm_last_dec    = nullptr; }
    if (state.d_mm_symbols)     { cudaFree(state.d_mm_symbols);     state.d_mm_symbols     = nullptr; }
    if (state.d_mm_dibits)      { cudaFree(state.d_mm_dibits);      state.d_mm_dibits      = nullptr; }
    if (state.d_mm_sym_count)   { cudaFree(state.d_mm_sym_count);   state.d_mm_sym_count   = nullptr; }
}
