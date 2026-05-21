// CUDA polyphase analysis channelizer — Milestone 1
//
// Algorithm:
//   1. poly_filter_kernel: for each output step s, thread k computes
//      y_k[s] = sum_{m=0}^{K-1}  poly[k][m] * x[(s-m)*M + k]
//      where x[] is the padded input (history prepended).
//   2. cuFFT forward fft_size-point batch FFT across the polyphase outputs
//      at each step.  With fft_oversample=1: fft_size==M (critically sampled).
//      With fft_oversample=2: fft_size==2M — the M polyphase outputs are
//      zero-padded to 2M, giving bins at channel_bw/2 spacing.
//   3. decarrier_process(): for fft_oversample>1 only — removes the implicit
//      Nyquist carrier (-1)^step from odd-bin outputs before FM demodulation.
//
// Memory layout conventions (C = fft_size = M * fft_oversample):
//   d_input  (padded):  [(K-1)*M  +  L*M]  cufftComplex, row-major
//   d_poly_phases:      [M * K]             float, row-major (phase × tap)
//   d_phase_out:        [L * C]             cufftComplex, step-major
//   d_output (caller):  [L * C]             cufftComplex, step-major (FFT result)

#include "../../include/cuda/channelizer.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define CUFFT_CHECK(call)                                                   \
    do {                                                                    \
        cufftResult _e = (call);                                            \
        if (_e != CUFFT_SUCCESS) {                                          \
            fprintf(stderr, "cuFFT error %s:%d  code=%d\n",                \
                    __FILE__, __LINE__, (int)_e);                           \
            return false;                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Polyphase FIR filter kernel
//
// Grid:  (ceil(M/512), L)  — 2D: x=channel groups, y=time steps
// Block: (512, 1)
//
// d_input_padded: [(K-1 + L) * M] complex samples (M = num_phases)
//                  index of phase k, step s, tap m: (K-1 + s - m) * M + k
//
// d_phase_out: [L * fft_size] — stride is fft_size (>= M).
//   With fft_oversample=1: fft_size == M, layout identical to before.
//   With fft_oversample=2: fft_size == 2M; entries [M..2M-1] per step are
//   zeroed at alloc and never written here, acting as zero-padding for the
//   oversampled FFT that follows.
// ---------------------------------------------------------------------------
__global__ void poly_filter_kernel(
    const cufftComplex* __restrict__ d_input_padded,
    const float*        __restrict__ d_poly,
    cufftComplex*                    d_phase_out,
    int M, int K, int fft_size)
{
    // 2D grid: blockIdx.x = channel group, blockIdx.y = time step.
    const int phase = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                    + static_cast<int>(threadIdx.x);
    const int step  = static_cast<int>(blockIdx.y);
    if (phase >= M) return;

    float re = 0.0f, im = 0.0f;
    int   base = (K - 1 + step) * M + phase;

    for (int m = 0; m < K; m++) {
        cufftComplex s_in = d_input_padded[base - m * M];
        float        coef = d_poly[phase * K + m];
        re += coef * s_in.x;
        im += coef * s_in.y;
    }

    // Write with fft_size stride; zero-padding occupies [M..fft_size-1] per step.
    d_phase_out[step * fft_size + phase] = {re, im};
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool channelizer_alloc(ChannelizerState& state,
                       const std::vector<float>& proto,
                       const std::vector<float>& poly,
                       int input_len)
{
    const ChannelizerConfig& cfg = state.config;
    const int M        = cfg.num_phases;
    const int K        = cfg.taps_per_phase;
    const int fft_size = cfg.fft_size;   // M * fft_oversample
    const int L        = input_len / M;  // output steps per block

    if (input_len % M != 0) {
        fprintf(stderr, "channelizer_alloc: input_len (%d) must be a multiple of M (%d)\n",
                input_len, M);
        return false;
    }
    if (static_cast<int>(poly.size()) != M * K) {
        fprintf(stderr, "channelizer_alloc: poly size %zu expected %d\n",
                poly.size(), M * K);
        return false;
    }

    state.input_len = input_len;

    // Prototype filter (reference copy on GPU, for potential future use)
    CUDA_CHECK(cudaMalloc(&state.d_proto_filter, proto.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_proto_filter, proto.data(),
                          proto.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Polyphase coefficient matrix [M * K]
    CUDA_CHECK(cudaMalloc(&state.d_poly_phases, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_poly_phases, poly.data(),
                          M * K * sizeof(float), cudaMemcpyHostToDevice));

    // Padded input buffer: (K-1)*M history + L*M new input, zeroed initially
    size_t padded_samples = static_cast<size_t>(K - 1 + L) * M;
    CUDA_CHECK(cudaMalloc(&state.d_input, padded_samples * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemset(state.d_input, 0, padded_samples * sizeof(cufftComplex)));

    // Phase filter output buffer: [L * fft_size], step-major.
    // Zeroed once here; poly_filter_kernel writes only phases [0..M-1] per step.
    // Entries [M..fft_size-1] stay zero, acting as zero-padding for the FFT when
    // fft_oversample > 1 (oversampled filter bank for sub-channel-bw bin spacing).
    CUDA_CHECK(cudaMalloc(&state.d_phase_out,
                          static_cast<size_t>(L) * fft_size * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemset(state.d_phase_out, 0,
                          static_cast<size_t>(L) * fft_size * sizeof(cufftComplex)));

    // cuFFT plan: fft_size-point C2C forward FFT, batched L times.
    // With fft_oversample=1: fft_size==M, critically sampled — bin spacing = channel_bw_hz.
    // With fft_oversample=2: fft_size==2M, zero-padded  — bin spacing = channel_bw_hz/2.
    int n_fft = fft_size;  // cufftPlanMany requires a mutable int*
    CUFFT_CHECK(cufftPlanMany(
        &state.fft_plan,
        1,                    // rank (1D)
        &n_fft,               // n: FFT length
        nullptr, 1, fft_size, // inembed, istride, idist
        nullptr, 1, fft_size, // onembed, ostride, odist
        CUFFT_C2C,
        L                     // batch
    ));

    return true;
}

// Process one block of new input samples.
//
// d_new_input:  [input_len] new complex samples on GPU
// d_output:     [input_len] complex output on GPU, layout [step * M + channel]
//               Channel j at output step s: d_output[s * M + j]
// Returns:      number of output samples per channel (= input_len / M)
int channelizer_process(ChannelizerState& state,
                        const cufftComplex* d_new_input,
                        cufftComplex* d_output)
{
    const ChannelizerConfig& cfg = state.config;
    const int M        = cfg.num_phases;
    const int K        = cfg.taps_per_phase;
    const int fft_size = cfg.fft_size;
    const int L        = state.input_len / M;

    // Shift history: copy old new-input tail into the history region.
    // Padded buffer layout: [history (K-1)*M | new input L*M]
    size_t history_bytes   = static_cast<size_t>(K - 1) * M * sizeof(cufftComplex);
    size_t tail_src_offset = static_cast<size_t>(L - (K - 1)) * M;

    if (K > 1) {
        cudaMemcpy(state.d_input,
                   state.d_input + tail_src_offset + (K - 1) * M,
                   history_bytes,
                   cudaMemcpyDeviceToDevice);
    }

    // Write new input into the padded buffer after the history
    cudaMemcpy(state.d_input + static_cast<size_t>(K - 1) * M,
               d_new_input,
               static_cast<size_t>(state.input_len) * sizeof(cufftComplex),
               cudaMemcpyDeviceToDevice);

    // Polyphase FIR filter: writes to d_phase_out with fft_size stride.
    // Grid: (ceil(M/512), L)   Block: (512, 1)
    constexpr int CHAN_W = 512;
    dim3 pf_block(CHAN_W);
    dim3 pf_grid((M + CHAN_W - 1) / CHAN_W, L);
    poly_filter_kernel<<<pf_grid, pf_block>>>(
        state.d_input,
        state.d_poly_phases,
        state.d_phase_out,
        M, K, fft_size);

    // fft_size-point forward FFT batched over L steps.
    // d_phase_out layout: [L * fft_size], with zero-padding in [M..fft_size-1] per step.
    cufftExecC2C(state.fft_plan, state.d_phase_out, d_output, CUFFT_FORWARD);

    return L;
}

void channelizer_free(ChannelizerState& state)
{
    if (state.d_proto_filter) { cudaFree(state.d_proto_filter); state.d_proto_filter = nullptr; }
    if (state.d_poly_phases)  { cudaFree(state.d_poly_phases);  state.d_poly_phases  = nullptr; }
    if (state.d_input)        { cudaFree(state.d_input);        state.d_input        = nullptr; }
    if (state.d_phase_out)    { cudaFree(state.d_phase_out);    state.d_phase_out    = nullptr; }
    if (state.fft_plan)       { cufftDestroy(state.fft_plan);   state.fft_plan       = 0;       }
}

// ---------------------------------------------------------------------------
// Nyquist carrier removal for oversampled (zero-padded) FFT
//
// For fft_oversample > 1, the zero-padded 2M FFT output for odd bin k=(2j+1)
// carries an implicit Nyquist carrier exp(jπ·k·step) = (-1)^step (since k is
// odd and exp(j·2π·j·step)=1 for integer j).  This ±π offset in the FM
// demodulated output maps all P25 C4FM symbols to wrong dibits, preventing
// frame sync.  Negating odd-bin outputs at odd steps removes the carrier.
//
// The batch size L=80 is always even, so the carrier phase resets to +1 at
// every batch boundary — no inter-batch state is needed.
//
// Grid: (ceil(n_odd_bins / CHAN_W), L)   Block: (CHAN_W, 1)
// n_odd_bins = fft_size / 2
// ---------------------------------------------------------------------------

__global__ void decarrier_kernel(cufftComplex* __restrict__ d_data,
                                  int fft_size)
{
    // Thread j handles odd bin k = 2j+1
    const int j    = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockIdx.y);
    const int k    = 2 * j + 1;
    if (k >= fft_size) return;

    if (step & 1) {
        const int idx = step * fft_size + k;
        d_data[idx].x = -d_data[idx].x;
        d_data[idx].y = -d_data[idx].y;
    }
}

void decarrier_process(ChannelizerState& state, cufftComplex* d_output, int L)
{
    const int fft_size   = state.config.fft_size;
    const int num_phases = state.config.num_phases;

    static int decarrier_call_count = 0;
    if (++decarrier_call_count <= 3)
        fprintf(stderr, "decarrier_process: call#%d fft_size=%d num_phases=%d L=%d ptr=%p %s\n",
                decarrier_call_count, fft_size, num_phases, L, (void*)d_output,
                (fft_size == num_phases) ? "SKIPPING(critically sampled)" : "RUNNING");

    if (fft_size == num_phases) return;  // critically sampled: no odd bins

    const int n_odd = fft_size / 2;
    constexpr int CHAN_W = 512;
    dim3 blk(CHAN_W);
    dim3 grd((n_odd + CHAN_W - 1) / CHAN_W, L);
    decarrier_kernel<<<grd, blk>>>(d_output, fft_size);
}
