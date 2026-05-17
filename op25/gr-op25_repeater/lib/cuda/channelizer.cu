// CUDA polyphase analysis channelizer — Milestone 1
//
// Algorithm:
//   1. poly_filter_kernel: for each output step s, thread k computes
//      y_k[s] = sum_{m=0}^{K-1}  poly[k][m] * x[(s-m)*M + k]
//      where x[] is the padded input (history prepended).
//   2. cuFFT forward M-point batch FFT across the M phase outputs at each step.
//      Output bin j of step s is the baseband signal for channel j.
//
// Memory layout conventions:
//   d_input  (padded):  [(K-1)*M  +  L*M]  cufftComplex, row-major
//                        history              new input
//   d_poly_phases:      [M * K]              float, row-major (phase × tap)
//   d_phase_out:        [L * M]              cufftComplex, step-major
//   d_channel_out:      [L * M]              cufftComplex, step-major
//                        (FFT output in-place into d_phase_out)

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
// Grid:  (L,)  — one block per output time step
// Block: (M,)  — one thread per phase
//
// d_input_padded: [(K-1 + L) * M] complex samples
//                  index of phase k, step s, tap m:
//                  (K-1 + s - m) * M + k
// ---------------------------------------------------------------------------
__global__ void poly_filter_kernel(
    const cufftComplex* __restrict__ d_input_padded,
    const float*        __restrict__ d_poly,
    cufftComplex*                    d_phase_out,
    int M, int K)
{
    // 2D grid: blockIdx.x = channel group, blockIdx.y = time step.
    // Supports M > 1024 (CUDA max threads/block).
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

    d_phase_out[step * M + phase] = {re, im};
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
    const int M = cfg.num_phases;
    const int K = cfg.taps_per_phase;
    const int L = input_len / M;    // output steps per block

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

    // Intermediate phase filter output + cuFFT output (in-place), [L * M]
    CUDA_CHECK(cudaMalloc(&state.d_phase_out,
                          static_cast<size_t>(L) * M * sizeof(cufftComplex)));

    // cuFFT plan: M-point C2C forward FFT, batched L times
    // Input/output layout: d_phase_out[step * M + phase] — row-major
    int fft_size = M;   // cufftPlanMany requires a mutable int*
    CUFFT_CHECK(cufftPlanMany(
        &state.fft_plan,
        1,              // rank (1D)
        &fft_size,      // n: FFT length
        nullptr, 1, M,  // inembed, istride, idist (contiguous rows)
        nullptr, 1, M,  // onembed, ostride, odist
        CUFFT_C2C,
        L               // batch
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
    const int M = cfg.num_phases;
    const int K = cfg.taps_per_phase;
    const int L = state.input_len / M;

    // Shift history: copy old new-input tail into the history region.
    // Padded buffer layout: [history (K-1)*M | new input L*M]
    // After each block, slide the last (K-1)*M samples of new input into history.
    size_t history_bytes = static_cast<size_t>(K - 1) * M * sizeof(cufftComplex);
    size_t tail_src_offset = static_cast<size_t>(L - (K - 1)) * M; // last (K-1)*M of new input

    if (K > 1) {
        // Source: last (K-1)*M samples from the *previous* block's new input,
        // which is currently at d_input[(K-1)*M .. (K-1+L)*M - 1].
        // We want to shift the tail of the old new-input into history for next block.
        // Simplest: copy from the existing padded buffer's tail into history slot.
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

    // Polyphase FIR filter: 2D grid supports M > 1024 threads/block.
    // Grid: (ceil(M/512), L)   Block: (512, 1)
    constexpr int CHAN_W = 512;
    dim3 pf_block(CHAN_W);
    dim3 pf_grid((M + CHAN_W - 1) / CHAN_W, L);
    poly_filter_kernel<<<pf_grid, pf_block>>>(
        state.d_input,
        state.d_poly_phases,
        state.d_phase_out,
        M, K);

    // M-point forward FFT across phases at each output step — in-place
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
