// Stage-2 per-channel decimating FIR filter — Milestone 2
//
// Takes the M-channel stage-1 output (step-major, at stage1_output_rate_hz)
// and decimates each channel independently by D = stage2_decimation,
// producing M channels at channel_bw_hz.
//
// Memory layout:
//   d_s2_padded:  [(N2-1 + L) * M]  cufftComplex, step-major
//                  First (N2-1)*M samples are the per-channel FIR history.
//                  Next L*M samples are the current stage-1 output.
//   d_s2_output:  [(L/D) * M]        cufftComplex, step-major
//
// Kernel: one thread per (output step, channel).
//   Grid: (L/D, 1), Block: (M, 1)
//   Each thread computes one output sample by dot-producting the FIR
//   coefficients against N2 consecutive (stride-M) input samples.

#include "fir_filter.h"
#include "proto_fir.h"    // for windowed-sinc helpers (reused)

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK_S2(call)                                                  \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            return false;                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Stage-2 FIR design (windowed sinc, same approach as prototype filter)
// ---------------------------------------------------------------------------

static float s2_sinc(float x) {
    if (fabsf(x) < 1e-9f) return 1.0f;
    return sinf(static_cast<float>(M_PI) * x) / (static_cast<float>(M_PI) * x);
}

static float s2_blackman_harris(int n, int N) {
    const float a0 = 0.35875f, a1 = 0.48829f, a2 = 0.14128f, a3 = 0.01168f;
    float t = 2.0f * static_cast<float>(M_PI) * n / (N - 1);
    return a0 - a1 * cosf(t) + a2 * cosf(2.0f * t) - a3 * cosf(3.0f * t);
}

void design_s2_filter(const ChannelizerConfig& cfg, int n_taps, std::vector<float>& h)
{
    const float fc = 0.5f / static_cast<float>(cfg.stage2_decimation);
    const float center = (n_taps - 1) / 2.0f;
    h.resize(n_taps);
    for (int n = 0; n < n_taps; n++) {
        float m = n - center;
        h[n] = 2.0f * fc * s2_sinc(2.0f * fc * m) * s2_blackman_harris(n, n_taps);
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------

// Grid: (L/D, 1)   Block: (M, 1)
// Thread (chan, out_step) computes:
//   y[out_step][chan] = sum_{k=0}^{N2-1}  coeff[k] * padded[(N2-1 + out_step*D - k)*M + chan]
//
// The padded buffer layout (step-major) means consecutive channels for the same
// step are contiguous — stride-M access for consecutive taps is unavoidable but
// the total number of taps (N2=128) limits the bandwidth impact.
__global__ void s2_decimate_kernel(
    const cufftComplex* __restrict__ d_padded,   // [(N2-1+L)*M], step-major
    const float*        __restrict__ d_coeff,    // [N2]
    cufftComplex*                    d_output,   // [(L/D)*M], step-major
    int M, int D, int N2)
{
    // 2D grid: blockIdx.x = channel group, blockIdx.y = output step.
    const int chan     = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                       + static_cast<int>(threadIdx.x);
    const int out_step = static_cast<int>(blockIdx.y);
    if (chan >= M) return;

    float re = 0.0f, im = 0.0f;
    // base index in padded array for this (out_step, tap=0): (N2-1 + out_step*D) * M + chan
    int base = (N2 - 1 + out_step * D) * M + chan;

    for (int k = 0; k < N2; k++) {
        cufftComplex s = d_padded[base - k * M];   // stride-M tap access
        re += d_coeff[k] * s.x;
        im += d_coeff[k] * s.y;
    }

    d_output[out_step * M + chan] = {re, im};
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool s2_filter_alloc(ChannelizerState& state, const std::vector<float>& h)
{
    const int M  = state.config.num_phases;
    const int L  = state.input_len / M;
    const int D  = state.config.stage2_decimation;
    const int N2 = static_cast<int>(h.size());

    if (L % D != 0) {
        fprintf(stderr, "s2_filter_alloc: L (%d) must be divisible by D (%d)\n", L, D);
        return false;
    }

    state.s2_filter_len = N2;

    CUDA_CHECK_S2(cudaMalloc(&state.d_s2_coeff, N2 * sizeof(float)));
    CUDA_CHECK_S2(cudaMemcpy(state.d_s2_coeff, h.data(),
                             N2 * sizeof(float), cudaMemcpyHostToDevice));

    // Padded buffer: [(N2-1 + L) * M], zeroed (history starts at zero)
    size_t padded_sz = static_cast<size_t>(N2 - 1 + L) * M;
    CUDA_CHECK_S2(cudaMalloc(&state.d_s2_padded, padded_sz * sizeof(cufftComplex)));
    CUDA_CHECK_S2(cudaMemset(state.d_s2_padded, 0, padded_sz * sizeof(cufftComplex)));

    // Decimated output: [(L/D) * M]
    CUDA_CHECK_S2(cudaMalloc(&state.d_s2_output,
                             static_cast<size_t>(L / D) * M * sizeof(cufftComplex)));

    return true;
}

int s2_filter_process(ChannelizerState& state, const cufftComplex* d_s1_output)
{
    const int M  = state.config.num_phases;
    const int L  = state.input_len / M;
    const int D  = state.config.stage2_decimation;
    const int N2 = state.s2_filter_len;

    // Shift history: copy last (N2-1)*M samples of current padded new-data into history.
    // Current new-data occupies d_s2_padded[(N2-1)*M .. (N2-1+L)*M - 1].
    // The last (N2-1)*M of that block starts at d_s2_padded[L*M].
    size_t history_bytes = static_cast<size_t>(N2 - 1) * M * sizeof(cufftComplex);
    if (N2 > 1) {
        cudaMemcpy(state.d_s2_padded,
                   state.d_s2_padded + static_cast<size_t>(L) * M,
                   history_bytes,
                   cudaMemcpyDeviceToDevice);
    }

    // Write new stage-1 output into the new-data region
    cudaMemcpy(state.d_s2_padded + static_cast<size_t>(N2 - 1) * M,
               d_s1_output,
               static_cast<size_t>(L) * M * sizeof(cufftComplex),
               cudaMemcpyDeviceToDevice);

    // Decimate: 2D grid supports M > 1024 threads/block.
    // Grid: (ceil(M/512), out_steps)   Block: (512, 1)
    int out_steps = L / D;
    constexpr int CHAN_W = 512;
    dim3 s2_block(CHAN_W);
    dim3 s2_grid((M + CHAN_W - 1) / CHAN_W, out_steps);
    s2_decimate_kernel<<<s2_grid, s2_block>>>(
        state.d_s2_padded,
        state.d_s2_coeff,
        state.d_s2_output,
        M, D, N2);

    return out_steps;
}

void s2_filter_free(ChannelizerState& state)
{
    if (state.d_s2_coeff)  { cudaFree(state.d_s2_coeff);  state.d_s2_coeff  = nullptr; }
    if (state.d_s2_padded) { cudaFree(state.d_s2_padded); state.d_s2_padded = nullptr; }
    if (state.d_s2_output) { cudaFree(state.d_s2_output); state.d_s2_output = nullptr; }
}
