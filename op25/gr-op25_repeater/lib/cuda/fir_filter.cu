// Copyright 2026, James Kirkham K4JK
// SPDX-License-Identifier: GPL-3.0-or-later

// Stage-2 per-channel decimating FIR filter — Milestone 2
//
// Takes the fft_size-channel stage-1 output (step-major, at stage1_output_rate_hz)
// and decimates each channel independently by D = stage2_decimation,
// producing fft_size channels at channel_bw_hz.
//
// fft_size = num_phases * fft_oversample.  With fft_oversample=1 (default)
// fft_size == num_phases and behaviour is identical to the original design.
// With fft_oversample=2 the filter bank doubles in channel count, enabling
// 6.25 kHz bin spacing while preserving 12.5 kHz per-channel bandwidth.
//
// Memory layout (C = fft_size):
//   d_s2_padded:  [(N2-1 + L) * C]  cufftComplex, step-major
//   d_s2_output:  [(L/D) * C]        cufftComplex, step-major

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

// Grid: (ceil(C/512), L/D)   Block: (512, 1)   where C = fft_size
__global__ void s2_decimate_kernel(
    const cufftComplex* __restrict__ d_padded,   // [(N2-1+L)*C], step-major
    const float*        __restrict__ d_coeff,    // [N2]
    cufftComplex*                    d_output,   // [(L/D)*C], step-major
    int C, int D, int N2)
{
    const int chan     = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                       + static_cast<int>(threadIdx.x);
    const int out_step = static_cast<int>(blockIdx.y);
    if (chan >= C) return;

    float re = 0.0f, im = 0.0f;
    int base = (N2 - 1 + out_step * D) * C + chan;

    for (int k = 0; k < N2; k++) {
        cufftComplex s = d_padded[base - k * C];
        re += d_coeff[k] * s.x;
        im += d_coeff[k] * s.y;
    }

    d_output[out_step * C + chan] = {re, im};
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool s2_filter_alloc(ChannelizerState& state, const std::vector<float>& h)
{
    const int num_phases = state.config.num_phases;
    const int C  = state.config.fft_size;   // channel count (= num_phases * fft_oversample)
    const int L  = state.input_len / num_phases;
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

    // Padded buffer: [(N2-1 + L) * C], zeroed
    size_t padded_sz = static_cast<size_t>(N2 - 1 + L) * C;
    CUDA_CHECK_S2(cudaMalloc(&state.d_s2_padded, padded_sz * sizeof(cufftComplex)));
    CUDA_CHECK_S2(cudaMemset(state.d_s2_padded, 0, padded_sz * sizeof(cufftComplex)));

    // Decimated output: [(L/D) * C]
    CUDA_CHECK_S2(cudaMalloc(&state.d_s2_output,
                             static_cast<size_t>(L / D) * C * sizeof(cufftComplex)));

    return true;
}

int s2_filter_process(ChannelizerState& state, const cufftComplex* d_s1_output)
{
    const int num_phases = state.config.num_phases;
    const int C  = state.config.fft_size;
    const int L  = state.input_len / num_phases;
    const int D  = state.config.stage2_decimation;
    const int N2 = state.s2_filter_len;

    // Shift history: copy last (N2-1)*C samples into history slot.
    size_t history_bytes = static_cast<size_t>(N2 - 1) * C * sizeof(cufftComplex);
    if (N2 > 1) {
        cudaMemcpy(state.d_s2_padded,
                   state.d_s2_padded + static_cast<size_t>(L) * C,
                   history_bytes,
                   cudaMemcpyDeviceToDevice);
    }

    // Write new stage-1 output into the new-data region
    cudaMemcpy(state.d_s2_padded + static_cast<size_t>(N2 - 1) * C,
               d_s1_output,
               static_cast<size_t>(L) * C * sizeof(cufftComplex),
               cudaMemcpyDeviceToDevice);

    int out_steps = L / D;
    constexpr int CHAN_W = 512;
    dim3 s2_block(CHAN_W);
    dim3 s2_grid((C + CHAN_W - 1) / CHAN_W, out_steps);
    s2_decimate_kernel<<<s2_grid, s2_block>>>(
        state.d_s2_padded,
        state.d_s2_coeff,
        state.d_s2_output,
        C, D, N2);

    return out_steps;
}

void s2_filter_free(ChannelizerState& state)
{
    if (state.d_s2_coeff)  { cudaFree(state.d_s2_coeff);  state.d_s2_coeff  = nullptr; }
    if (state.d_s2_padded) { cudaFree(state.d_s2_padded); state.d_s2_padded = nullptr; }
    if (state.d_s2_output) { cudaFree(state.d_s2_output); state.d_s2_output = nullptr; }
}
