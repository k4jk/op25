// FM discriminator kernel — Milestone 2
//
// Differential phase demodulator for all M channels in parallel.
// Input:  cufftComplex [out_steps * M], step-major (state.d_s2_output)
// Output: float [out_steps * M], step-major (state.d_fm_output)
//
// demod[n][k] = atan2f( Im(x[n][k] * conj(x[n-1][k])),
//                        Re(x[n][k] * conj(x[n-1][k])) )
//
// = atan2f( x[n].im * x[n-1].re - x[n].re * x[n-1].im,
//           x[n].re * x[n-1].re + x[n].im * x[n-1].im )
//
// The previous sample across blocks is maintained in state.d_fm_prev [M].
//
// Grid: (out_steps, 1)   Block: (M, 1)

#include "fm_demod.h"
#include <cstdio>

#define CUDA_CHECK_FM(call)                                                  \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            return;                                                          \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void fm_demod_kernel(
    const cufftComplex* __restrict__ d_input,   // [out_steps * M], step-major
    const cufftComplex* __restrict__ d_prev,    // [M] previous sample per channel
    float*                           d_output,  // [out_steps * M], step-major
    cufftComplex*                    d_prev_out,// [M] updated previous sample (last step)
    int M, int out_steps)
{
    const int chan = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockIdx.x);

    // Load current and previous samples
    cufftComplex curr = d_input[step * M + chan];
    cufftComplex prev = (step == 0) ? d_prev[chan] : d_input[(step - 1) * M + chan];

    // Cross product: curr * conj(prev)
    float re =  curr.x * prev.x + curr.y * prev.y;
    float im =  curr.y * prev.x - curr.x * prev.y;

    d_output[step * M + chan] = atan2f(im, re);

    // Store last sample as the "previous" for next block
    if (step == out_steps - 1)
        d_prev_out[chan] = curr;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool fm_demod_alloc(ChannelizerState& state)
{
    const int M = state.config.num_phases;
    const int L = state.input_len / M;
    const int D = state.config.stage2_decimation;
    const int out_steps = L / D;

    cudaError_t e;

    e = cudaMalloc(&state.d_fm_prev, M * sizeof(cufftComplex));
    if (e != cudaSuccess) {
        fprintf(stderr, "fm_demod_alloc d_fm_prev: %s\n", cudaGetErrorString(e));
        return false;
    }
    cudaMemset(state.d_fm_prev, 0, M * sizeof(cufftComplex));

    e = cudaMalloc(&state.d_fm_output,
                   static_cast<size_t>(out_steps) * M * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "fm_demod_alloc d_fm_output: %s\n", cudaGetErrorString(e));
        return false;
    }

    return true;
}

void fm_demod_process(ChannelizerState& state, int out_steps)
{
    const int M = state.config.num_phases;

    fm_demod_kernel<<<out_steps, M>>>(
        state.d_s2_output,
        state.d_fm_prev,
        state.d_fm_output,
        state.d_fm_prev,   // updated in-place: last step writes here
        M, out_steps);
}

void fm_demod_free(ChannelizerState& state)
{
    if (state.d_fm_prev)   { cudaFree(state.d_fm_prev);   state.d_fm_prev   = nullptr; }
    if (state.d_fm_output) { cudaFree(state.d_fm_output); state.d_fm_output = nullptr; }
}
