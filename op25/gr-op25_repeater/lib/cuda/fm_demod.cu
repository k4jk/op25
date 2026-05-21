// FM discriminator kernel — Milestone 2
//
// Differential phase demodulator for all M channels in parallel.
// Input:  cufftComplex [out_steps * M], step-major (state.d_s2_output)
// Output: float [out_steps * M], step-major (state.d_fm_output)
//
// demod[n][k] = atan2( Im(x[n][k] * conj(x[n-1][k])),
//                       Re(x[n][k] * conj(x[n-1][k])) )  [double precision]
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

// C = fft_size (channel count)
// d_prev is read-only (no aliasing with d_output).  The caller updates
// d_fm_prev after this kernel returns via a separate cudaMemcpy, which
// serialises the write after all step=0 reads and eliminates the race.
__global__ void fm_demod_kernel(
    const cufftComplex* __restrict__ d_input,  // [out_steps * C], step-major
    const cufftComplex* __restrict__ d_prev,   // [C] previous sample per channel
    float*                           d_output, // [out_steps * C], step-major
    int C)
{
    const int chan = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockIdx.y);
    if (chan >= C) return;

    cufftComplex curr = d_input[step * C + chan];
    cufftComplex prev = (step == 0) ? d_prev[chan] : d_input[(step - 1) * C + chan];

    float re =  curr.x * prev.x + curr.y * prev.y;
    float im =  curr.y * prev.x - curr.x * prev.y;

    d_output[step * C + chan] = static_cast<float>(atan2(static_cast<double>(im),
                                                         static_cast<double>(re)));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool fm_demod_alloc(ChannelizerState& state)
{
    const int num_phases = state.config.num_phases;
    const int C          = state.config.fft_size;
    const int L          = state.input_len / num_phases;
    const int D          = state.config.stage2_decimation;
    const int out_steps  = L / D;

    cudaError_t e;

    e = cudaMalloc(&state.d_fm_prev, C * sizeof(cufftComplex));
    if (e != cudaSuccess) {
        fprintf(stderr, "fm_demod_alloc d_fm_prev: %s\n", cudaGetErrorString(e));
        return false;
    }
    cudaMemset(state.d_fm_prev, 0, C * sizeof(cufftComplex));

    e = cudaMalloc(&state.d_fm_output,
                   static_cast<size_t>(out_steps) * C * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "fm_demod_alloc d_fm_output: %s\n", cudaGetErrorString(e));
        return false;
    }

    return true;
}

void fm_demod_process(ChannelizerState& state, int out_steps)
{
    const int C = state.config.fft_size;

    constexpr int CHAN_W = 512;
    dim3 fm_block(CHAN_W);
    dim3 fm_grid((C + CHAN_W - 1) / CHAN_W, out_steps);
    fm_demod_kernel<<<fm_grid, fm_block>>>(
        state.d_s2_output,
        state.d_fm_prev,
        state.d_fm_output,
        C);

    // Update d_fm_prev AFTER the kernel has returned so that the step=0 read
    // (which uses d_fm_prev as the previous-batch last sample) can never race
    // with this write.  The copy is free at the device level — it reuses the
    // last row already in d_s2_output without touching d_fm_output.
    cudaMemcpy(state.d_fm_prev,
               state.d_s2_output + static_cast<size_t>(out_steps - 1) * C,
               static_cast<size_t>(C) * sizeof(cufftComplex),
               cudaMemcpyDeviceToDevice);
}

void fm_demod_free(ChannelizerState& state)
{
    if (state.d_fm_prev)   { cudaFree(state.d_fm_prev);   state.d_fm_prev   = nullptr; }
    if (state.d_fm_output) { cudaFree(state.d_fm_output); state.d_fm_output = nullptr; }
}

// ---------------------------------------------------------------------------
// C4FM matched receive filter
//
// 19-tap symmetric FIR computed from op25_c4fm_mod.transfer_function_rx at
// 12500 Hz / 4800 baud (span=9).  Applied to FM demod output (radians/sample)
// before Gardner clock recovery; opens the eye for inner symbols and reduces ISI.
//
// Grid: (ceil(M/128), out_steps)   Block: (128, 1)
// ---------------------------------------------------------------------------

#define C4FM_NTAPS 19
#define C4FM_HIST  (C4FM_NTAPS - 1)   // 18 history samples needed across block boundary

__constant__ float d_c4fm_taps[C4FM_NTAPS] = {
     0.00153655f, -0.00196668f,  0.00133832f,  0.00074619f, -0.00430192f,
     0.00879724f, -0.01302746f,  0.01586670f,  0.26490147f,  0.45221919f,
     0.26490147f,  0.01586670f, -0.01302746f,  0.00879724f, -0.00430192f,
     0.00074619f,  0.00133832f, -0.00196668f,  0.00153655f
};

// C = fft_size (channel count)
__global__ void c4fm_filter_kernel(
    const float* __restrict__ d_fm,      // [out_steps * C], step-major
    const float* __restrict__ d_history, // [C4FM_HIST * C], step-major
    float*                    d_out,     // [out_steps * C], step-major
    int C, int out_steps)
{
    const int chan = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockIdx.y);
    if (chan >= C || step >= out_steps) return;

    float acc = 0.0f;
    for (int t = 0; t < C4FM_NTAPS; t++) {
        int src = step - C4FM_HIST + t;
        float x;
        if (src < 0)
            x = d_history[(C4FM_HIST + src) * C + chan];
        else
            x = d_fm[src * C + chan];
        acc += d_c4fm_taps[t] * x;
    }
    d_out[step * C + chan] = acc;
}

bool c4fm_filter_alloc(ChannelizerState& state)
{
    const int num_phases = state.config.num_phases;
    const int C          = state.config.fft_size;
    const int L          = state.input_len / num_phases;
    const int D          = state.config.stage2_decimation;
    const int out_steps  = L / D;

    cudaError_t e;
    e = cudaMalloc(&state.d_c4fm_history,
                   static_cast<size_t>(C4FM_HIST) * C * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "c4fm_filter_alloc d_c4fm_history: %s\n", cudaGetErrorString(e));
        return false;
    }
    cudaMemset(state.d_c4fm_history, 0, static_cast<size_t>(C4FM_HIST) * C * sizeof(float));

    e = cudaMalloc(&state.d_fm_filtered,
                   static_cast<size_t>(out_steps) * C * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "c4fm_filter_alloc d_fm_filtered: %s\n", cudaGetErrorString(e));
        return false;
    }

    return true;
}

void c4fm_filter_process(ChannelizerState& state, int out_steps)
{
    const int C = state.config.fft_size;

    constexpr int BLK = 128;
    dim3 blk(BLK);
    dim3 grd((C + BLK - 1) / BLK, out_steps);
    c4fm_filter_kernel<<<grd, blk>>>(
        state.d_fm_output,
        state.d_c4fm_history,
        state.d_fm_filtered,
        C, out_steps);

    if (out_steps >= C4FM_HIST) {
        cudaMemcpy(state.d_c4fm_history,
                   state.d_fm_output + (out_steps - C4FM_HIST) * C,
                   static_cast<size_t>(C4FM_HIST) * C * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    } else {
        const int keep = C4FM_HIST - out_steps;
        cudaMemcpy(state.d_c4fm_history,
                   state.d_c4fm_history + out_steps * C,
                   static_cast<size_t>(keep) * C * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_c4fm_history + keep * C,
                   state.d_fm_output,
                   static_cast<size_t>(out_steps) * C * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
}

void c4fm_filter_free(ChannelizerState& state)
{
    if (state.d_c4fm_history) { cudaFree(state.d_c4fm_history); state.d_c4fm_history = nullptr; }
    if (state.d_fm_filtered)  { cudaFree(state.d_fm_filtered);  state.d_fm_filtered  = nullptr; }
}

// ---------------------------------------------------------------------------
// DC blocker — equivalent to dc_blocker_ff in the CPU cqpsk path.
//
// IIR first-difference highpass applied per-channel to d_fm_filtered in-place:
//   y[n] = x[n] - x[n-1] + alpha * y[n-1]
//
// alpha = 0.9995  →  cutoff ≈ 1 Hz at 12500 sps.
// One thread per channel; processes all out_steps sequentially (IIR is causal).
// Cross-batch state in d_dcb_x_prev[C] and d_dcb_y_prev[C].
//
// Grid: ceil(C / CHAN_W)   Block: CHAN_W
// ---------------------------------------------------------------------------

#define DCB_ALPHA 0.99f

__global__ void dc_blocker_kernel(
    float* __restrict__ d_fm,   // [out_steps * C], step-major, modified in-place
    float*              d_xp,   // [C] previous input sample (cross-batch state)
    float*              d_yp,   // [C] previous output sample (cross-batch state)
    int C, int out_steps)
{
    const int k = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                + static_cast<int>(threadIdx.x);
    if (k >= C) return;

    float x_prev = d_xp[k];
    float y_prev = d_yp[k];

    for (int s = 0; s < out_steps; s++) {
        float x = d_fm[s * C + k];
        float y = x - x_prev + DCB_ALPHA * y_prev;
        d_fm[s * C + k] = y;
        x_prev = x;
        y_prev = y;
    }

    d_xp[k] = x_prev;
    d_yp[k] = y_prev;
}

bool dc_blocker_alloc(ChannelizerState& state)
{
    const int C = state.config.fft_size;
    cudaError_t e;

    e = cudaMalloc(&state.d_dcb_x_prev, static_cast<size_t>(C) * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "dc_blocker_alloc d_dcb_x_prev: %s\n", cudaGetErrorString(e));
        return false;
    }
    cudaMemset(state.d_dcb_x_prev, 0, static_cast<size_t>(C) * sizeof(float));

    e = cudaMalloc(&state.d_dcb_y_prev, static_cast<size_t>(C) * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "dc_blocker_alloc d_dcb_y_prev: %s\n", cudaGetErrorString(e));
        return false;
    }
    cudaMemset(state.d_dcb_y_prev, 0, static_cast<size_t>(C) * sizeof(float));

    return true;
}

void dc_blocker_process(ChannelizerState& state, int out_steps)
{
    const int C = state.config.fft_size;
    constexpr int CHAN_W = 512;
    dc_blocker_kernel<<<(C + CHAN_W - 1) / CHAN_W, CHAN_W>>>(
        state.d_fm_filtered,
        state.d_dcb_x_prev,
        state.d_dcb_y_prev,
        C, out_steps);
}

void dc_blocker_free(ChannelizerState& state)
{
    if (state.d_dcb_x_prev) { cudaFree(state.d_dcb_x_prev); state.d_dcb_x_prev = nullptr; }
    if (state.d_dcb_y_prev) { cudaFree(state.d_dcb_y_prev); state.d_dcb_y_prev = nullptr; }
}
