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
    // 2D grid: blockIdx.x = channel group, blockIdx.y = time step.
    const int chan = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockIdx.y);
    if (chan >= M) return;

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

    // 2D grid: (ceil(M/512), out_steps)   Block: (512, 1)
    constexpr int CHAN_W = 512;
    dim3 fm_block(CHAN_W);
    dim3 fm_grid((M + CHAN_W - 1) / CHAN_W, out_steps);
    fm_demod_kernel<<<fm_grid, fm_block>>>(
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

__global__ void c4fm_filter_kernel(
    const float* __restrict__ d_fm,      // [out_steps * M], step-major (raw FM demod)
    const float* __restrict__ d_history, // [C4FM_HIST * M], step-major (last 18 FM samples of prev batch)
    float*                    d_out,     // [out_steps * M], step-major (filtered output)
    int M, int out_steps)
{
    const int chan = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockIdx.y);
    if (chan >= M || step >= out_steps) return;

    // For output sample at 'step', the filter needs input samples at steps
    // [step - C4FM_HIST .. step], i.e. 19 samples ending at 'step'.
    float acc = 0.0f;
    for (int t = 0; t < C4FM_NTAPS; t++) {
        int src = step - C4FM_HIST + t;  // input index (may be < 0 → use history)
        float x;
        if (src < 0)
            x = d_history[(C4FM_HIST + src) * M + chan];  // C4FM_HIST + src = 18 + src (src is negative)
        else
            x = d_fm[src * M + chan];
        acc += d_c4fm_taps[t] * x;
    }
    d_out[step * M + chan] = acc;
}

bool c4fm_filter_alloc(ChannelizerState& state)
{
    const int M        = state.config.num_phases;
    const int L        = state.input_len / M;
    const int D        = state.config.stage2_decimation;
    const int out_steps = L / D;

    cudaError_t e;
    e = cudaMalloc(&state.d_c4fm_history,
                   static_cast<size_t>(C4FM_HIST) * M * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "c4fm_filter_alloc d_c4fm_history: %s\n", cudaGetErrorString(e));
        return false;
    }
    cudaMemset(state.d_c4fm_history, 0, static_cast<size_t>(C4FM_HIST) * M * sizeof(float));

    e = cudaMalloc(&state.d_fm_filtered,
                   static_cast<size_t>(out_steps) * M * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "c4fm_filter_alloc d_fm_filtered: %s\n", cudaGetErrorString(e));
        return false;
    }

    return true;
}

void c4fm_filter_process(ChannelizerState& state, int out_steps)
{
    const int M = state.config.num_phases;

    constexpr int BLK = 128;
    dim3 blk(BLK);
    dim3 grd((M + BLK - 1) / BLK, out_steps);
    c4fm_filter_kernel<<<grd, blk>>>(
        state.d_fm_output,
        state.d_c4fm_history,
        state.d_fm_filtered,
        M, out_steps);

    // Update cross-block history: copy last C4FM_HIST steps of d_fm_output into d_c4fm_history.
    // d_fm_output layout: [out_steps * M], step-major.
    // The last C4FM_HIST rows start at offset (out_steps - C4FM_HIST) * M.
    if (out_steps >= C4FM_HIST) {
        cudaMemcpy(state.d_c4fm_history,
                   state.d_fm_output + (out_steps - C4FM_HIST) * M,
                   static_cast<size_t>(C4FM_HIST) * M * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    } else {
        // Rare: fewer output steps than history length — shift existing history and append.
        // Shift the tail of existing history forward.
        const int keep = C4FM_HIST - out_steps;
        cudaMemcpy(state.d_c4fm_history,
                   state.d_c4fm_history + out_steps * M,
                   static_cast<size_t>(keep) * M * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_c4fm_history + keep * M,
                   state.d_fm_output,
                   static_cast<size_t>(out_steps) * M * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
}

void c4fm_filter_free(ChannelizerState& state)
{
    if (state.d_c4fm_history) { cudaFree(state.d_c4fm_history); state.d_c4fm_history = nullptr; }
    if (state.d_fm_filtered)  { cudaFree(state.d_fm_filtered);  state.d_fm_filtered  = nullptr; }
}
