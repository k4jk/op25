// Copyright 2026, James Kirkham K4JK
// SPDX-License-Identifier: GPL-3.0-or-later

// Milestone 2 — integrated pipeline test: stage-1 + stage-2 FIR + FM demod
//
// Build:
//   nvcc -std=c++17 -O2 -DHAVE_CUDA -I../../include \
//        -o test_stage2 test_stage2.cu channelizer.cu fir_filter.cu \
//        fm_demod.cu proto_fir.cc -lcufft
//
// Test: inject x[n] = exp(j * 2π * (fc + delta_f/Fs) * n) at channel TEST_CHAN.
//   = carrier at channel center + small frequency offset delta_f.
// After full pipeline the FM demod output for channel TEST_CHAN should converge
// to 2π * delta_f / channel_bw_hz radians/sample.

#include "../../include/cuda/channelizer.h"
#include "proto_fir.h"
#include "fir_filter.h"
#include "fm_demod.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <vector>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations from channelizer.cu
bool channelizer_alloc(ChannelizerState&, const std::vector<float>&,
                       const std::vector<float>&, int);
int  channelizer_process(ChannelizerState&, const cufftComplex*, cufftComplex*);
void channelizer_free(ChannelizerState&);

// S2_FILTER_LEN: 128 taps gives ~2.6 dB ripple + 8/128=0.0625 transition BW
// at stage-1 rate. The first alias is at (D-1)*channel_bw_hz = 300 kHz, well
// past the stopband.
static constexpr int S2_FILTER_LEN = 128;

static ChannelizerConfig make_config()
{
    ChannelizerConfig cfg{};
    cfg.sdr_sample_rate_hz = 20000000.0f;
    cfg.sdr_center_freq_hz = 851000000.0f;
    cfg.channel_bw_hz      = 12500.0f;
    cfg.max_channels       = 20;
    cfg.taps_per_phase     = 12;
    cfg.control_port       = 23457;
    cfg.status_port        = 23458;
    return cfg;
}

// Generate one block of a frequency-offset complex tone (double precision).
// Carrier at channel `chan` centre, offset by `delta_f_norm` (cycles/sample
// at the SDR input rate).
// delta_f_norm = delta_f_hz / sdr_sample_rate_hz
static void gen_fm_tone(std::vector<cufftComplex>& buf, int M,
                        int chan, double delta_f_norm)
{
    // Total normalized frequency = (chan/M) + delta_f_norm
    const double omega = 2.0 * M_PI * (static_cast<double>(chan) / M + delta_f_norm);
    std::complex<double> phasor(1.0, 0.0);
    const std::complex<double> step(std::cos(omega), std::sin(omega));
    for (auto& s : buf) {
        s = {static_cast<float>(phasor.real()), static_cast<float>(phasor.imag())};
        phasor *= step;
    }
}

// Expected demod output (radians/sample at channel_bw_hz):
//   delta_f_hz * 2π / channel_bw_hz
static double expected_demod(double delta_f_hz, double channel_bw_hz)
{
    return 2.0 * M_PI * delta_f_hz / channel_bw_hz;
}

int main()
{
    printf("=== OP25 CUDA Channelizer — Stage-2 + FM Demod Test ===\n\n");

    ChannelizerConfig cfg = make_config();
    char err[256];
    if (!compute_config(cfg, err)) {
        fprintf(stderr, "compute_config: %s\n", err); return 1;
    }

    const int M = cfg.num_phases;         // 64
    const int D = cfg.stage2_decimation;  // 25

    printf("Config: M=%d  D=%d  S1 rate=%.0f Hz  S2 rate=%.0f Hz\n\n",
           M, D, cfg.stage1_output_rate_hz, cfg.stage2_output_rate_hz);

    // Design filters
    std::vector<float> proto, poly, s2_h;
    design_proto_filter(cfg, proto);
    polyphase_decompose(cfg, proto, poly);
    design_s2_filter(cfg, S2_FILTER_LEN, s2_h);

    // input_len must be a multiple of M*D so stage-2 output is an integer
    // L = 2000 → L/D = 80 output samples per channel per block
    const int INPUT_LEN = 2000 * M;   // L=2000, L/D=80

    // Allocate all pipeline stages
    ChannelizerState state{};
    state.config = cfg;
    if (!channelizer_alloc(state, proto, poly, INPUT_LEN)) { return 1; }
    if (!s2_filter_alloc(state, s2_h))                    { return 1; }
    if (!fm_demod_alloc(state))                            { return 1; }

    const int L        = INPUT_LEN / M;
    const int OUT_STEPS = L / D;     // stage-2 output samples per channel per block

    // GPU I/O buffers
    cufftComplex *d_in = nullptr, *d_s1_out = nullptr;
    cudaMalloc(&d_in,     INPUT_LEN * sizeof(cufftComplex));
    cudaMalloc(&d_s1_out, INPUT_LEN * sizeof(cufftComplex));

    // CPU output buffer for measurement
    std::vector<float> h_fm_out(OUT_STEPS * M);

    // ---------------------------------------------------------------------------
    // Test cases: channel, delta_f_hz
    // ---------------------------------------------------------------------------
    struct TestCase { int chan; double delta_f_hz; };
    TestCase cases[] = {
        { 0,  +1000.0 },
        { 1,  -1000.0 },
        { 5,  +600.0  },   // P25 narrow deviation symbol
        { 16, +1800.0 },   // P25 wide deviation symbol
        { 32, -600.0  },
        { 63, -1800.0 },
    };
    int n_cases = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

    int pass = 0, fail = 0;

    for (int ci = 0; ci < n_cases; ci++) {
        int    chan       = cases[ci].chan;
        double delta_f_hz = cases[ci].delta_f_hz;
        double delta_f_norm = delta_f_hz / cfg.sdr_sample_rate_hz;

        // Reset all pipeline state
        size_t s1_pad = static_cast<size_t>(cfg.taps_per_phase - 1 + L) * M;
        cudaMemset(state.d_input,   0, s1_pad * sizeof(cufftComplex));
        size_t s2_pad = static_cast<size_t>(S2_FILTER_LEN - 1 + L) * M;
        cudaMemset(state.d_s2_padded, 0, s2_pad * sizeof(cufftComplex));
        cudaMemset(state.d_fm_prev, 0, M * sizeof(cufftComplex));

        std::vector<cufftComplex> h_in(INPUT_LEN);

        // Feed 3 warmup blocks + 1 measurement block
        for (int blk = 0; blk < 4; blk++) {
            gen_fm_tone(h_in, M, chan, delta_f_norm);
            cudaMemcpy(d_in, h_in.data(),
                       INPUT_LEN * sizeof(cufftComplex), cudaMemcpyHostToDevice);

            channelizer_process(state, d_in, d_s1_out);
            int out = s2_filter_process(state, d_s1_out);
            fm_demod_process(state, out);
        }

        // Download FM demod output
        cudaMemcpy(h_fm_out.data(), state.d_fm_output,
                   OUT_STEPS * M * sizeof(float), cudaMemcpyDeviceToHost);

        // Measure mean demod output for the target channel (skip first 5 samples)
        double expected = expected_demod(delta_f_hz, cfg.channel_bw_hz);
        double sum = 0.0;
        int n_meas = OUT_STEPS - 5;
        for (int s = 5; s < OUT_STEPS; s++)
            sum += h_fm_out[s * M + chan];
        double measured = sum / n_meas;
        double err_hz   = (measured - expected) / (2.0 * M_PI) * cfg.channel_bw_hz;

        bool ok = std::fabs(err_hz) < 10.0;   // within 10 Hz
        printf("  ch %-2d  Δf=%+7.0f Hz  expected=%+.5f rad/s  "
               "measured=%+.5f rad/s  err=%+.1f Hz  %s\n",
               chan, delta_f_hz, expected, measured, err_hz,
               ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
    }

    printf("\nResults: %d/%d PASS\n", pass, pass + fail);

    // GPU memory summary
    size_t s1_pad_b = static_cast<size_t>(cfg.taps_per_phase - 1 + L) * M * sizeof(cufftComplex);
    size_t s2_pad_b = static_cast<size_t>(S2_FILTER_LEN - 1 + L) * M * sizeof(cufftComplex);
    size_t s2_out_b = static_cast<size_t>(OUT_STEPS) * M * sizeof(cufftComplex);
    size_t fm_out_b = static_cast<size_t>(OUT_STEPS) * M * sizeof(float);
    printf("\nGPU memory: %.2f MB (stage-1 pad + stage-2 pad + outputs)\n",
           (s1_pad_b + s2_pad_b + s2_out_b + fm_out_b) / 1048576.0);

    cudaFree(d_in);
    cudaFree(d_s1_out);
    fm_demod_free(state);
    s2_filter_free(state);
    channelizer_free(state);

    return fail > 0 ? 1 : 0;
}
