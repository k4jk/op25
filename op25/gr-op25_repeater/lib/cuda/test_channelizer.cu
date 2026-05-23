// Copyright 2026, James Kirkham K4JK
// SPDX-License-Identifier: GPL-3.0-or-later

// Standalone CUDA polyphase channelizer test.
//
// Build:
//   nvcc -std=c++17 -O2 -I../../include \
//        -o test_channelizer test_channelizer.cu channelizer.cu proto_fir.cc \
//        -lcufft
//
// Test: inject a complex tone at channel C's center frequency.
//   Expected: output bin C has near-unity magnitude, all other bins near zero.

#include "../../include/cuda/channelizer.h"
#include "proto_fir.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations from channelizer.cu
bool channelizer_alloc(ChannelizerState& state,
                       const std::vector<float>& proto,
                       const std::vector<float>& poly,
                       int input_len);
int  channelizer_process(ChannelizerState& state,
                         const cufftComplex* d_new_input,
                         cufftComplex* d_output);
void channelizer_free(ChannelizerState& state);

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

// Generate one block of a complex exponential at bin `chan`.
// x[n] = exp(j * 2π * chan * n / M)
//
// n_start is always a multiple of INPUT_LEN = L*M, so
//   exp(j * omega * n_start) = exp(j * 2π * chan * L) = 1 exactly.
// We start the phasor at (1, 0) and propagate with double-precision complex
// multiplication to avoid the float precision loss of computing cos/sin(omega*n)
// for large n.
static void gen_tone(std::vector<cufftComplex>& buf, int M, int chan, int /*n_start*/)
{
    const double omega = 2.0 * M_PI * chan / M;
    std::complex<double> phasor(1.0, 0.0);
    const std::complex<double> step(std::cos(omega), std::sin(omega));
    for (auto& s : buf) {
        s = {static_cast<float>(phasor.real()), static_cast<float>(phasor.imag())};
        phasor *= step;
    }
}

// Run one test case: inject a tone at `test_chan`, verify energy lands there.
static bool run_tone_test(ChannelizerState& state,
                          cufftComplex* d_in,
                          cufftComplex* d_out,
                          int input_len,
                          int test_chan,
                          bool verbose)
{
    const int M = state.config.num_phases;
    const int L = input_len / M;

    std::vector<cufftComplex> h_in(input_len);
    std::vector<cufftComplex> h_out(input_len);

    // Warm up: feed two blocks to flush the FIR transient
    int n_offset = 0;
    for (int warmup = 0; warmup < 2; warmup++) {
        gen_tone(h_in, M, test_chan, n_offset);
        cudaMemcpy(d_in, h_in.data(),
                   input_len * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        channelizer_process(state, d_in, d_out);
        n_offset += input_len;
    }

    // Measurement block
    gen_tone(h_in, M, test_chan, n_offset);
    cudaMemcpy(d_in, h_in.data(),
               input_len * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    channelizer_process(state, d_in, d_out);
    cudaMemcpy(h_out.data(), d_out,
               input_len * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Measure channel energies averaged over the last L/2 output steps
    std::vector<double> chan_energy(M, 0.0);
    int steps_to_avg = L / 2;
    for (int s = L - steps_to_avg; s < L; s++) {
        for (int k = 0; k < M; k++) {
            float re = h_out[s * M + k].x;
            float im = h_out[s * M + k].y;
            chan_energy[k] += re * re + im * im;
        }
    }
    for (double& e : chan_energy) e /= steps_to_avg;

    // Expected: unity energy (0 dB) at test_chan, deep null everywhere else
    double target_energy  = chan_energy[test_chan];
    double target_db      = 10.0 * std::log10(target_energy + 1e-30);
    double max_other      = 0.0;
    int    max_other_chan  = -1;
    for (int k = 0; k < M; k++) {
        if (k == test_chan) continue;
        if (chan_energy[k] > max_other) {
            max_other = chan_energy[k];
            max_other_chan = k;
        }
    }
    double max_other_db = 10.0 * std::log10(max_other + 1e-30);
    double rejection_db = target_db - max_other_db;

    if (verbose) {
        printf("  Tone at ch %-3d :  target %.2f dB,  "
               "worst other ch %-3d = %.2f dB,  rejection = %.1f dB  %s\n",
               test_chan, target_db,
               max_other_chan, max_other_db,
               rejection_db,
               rejection_db > 60.0 ? "PASS" : "FAIL");
    }

    return (target_db > -3.0) && (rejection_db > 60.0);
}

int main()
{
    printf("=== OP25 CUDA Channelizer — CUDA Kernel Test ===\n\n");

    // Config + filter design
    ChannelizerConfig cfg = make_config();
    char err[256];
    if (!compute_config(cfg, err)) {
        fprintf(stderr, "compute_config: %s\n", err); return 1;
    }

    std::vector<float> proto, poly;
    design_proto_filter(cfg, proto);
    polyphase_decompose(cfg, proto, poly);

    printf("Config: M=%d phases, K=%d taps/phase, filter len=%d\n",
           cfg.num_phases, cfg.taps_per_phase, cfg.prototype_filter_len);
    printf("        stage1=÷%d → %.0f Hz/ch,  stage2=÷%d → %.0f Hz/ch\n\n",
           cfg.stage1_decimation, cfg.stage1_output_rate_hz,
           cfg.stage2_decimation, cfg.stage2_output_rate_hz);

    // Allocate channelizer state
    ChannelizerState state{};
    state.config = cfg;

    const int INPUT_LEN = 2048 * cfg.num_phases;  // L=2048 output steps per block

    if (!channelizer_alloc(state, proto, poly, INPUT_LEN)) {
        fprintf(stderr, "channelizer_alloc failed\n"); return 1;
    }

    // GPU I/O buffers
    cufftComplex *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  INPUT_LEN * sizeof(cufftComplex));
    cudaMalloc(&d_out, INPUT_LEN * sizeof(cufftComplex));

    printf("Running tone injection tests (expect ≥60 dB channel rejection):\n");
    int pass = 0, fail = 0;

    // Test a spread of channels: DC, midband, and near-Nyquist
    int test_chans[] = {0, 1, 5, 16, 31, 32, 48, 62, 63};
    int n_tests = static_cast<int>(sizeof(test_chans) / sizeof(test_chans[0]));

    for (int i = 0; i < n_tests; i++) {
        int ch = test_chans[i];
        if (ch >= cfg.num_phases) continue;
        // Reset history for each test
        cudaMemset(state.d_input, 0,
                   static_cast<size_t>(cfg.taps_per_phase - 1 + INPUT_LEN / cfg.num_phases)
                   * cfg.num_phases * sizeof(cufftComplex));

        bool ok = run_tone_test(state, d_in, d_out, INPUT_LEN, ch, true);
        ok ? pass++ : fail++;
    }

    printf("\nResults: %d/%d PASS\n", pass, pass + fail);

    // Print GPU memory used
    size_t padded   = static_cast<size_t>(cfg.taps_per_phase - 1 + INPUT_LEN / cfg.num_phases)
                      * cfg.num_phases;
    size_t poly_sz  = static_cast<size_t>(cfg.num_phases * cfg.taps_per_phase);
    size_t phase_sz = static_cast<size_t>(INPUT_LEN);
    size_t total_bytes = (padded + phase_sz) * sizeof(cufftComplex)
                       + poly_sz * sizeof(float)
                       + static_cast<size_t>(cfg.prototype_filter_len) * sizeof(float);
    printf("\nGPU memory: %.2f MB (padded buf + poly + phase_out)\n",
           total_bytes / 1048576.0);

    cudaFree(d_in);
    cudaFree(d_out);
    channelizer_free(state);

    return fail > 0 ? 1 : 0;
}
