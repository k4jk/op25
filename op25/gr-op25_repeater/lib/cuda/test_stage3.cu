// Copyright 2026, James Kirkham K4JK
// SPDX-License-Identifier: GPL-3.0-or-later

// Milestone 3 — clock recovery + C4FM slicer unit test
//
// Generates a synthetic FM-demod signal (rectangular C4FM pulses at 4800 baud,
// 12500 Hz sample rate) and feeds it directly into mm_process(), bypassing the
// RF pipeline.  This isolates the symbol recovery from channelizer noise.
//
// Build:
//   nvcc -std=c++17 -O2 -DHAVE_CUDA -I../../include \
//        -o test_stage3 test_stage3.cu symbol_recovery.cu proto_fir.cc -lcufft
//
// Test:
//   A repeating 16-dibit pattern is encoded as rectangular C4FM pulses.
//   After 3 warmup blocks the  should be locked; the measurement block
//   must recover all dibits with < 5% error (expected: 0 errors).

#include "../../include/cuda/channelizer.h"
#include "symbol_recovery.h"
#include "proto_fir.h"        // compute_config

#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------

static constexpr int TEST_CHAN = 7;    // channel under test (arbitrary)

// 16-dibit repeating pattern covering all 4 symbol values.
static const int8_t PATTERN[] = { 0,1,2,3, 3,2,1,0, 0,0,1,1, 2,2,3,3 };
static constexpr int PAT_LEN  = static_cast<int>(sizeof(PATTERN));

// P25 C4FM FM-demod levels (radians/sample at channel_bw_hz = 12500 Hz).
//   deviation = symbol_value × 600 Hz × 2π / 12500
static constexpr float DEV_OUTER = 1800.0f * 2.0f * static_cast<float>(M_PI) / 12500.0f; // ±0.9048
static constexpr float DEV_INNER =  600.0f * 2.0f * static_cast<float>(M_PI) / 12500.0f; // ±0.3016

static float dibit_to_fm(int8_t d)
{
    switch (d) {
        case 0: return +DEV_INNER;
        case 1: return +DEV_OUTER;
        case 2: return -DEV_INNER;
        case 3: return -DEV_OUTER;
        default: return 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Generate one block of synthetic FM-demod output.
//
// buf[step * M + TEST_CHAN] = FM deviation of the C4FM symbol active at
// absolute sample position (block_idx * IN_STEPS + step).
// All other channels are set to 0.
//
// The symbol at absolute sample n is PATTERN[(int)(n / sps) % PAT_LEN].
// ---------------------------------------------------------------------------

static void gen_fm_block(float* buf, int M, int IN_STEPS, int block_idx)
{
    const double sps = 12500.0 / 4800.0;
    memset(buf, 0, static_cast<size_t>(IN_STEPS) * M * sizeof(float));
    for (int s = 0; s < IN_STEPS; s++) {
        double abs_pos  = static_cast<double>(block_idx * IN_STEPS + s);
        int    sym_idx  = static_cast<int>(abs_pos / sps) % PAT_LEN;
        buf[s * M + TEST_CHAN] = dibit_to_fm(PATTERN[sym_idx]);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    printf("=== OP25 CUDA Channelizer — Stage-3 M&M Symbol Recovery Test ===\n\n");

    // Minimal config: only num_phases and channel_bw_hz matter here.
    ChannelizerConfig cfg{};
    cfg.sdr_sample_rate_hz = 20000000.0f;
    cfg.sdr_center_freq_hz = 851000000.0f;
    cfg.channel_bw_hz      = 12500.0f;
    cfg.max_channels       = 20;
    cfg.taps_per_phase     = 12;
    cfg.control_port       = 23457;
    cfg.status_port        = 23458;

    char err[256];
    if (!compute_config(cfg, err)) {
        fprintf(stderr, "compute_config: %s\n", err);
        return 1;
    }

    const int M        = cfg.num_phases;
    const int D        = cfg.stage2_decimation;
    const float sps_p1 = cfg.channel_bw_hz / 4800.0f;
    const float sps_p2 = cfg.channel_bw_hz / 6000.0f;

    // IN_STEPS: number of FM-demod samples per block per channel.
    // Must equal L/D from the full pipeline; choose L=2000, D=25 → 80 steps.
    const int L        = 2000;
    const int IN_STEPS = L / D;  // 80

    printf("Config: M=%d  D=%d  IN_STEPS=%d\n", M, D, IN_STEPS);
    printf("Phase 1: 4800 baud  SPS=%.4f\n", sps_p1);
    printf("Phase 2: 6000 baud  SPS=%.4f  (TDMA — timing only in this test)\n\n",
           sps_p2);
    printf("Symbol devations: outer=±%.4f rad/samp  inner=±%.4f rad/samp\n",
           DEV_OUTER, DEV_INNER);
    printf("Pattern length: %d dibits\n\n", PAT_LEN);

    // -------------------------------------------------------------------------
    // Set up a minimal ChannelizerState with only the M&M fields populated.
    // We skip stages 1-2 and feed the FM-demod buffer directly.
    // -------------------------------------------------------------------------
    ChannelizerState state{};
    state.config    = cfg;
    state.input_len = L * M;   // needed by mm_alloc to size the output buffer (not actually used)

    // Allocate the FM output buffer (mm_process reads from here).
    cudaMalloc(&state.d_fm_output,
               static_cast<size_t>(IN_STEPS) * M * sizeof(float));
    cudaMemset(state.d_fm_output, 0,
               static_cast<size_t>(IN_STEPS) * M * sizeof(float));

    if (!mm_alloc(state)) return 1;

    // Initialise mu to sps/2 so the first strobe lands at the centre of the
    // first symbol rather than the leading edge.
    {
        std::vector<float> h_mu(M, sps_p1 / 2.0f);
        cudaMemcpy(state.d_mm_mu, h_mu.data(),
                   M * sizeof(float), cudaMemcpyHostToDevice);
    }

    // -------------------------------------------------------------------------
    // Run 3 warmup blocks + 1 measurement block
    // -------------------------------------------------------------------------
    std::vector<float> h_fm(static_cast<size_t>(IN_STEPS) * M);

    // CPU output buffers for the measurement block
    std::vector<float>   h_symbols(MM_MAX_SYM * M);
    std::vector<int8_t>  h_dibits (MM_MAX_SYM * M);
    std::vector<int32_t> h_counts (M);

    for (int blk = 0; blk < 4; blk++) {
        gen_fm_block(h_fm.data(), M, IN_STEPS, blk);
        cudaMemcpy(state.d_fm_output, h_fm.data(),
                   static_cast<size_t>(IN_STEPS) * M * sizeof(float),
                   cudaMemcpyHostToDevice);
        mm_process(state, IN_STEPS, P25Mode::PHASE1);
    }

    // Download results
    cudaMemcpy(h_symbols.data(), state.d_mm_symbols,
               MM_MAX_SYM * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dibits.data(), state.d_mm_dibits,
               MM_MAX_SYM * M * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts.data(), state.d_mm_sym_count,
               M * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // -------------------------------------------------------------------------
    // Evaluate channel TEST_CHAN
    // -------------------------------------------------------------------------
    int n_sym = h_counts[TEST_CHAN];
    printf("Channel %d: recovered %d symbols\n", TEST_CHAN, n_sym);

    if (n_sym < 20) {
        printf("  FAIL — too few symbols (%d < 20)\n", n_sym);
        mm_free(state);
        cudaFree(state.d_fm_output);
        return 1;
    }

    // The recovered dibits form a periodic repeat of PATTERN (possibly with
    // a phase offset).  Find the offset that gives the best match, then count
    // errors over all symbols.
    int best_offset = 0, best_matches = -1;
    for (int off = 0; off < PAT_LEN; off++) {
        int matches = 0;
        for (int s = 0; s < n_sym; s++) {
            int8_t got      = h_dibits[s * M + TEST_CHAN];
            int8_t expected = PATTERN[(s + off) % PAT_LEN];
            if (got == expected) matches++;
        }
        if (matches > best_matches) {
            best_matches = matches;
            best_offset  = off;
        }
    }

    int errors = n_sym - best_matches;
    double error_rate = static_cast<double>(errors) / n_sym * 100.0;

    printf("  best pattern offset: %d\n", best_offset);
    printf("  errors: %d / %d  (%.1f%%)\n", errors, n_sym, error_rate);

    // Print symbol-by-symbol detail for inspection
    printf("\n  Sym  Got  Expected  FM(rad/s)\n");
    for (int s = 0; s < n_sym; s++) {
        int8_t got = h_dibits[s * M + TEST_CHAN];
        int8_t exp = PATTERN[(s + best_offset) % PAT_LEN];
        float  fm  = h_symbols[s * M + TEST_CHAN];
        printf("  %3d   %d     %d       %+.4f%s\n",
               s, (int)got, (int)exp, fm,
               (got != exp) ? "  *** MISMATCH" : "");
    }

    // Sanity-check inactive channels: zero-input gives near-zero FM, so M&M
    // may still recover some symbols (all at the zero level), but the count
    // should not wildly exceed what the signal channel sees.
    for (int c = 0; c < M; c++) {
        if (c == TEST_CHAN) continue;
        if (h_counts[c] > n_sym + 5)
            printf("WARN: channel %d produced %d symbols (expected ~%d)\n",
                   c, h_counts[c], n_sym);
    }

    // Pass if < 5% error after warmup
    bool ok = (error_rate < 5.0);
    printf("\nResult: %s\n", ok ? "PASS" : "FAIL");

    mm_free(state);
    cudaFree(state.d_fm_output);

    return ok ? 0 : 1;
}
