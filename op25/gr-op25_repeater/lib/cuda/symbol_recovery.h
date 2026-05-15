#pragma once

#include "../../include/cuda/channelizer.h"

// Maximum symbols extractable from one 80-sample block.
// Phase 1: 80 / 2.6042 ≈ 30.7 → use 40 for margin.
// Phase 2: 80 / 2.0833 ≈ 38.4 → same 40-sample buffer covers both.
static constexpr int MM_MAX_SYM = 40;

// P25 modulation mode — determines baud rate and slicer thresholds.
//
// PHASE1: 4800 baud, C4FM (frequency modulation).
//   SPS = channel_bw_hz / 4800 = 12500 / 4800 = 2.6042
//   FM demod output levels: ±0.9048 rad/samp (outer) ±0.3016 (inner)
//   Decision thresholds: ±0.6032 rad/samp
//
// PHASE2: 6000 baud, TDMA/H-DQPSK (phase modulation).
//   SPS = 12500 / 6000 = 2.0833
//   Note: Phase 2 signal path uses DQPSK, not FM.  d_fm_output contains the
//   differential-phase output which encodes 2-bit phase increments.  The M&M
//   timing loop runs identically; only slicer thresholds differ (future work).
enum class P25Mode : int {
    PHASE1 = 1,
    PHASE2 = 2,
};

// Allocate GPU state for M&M symbol timing recovery across all M channels.
// Must be called after fm_demod_alloc() (needs state.config and input_len).
bool mm_alloc(ChannelizerState& state);

// Run M&M clock recovery + 4-level slicer for all M channels in parallel.
//
// Reads:  state.d_fm_output   [in_steps * M], step-major
// Writes: state.d_mm_symbols  [MM_MAX_SYM * M], step-major (float, rad/samp)
//         state.d_mm_dibits   [MM_MAX_SYM * M], step-major (int8_t, 0-3)
//         state.d_mm_sym_count[M]               (int32_t, valid symbols per channel)
//
// Cross-block state (d_mm_mu, d_mm_fm_last, d_mm_last_interp, d_mm_last_dec)
// is updated in-place.
void mm_process(ChannelizerState& state, int in_steps,
                P25Mode mode = P25Mode::PHASE1);

void mm_free(ChannelizerState& state);
