#pragma once

#include "../../include/cuda/channelizer.h"

// Maximum symbols extractable from one 80-sample block.
// Phase 1: 80 / 2.6042 ≈ 30.7 → use 40 for margin.
// Phase 2: 80 / 2.0833 ≈ 38.4 → same 40-sample buffer covers both.
static constexpr int MM_MAX_SYM     = 40;
static constexpr int MM_FAST_SYMBOLS = 100;  // symbols at fast gain after channel reset

// Allocate GPU state for Gardner symbol timing recovery across all M channels.
// Must be called after fm_demod_alloc() (needs state.config and input_len).
bool mm_alloc(ChannelizerState& state);

// Run Gardner clock recovery for all M channels in parallel.
//
// Phase 1 channels (d_channel_mode[k] == 1):
//   Reads:  state.d_fm_filtered  [in_steps * M], step-major (FM demod, C4FM-filtered)
//   Slicer: C4FM 4-level (±600 Hz / ±1800 Hz thresholds)
//
// Phase 2 channels (d_channel_mode[k] == 2):
//   Reads:  state.d_s2_output    [in_steps * M], step-major (complex IQ)
//   Uses complex Gardner on IQ, then differential decode at the symbol level
//   (curr × conj(prev) → arg → ±π/4 or ±3π/4 → rescale by 4/π → dibit).
//   Slicer: H-DQPSK 4-level (±π/4 / ±3π/4 thresholds after DC removal)
//
// Both modes write to:
//   state.d_mm_symbols  [MM_MAX_SYM * M], step-major (float, rad/samp or rad)
//   state.d_mm_dibits   [MM_MAX_SYM * M], step-major (int8_t, 0-3)
//   state.d_mm_sym_count[M]               (int32_t, valid symbols per channel)
//
// Per-channel mode is set via d_channel_mode (updated by apply_pending_mm_resets).
void mm_process(ChannelizerState& state, int in_steps);

void mm_free(ChannelizerState& state);
