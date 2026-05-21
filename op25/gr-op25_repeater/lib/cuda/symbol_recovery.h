#pragma once

#include "../../include/cuda/channelizer.h"

// Maximum symbols extractable from one 80-sample block.
// Phase 1: 80 / 2.6042 ≈ 30.7 → use 40 for margin.
// Phase 2: 80 / 2.0833 ≈ 38.4 → same 40-sample buffer covers both.
static constexpr int MM_MAX_SYM      = 40;
static constexpr int MM_FAST_SYMBOLS = 100;  // symbols at fast gain after channel reset

// Cross-batch history depth.  Must satisfy:
//   floor(mu_carry_min − sps/2) − 3 ≥ −MM_HIST
// where mu_carry_min is the most-negative carry that can occur.
// Carry oscillates ~[−1.4, −3.6] from 30/31 strobe aliasing, with an additional
// ±1.55 shift from Gardner gain (fast_ctr, gain_mu=0.05, 31 symbols) → min ≈ −5.1.
// At mu=−5.1: floor(−5.1−1.302)−3 = −10 → MM_HIST must be ≥ 10.  Use 12 for margin.
static constexpr int MM_HIST         = 12;

// Allocate GPU state for Gardner symbol timing recovery across all M channels.
// Must be called after fm_demod_alloc() (needs state.config and input_len).
bool mm_alloc(ChannelizerState& state);

// Run Gardner clock recovery for all M channels in parallel.
//
// Phase 1 FM/FSK4 channels (d_channel_mode[k] == 1):
//   Reads:  state.d_fm_filtered  [in_steps * M], step-major (FM demod, C4FM-filtered)
//   Slicer: C4FM 4-level (±600 Hz / ±1800 Hz thresholds), sps = channel_bw / 4800
//
// Phase 2 H-DQPSK channels (d_channel_mode[k] == 2):
//   Reads:  state.d_s2_output    [in_steps * M], step-major (complex IQ)
//   Uses complex Gardner on IQ, then differential decode at the symbol level
//   (curr × conj(prev) → arg → ±π/4 or ±3π/4 → rescale by 4/π → dibit).
//   Slicer: DQPSK 4-level (±π/4 / ±3π/4 thresholds after DC removal), sps = channel_bw / 6000
//
// Phase 1 CQPSK channels (d_channel_mode[k] == 3):
//   Same IQ path as Phase 2 (bypasses FM demod entirely), but sps = channel_bw / 4800.
//   P25 C4FM and H-DQPSK produce identical differential phase transitions (±π/4, ±3π/4)
//   so the same dqpsk_to_dibit slicer applies.  FM pipeline still runs for mode 3
//   bins but its output is unused — the IQ path reads from d_s2_output directly.
//
// All modes write to:
//   state.d_mm_symbols  [MM_MAX_SYM * M], step-major (float, rad/samp or rad)
//   state.d_mm_dibits   [MM_MAX_SYM * M], step-major (int8_t, 0-3)
//   state.d_mm_sym_count[M]               (int32_t, valid symbols per channel)
//
// Per-channel mode is set via d_channel_mode (updated by apply_pending_mm_resets).
void mm_process(ChannelizerState& state, int in_steps);

void mm_free(ChannelizerState& state);
