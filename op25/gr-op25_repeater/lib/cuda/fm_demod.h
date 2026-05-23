// Copyright 2026, James Kirkham K4JK
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../../include/cuda/channelizer.h"

// Allocate d_fm_prev [M] and d_fm_output [(L/D) * M].
// Must be called after s2_filter_alloc() so that d_s2_output is sized.
bool fm_demod_alloc(ChannelizerState& state);

// FM discriminator: compute instantaneous frequency for all M channels.
//
// Uses the differential phase method:
//   demod[n] = atan2f( Im(x[n] * conj(x[n-1])),
//                      Re(x[n] * conj(x[n-1])) )
//
// Output is in radians per sample.  At channel_bw_hz sample rate:
//   frequency_Hz = demod * channel_bw_hz / (2π)
//
// d_s2_output is read from state.d_s2_output (written by s2_filter_process).
// Result is written to state.d_fm_output.
// out_steps: number of output samples per channel (= L / stage2_decimation).
void fm_demod_process(ChannelizerState& state, int out_steps);

void fm_demod_free(ChannelizerState& state);

// C4FM matched receive filter — applied to d_fm_output before Gardner clock recovery.
// 19-tap symmetric FIR (op25_c4fm_mod.transfer_function_rx, 12500 Hz / 4800 baud).
// Reduces ISI and opens the eye for inner symbols.
//
// c4fm_filter_alloc: allocates d_c4fm_history [18*M] and d_fm_filtered [out_steps*M].
// c4fm_filter_process: runs the FIR kernel, then updates d_c4fm_history for next block.
// c4fm_filter_free: frees both buffers.
bool c4fm_filter_alloc(ChannelizerState& state);
void c4fm_filter_process(ChannelizerState& state, int out_steps);
void c4fm_filter_free(ChannelizerState& state);

// DC blocker — applied to d_fm_filtered in-place after the C4FM filter and
// before the Gardner timing loop.  Equivalent to dc_blocker_ff in the CPU
// cqpsk path: removes the carrier frequency offset so that Gardner's mid-sample
// probe is zero-mean and timing jitter is not amplified by the DC bias.
//
// IIR: y[n] = x[n] - x[n-1] + alpha*y[n-1], alpha=0.99
// Cutoff ≈ 20 Hz at 12500 sps; settles in ~100 samples (8 ms) after reset,
// within the fast_ctr window.  State is zeroed on channel reset.
bool dc_blocker_alloc(ChannelizerState& state);
void dc_blocker_process(ChannelizerState& state, int out_steps);
void dc_blocker_free(ChannelizerState& state);
