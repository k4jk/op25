// Copyright 2026, James Kirkham K4JK
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../../include/cuda/channelizer.h"
#include <vector>

// Design the stage-2 lowpass FIR (windowed sinc, Blackman-Harris window).
//
// Decimation factor D = cfg.stage2_decimation.
// Normalized cutoff fc = 0.5 / D  (at the stage-1 output sample rate).
// Length: n_taps (caller chooses; 128 is a good default for D=25).
void design_s2_filter(const ChannelizerConfig& cfg,
                      int n_taps,
                      std::vector<float>& h);

// Allocate GPU buffers and upload stage-2 FIR coefficients.
// Must be called after channelizer_alloc() so that input_len is already set.
// Returns false on error.
bool s2_filter_alloc(ChannelizerState& state, const std::vector<float>& h);

// Decimate and filter all M channels in one pass.
//
// d_s1_output: stage-1 channelizer output [L * M], step-major (d_phase_out after FFT)
// d_s2_output: decimated output [(L/D) * M], step-major  (written to state.d_s2_output)
// Returns the number of output samples per channel (= L / D).
int s2_filter_process(ChannelizerState& state, const cufftComplex* d_s1_output);

void s2_filter_free(ChannelizerState& state);
