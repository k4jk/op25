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
