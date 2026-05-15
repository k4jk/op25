#pragma once

#include "../../include/cuda/channelizer.h"
#include <vector>

// Design a windowed-sinc prototype lowpass filter.
//
// Uses a 4-term Blackman-Harris window (~92 dB stopband attenuation).
// Normalized cutoff: fc = 1 / (2 * cfg.num_phases)
//   — passes one channel width out of the full input band.
//
// Output h[] is resized to cfg.prototype_filter_len.
// DC gain (sum of taps) ≈ 1/num_phases; caller should scale outputs by
// num_phases after the analysis FFT to restore unity passband gain.
void design_proto_filter(const ChannelizerConfig& cfg, std::vector<float>& h);

// Split prototype filter h[] into a polyphase matrix stored row-major:
//   poly[phase * taps_per_phase + tap] = h[phase + tap * num_phases]
//
// This layout places each phase's taps in contiguous memory, which is
// optimal for the CUDA kernel where one thread block handles one phase.
//
// Output poly[] is resized to cfg.num_phases * cfg.taps_per_phase.
void polyphase_decompose(const ChannelizerConfig& cfg,
                         const std::vector<float>& h,
                         std::vector<float>& poly);

// Print prototype filter statistics to stdout.
void print_filter_stats(const ChannelizerConfig& cfg,
                        const std::vector<float>& h);
