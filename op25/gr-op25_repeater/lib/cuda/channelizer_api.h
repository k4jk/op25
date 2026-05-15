#pragma once

// Public C++ API for the polyphase analysis channelizer stage.
// Functions are defined in channelizer.cu and compiled by nvcc.

#include "../../include/cuda/channelizer.h"
#include <vector>

// Allocate all stage-1 GPU buffers and create the cuFFT plan.
//
//   proto      — prototype FIR coefficients (length cfg.prototype_filter_len)
//   poly       — polyphase matrix (length M * K), row-major (phase × tap)
//   input_len  — new complex samples per block (must be a multiple of M)
//
// Sets state.input_len; populates state.d_proto_filter, d_poly_phases,
// d_input (padded), d_phase_out, and fft_plan.
// Returns false on any CUDA/cuFFT error.
bool channelizer_alloc(ChannelizerState& state,
                       const std::vector<float>& proto,
                       const std::vector<float>& poly,
                       int input_len);

// Process one block of IQ data through the polyphase filter bank.
//
//   d_new_input — [input_len] complex samples on GPU (new block)
//   d_output    — [input_len] complex output on GPU, step-major [L * M]
//                 where L = input_len / M
//
// Returns L (number of output samples per channel).
int channelizer_process(ChannelizerState& state,
                        const cufftComplex* d_new_input,
                        cufftComplex* d_output);

// Free all stage-1 GPU buffers and destroy the cuFFT plan.
void channelizer_free(ChannelizerState& state);
