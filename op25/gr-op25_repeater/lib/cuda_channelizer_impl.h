/* -*- c++ -*- */
/*
 * Copyright 2024 OP25 Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_IMPL_H
#define INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_IMPL_H

#include <gnuradio/op25_repeater/cuda_channelizer.h>
#include <deque>
#include <mutex>
#include <vector>
#include <array>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuda/channelizer.h"
#endif

namespace gr {
namespace op25_repeater {

class cuda_channelizer_impl : public cuda_channelizer
{
public:
    cuda_channelizer_impl(const std::string& config_path,
                          int max_channels,
                          int debug);
    ~cuda_channelizer_impl();

    void forecast(int noutput_items,
                  gr_vector_int& ninput_items_required) override;

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override;

    void set_channel(int slot, float center_freq_hz) override;
    void clear_channel(int slot) override;

private:
    int  d_max_channels;
    int  d_debug;
    int  d_M;          // num_phases (= polyphase bin count = FFT size)
    int  d_input_len;  // complex samples per GPU batch (= L * M)

    // slot_to_bin[slot] = polyphase bin index, or -1 if inactive
    std::vector<int>   d_slot_to_bin;
    std::mutex         d_slot_mutex;

    // Per-slot recovered dibit queues (drained to GR output buffers)
    std::vector<std::deque<uint8_t>> d_out_queues;

    // CPU staging buffers (re-used each batch)
    std::vector<int32_t> d_h_counts;   // [M] sym counts from GPU
    std::vector<int8_t>  d_h_dibits;   // [MM_MAX_SYM * M] dibits from GPU

#ifdef HAVE_CUDA
    ChannelizerState   d_state;
    cufftComplex*      d_gpu_in;       // [input_len] — IQ staging buffer on GPU
    cufftComplex*      d_s1_scratch;   // [input_len] — stage-1 output buffer on GPU
    float              d_sdr_center_freq_hz;
    float              d_channel_bw_hz;

    // Allocate the full pipeline (called from constructor).
    bool alloc_pipeline(const std::string& config_path);
    // Run one GPU batch; results land in d_out_queues.
    void run_gpu_batch(const void* cpu_iq_buf);
    void free_pipeline();
#endif
};

} // namespace op25_repeater
} // namespace gr

#endif /* INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_IMPL_H */
