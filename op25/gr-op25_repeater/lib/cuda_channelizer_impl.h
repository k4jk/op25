/* -*- c++ -*- */
/*
 * Copyright 2024 OP25 Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_IMPL_H
#define INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_IMPL_H

#include <gnuradio/op25_repeater/cuda_channelizer.h>
#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>

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

    // ---- GR block interface ------------------------------------------------
    void forecast(int noutput_items,
                  gr_vector_int& ninput_items_required) override;

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override;

    // ---- Channel control (all thread-safe) ---------------------------------

    // Assign a polyphase bin to output port [slot].
    // tdma_slot: 0 or 1 for Phase 2 TDMA slots (ignored for Phase 1).
    // p25_mode:  1=Phase1 C4FM, 2=Phase2 H-DQPSK TDMA.
    void set_channel(int slot, float center_freq_hz,
                     int tdma_slot = 0, int p25_mode = 1) override;

    // Deactivate output port [slot].
    void clear_channel(int slot) override;

    // Find the first free slot, set it to center_freq_hz, return its index
    // (or -1 if all slots are occupied).
    int alloc_slot(float center_freq_hz) override;

    // Release a slot back to the free pool.
    void free_slot(int slot) override;

private:
    // ---- Slot state --------------------------------------------------------
    int               d_max_channels;
    int               d_debug;
    int               d_M;          // polyphase bin count (= num_phases = FFT size)
    int               d_input_len;  // complex samples per GPU batch

    std::vector<int>    d_slot_to_bin;    // [max_channels]: bin index, or -1 inactive
    std::vector<int8_t> d_slot_to_mode;  // [max_channels]: 1=Phase1, 2=Phase2
    std::vector<int8_t> d_bin_to_mode;   // [M]: per-bin mode, set when bin is assigned
    std::vector<bool>   d_slot_allocated; // [max_channels]: true if alloc_slot owns it
    std::mutex          d_slot_mutex;

    // Per-slot recovered dibit queues (drained to GR output buffers)
    std::vector<std::deque<uint8_t>> d_out_queues;

    // CPU staging buffers (reused each batch)
    std::vector<int32_t> d_h_counts;
    std::vector<int8_t>  d_h_dibits;

    // Internal IQ accumulation — collects samples across general_work() calls
    // until a full d_input_len batch is ready for the GPU.
    std::vector<gr_complex> d_accum_buf;
    size_t                  d_accum_fill{0};

    // ---- UDP control listener ----------------------------------------------
    int             d_ctrl_port;     // listen port
    int             d_status_port;   // reply port (status / ACK)
    std::thread     d_ctrl_thread;
    std::atomic<bool> d_ctrl_running{false};

    void ctrl_loop();          // runs in d_ctrl_thread
    void handle_command(const std::string& json_in,
                        int sock,
                        const sockaddr_in& sender);

#ifdef HAVE_CUDA
    // ---- CUDA pipeline state -----------------------------------------------
    ChannelizerState   d_state;
    cufftComplex*      d_gpu_in{nullptr};      // [input_len] IQ staging on GPU
    cufftComplex*      d_s1_scratch{nullptr};   // [input_len] stage-1 output on GPU
    float              d_sdr_center_freq_hz;
    float              d_channel_bw_hz;

    // One CUDA stream per slot — allows future async per-channel work to
    // overlap.  The full pipeline batch uses d_pipeline_stream.
    std::vector<cudaStream_t> d_slot_streams;  // [max_channels]
    cudaStream_t              d_pipeline_stream{nullptr};

    // Per-bin M&M reset: set true in set_channel(), consumed in run_gpu_batch()
    // before mm_process() launches. Protected by d_slot_mutex.
    std::vector<bool> d_bin_reset_pending;

    bool alloc_pipeline(const std::string& config_path);
    void run_gpu_batch(const void* cpu_iq_buf);
    void apply_pending_mm_resets();   // called from run_gpu_batch(), default stream
    void free_pipeline();

    // Frequency → polyphase bin (wraps into [0, M))
    int freq_to_bin(float freq_hz) const;
#endif
};

} // namespace op25_repeater
} // namespace gr

#endif /* INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_IMPL_H */
