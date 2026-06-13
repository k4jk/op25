/* -*- c++ -*- */
/*
 * Copyright 2026, James Kirkham K4JK
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
    int               d_M;          // polyphase branch count (= num_phases)
    int               d_fft_size;   // FFT bin count (= num_phases * fft_oversample)
    int               d_input_len;  // complex samples per GPU batch

    std::vector<int>    d_slot_to_bin;    // [max_channels]: bin index, or -1 inactive
    std::vector<int8_t> d_slot_to_mode;  // [max_channels]: 1=Phase1, 2=Phase2
    std::vector<int8_t> d_bin_to_mode;   // [fft_size]: per-bin mode, set when bin is assigned
    std::vector<bool>   d_slot_allocated; // [max_channels]: true if alloc_slot owns it
    std::mutex          d_slot_mutex;

    // Per-slot recovered dibit queues (drained to GR output buffers)
    std::vector<std::deque<uint8_t>> d_out_queues;

    // Per-slot P25 frame sync sliding correlator.
    // Updated in the dibit-push loop; reset on set_channel() bin change.
    // Checks both MAGIC (normal) and REV_P (inverted polarity) to diagnose
    // whether the sync word is present but in inverted-polarity form.
    std::vector<uint64_t> d_sync_sr;        // 48-bit dibit shift register
    std::vector<int>      d_sync_count;     // P25_FRAME_SYNC_MAGIC hit count
    std::vector<int>      d_sync_rev_count; // P25_FRAME_SYNC_REV_P hit count

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
    std::vector<bool>  d_bin_reset_pending;

    // Per-bin IQ derotation step computed at set_channel() time.
    // derot_step = -2π × (channel_freq - bin_center_freq) / channel_bw_hz [rad/sample].
    // Applied by mm_recovery_kernel for modes 2 and 3 (IQ Gardner path).
    // Zeroed for mode 1 (FM/FSK4) bins — FM discriminator handles carrier offset natively.
    std::vector<float> d_bin_derot_step;

    // Per-bin Costas initial phase for mode 3, computed in set_channel() and
    // consumed by apply_pending_mm_resets().  Seeded from any fully-converged
    // mode 3 slot so voice channels inherit the CC's correct phase basin
    // instead of starting at 0 and risking Costas locking 90° off.
    std::vector<float> d_bin_costas_seed;

    // Per-bin S2 filter warmup gate: counts down S2 steps remaining after a
    // reset before dibit output is allowed.  Prevents corrupted FM from the
    // zero→signal transition in the S2 padded history from reaching the P25
    // decoder.  Set to (s2_filter_len-1) by apply_pending_mm_resets(); decremented
    // by out_steps each batch; dibit push is skipped while > 0.
    // Only accessed from run_gpu_batch() — no mutex needed.
    std::vector<int> d_bin_warmup_steps;

    // Batch counter for the IQ diagnostic print (throttled to avoid spam).
    // Fires at batch 50 (~0.3 s) then every 500 batches (~3 s) thereafter.
    int d_diag_batch_ctr{0};

    // ---- Costas PPM convergence tracker ------------------------------------
    // When costas_freq stays within ±10% of a reference value for
    // COSTAS_PPM_STABLE_CHECKS consecutive checks (each COSTAS_PPM_CHECK_INTERVAL
    // batches apart), log a suggested PPM trim to eliminate the residual offset.
    struct CostasStabState {
        float ref_freq;   // reference costas_freq for stability comparison [rad/sym]
        int   stable_ctr; // consecutive in-tolerance checks
        bool  reported;   // true once the suggestion has been logged for this assignment
    };
    std::vector<CostasStabState> d_costas_stab;  // [max_channels]
    int d_ppm_batch_ctr{0};

    bool alloc_pipeline(const std::string& config_path);
    void run_gpu_batch(const void* cpu_iq_buf);
    void apply_pending_mm_resets();   // called from run_gpu_batch(), default stream
    void print_iq_diagnostics(int out_steps); // periodic IQ health report to stderr
    void check_costas_ppm_convergence();      // periodic PPM trim suggestion
    void free_pipeline();

    // Frequency → polyphase bin (wraps into [0, M))
    int freq_to_bin(float freq_hz) const;
#endif
};

} // namespace op25_repeater
} // namespace gr

#endif /* INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_IMPL_H */
