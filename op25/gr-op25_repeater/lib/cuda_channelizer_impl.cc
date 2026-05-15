/* -*- c++ -*- */
/*
 * Copyright 2024 OP25 Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_channelizer_impl.h"

#include <gnuradio/io_signature.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>

#ifdef HAVE_CUDA
#include "cuda/channelizer_api.h"
#include "cuda/fir_filter.h"
#include "cuda/fm_demod.h"
#include "cuda/symbol_recovery.h"
#include "cuda/proto_fir.h"
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

namespace gr {
namespace op25_repeater {

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

cuda_channelizer::sptr
cuda_channelizer::make(const std::string& config_path, int max_channels, int debug)
{
    return gnuradio::make_block_sptr<cuda_channelizer_impl>(
        config_path, max_channels, debug);
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

cuda_channelizer_impl::cuda_channelizer_impl(const std::string& config_path,
                                             int max_channels,
                                             int debug)
    : gr::block("cuda_channelizer",
                gr::io_signature::make(1, 1, sizeof(gr_complex)),
                gr::io_signature::make(max_channels, max_channels, sizeof(uint8_t))),
      d_max_channels(max_channels),
      d_debug(debug),
      d_M(0),
      d_input_len(0),
      d_slot_to_bin(max_channels, -1),
      d_out_queues(max_channels)
{
#ifndef HAVE_CUDA
    throw std::runtime_error(
        "cuda_channelizer: not compiled with HAVE_CUDA — "
        "rebuild with -DENABLE_CUDA=ON");
#else
    if (!alloc_pipeline(config_path))
        throw std::runtime_error("cuda_channelizer: GPU pipeline initialisation failed");

    d_h_counts.resize(d_M);
    d_h_dibits.resize(static_cast<size_t>(MM_MAX_SYM) * d_M);

    set_tag_propagation_policy(TPP_DONT);
#endif
}

cuda_channelizer_impl::~cuda_channelizer_impl()
{
#ifdef HAVE_CUDA
    free_pipeline();
#endif
}

// ---------------------------------------------------------------------------
// GR scheduling
// ---------------------------------------------------------------------------

void cuda_channelizer_impl::forecast(int noutput_items,
                                     gr_vector_int& ninput_items_required)
{
    // If any output queue already has dibits we can serve without consuming input.
    bool has_queued = false;
    for (const auto& q : d_out_queues)
        if (!q.empty()) { has_queued = true; break; }

    // When queues are empty, require a full GPU batch worth of input.
    // When queues are non-empty, request 1 so GR keeps us schedulable.
    ninput_items_required[0] = has_queued ? 1 : d_input_len;
}

int cuda_channelizer_impl::general_work(int noutput_items,
                                        gr_vector_int& ninput_items,
                                        gr_vector_const_void_star& input_items,
                                        gr_vector_void_star& output_items)
{
#ifndef HAVE_CUDA
    consume_each(ninput_items[0]);
    for (int i = 0; i < d_max_channels; i++) produce(i, 0);
    return WORK_CALLED_PRODUCE;
#else
    // Step 1: if all queues are empty and we have a full batch, run the GPU.
    bool all_empty = true;
    for (const auto& q : d_out_queues)
        if (!q.empty()) { all_empty = false; break; }

    int consumed = 0;
    if (all_empty) {
        if (ninput_items[0] < d_input_len) {
            // Not enough input yet; tell GR nothing was produced or consumed.
            consume_each(0);
            for (int i = 0; i < d_max_channels; i++) produce(i, 0);
            return WORK_CALLED_PRODUCE;
        }
        run_gpu_batch(input_items[0]);
        consumed = d_input_len;
    }

    // Step 2: drain per-slot queues into GR output buffers.
    for (int slot = 0; slot < d_max_channels; slot++) {
        auto& q = d_out_queues[slot];
        int n = std::min(static_cast<int>(q.size()), noutput_items);
        uint8_t* out = static_cast<uint8_t*>(output_items[slot]);
        for (int i = 0; i < n; i++) {
            out[i] = q.front();
            q.pop_front();
        }
        produce(slot, n);
    }

    consume_each(consumed);
    return WORK_CALLED_PRODUCE;
#endif
}

// ---------------------------------------------------------------------------
// Channel control (thread-safe)
// ---------------------------------------------------------------------------

void cuda_channelizer_impl::set_channel(int slot, float center_freq_hz)
{
    if (slot < 0 || slot >= d_max_channels) return;
#ifdef HAVE_CUDA
    // Map absolute frequency to polyphase bin: bin = round(offset / channel_bw_hz)
    // Wrap into [0, M) so negative offsets alias correctly.
    float offset = center_freq_hz - d_sdr_center_freq_hz;
    int bin = static_cast<int>(std::round(offset / d_channel_bw_hz));
    bin = ((bin % d_M) + d_M) % d_M;

    if (d_debug)
        fprintf(stderr, "cuda_channelizer: slot %d → bin %d  (%.0f Hz)\n",
                slot, bin, center_freq_hz);

    std::lock_guard<std::mutex> lk(d_slot_mutex);
    d_slot_to_bin[slot] = bin;
    d_out_queues[slot].clear();   // flush stale dibits for this slot
#endif
}

void cuda_channelizer_impl::clear_channel(int slot)
{
    if (slot < 0 || slot >= d_max_channels) return;
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    d_slot_to_bin[slot] = -1;
    d_out_queues[slot].clear();
}

// ---------------------------------------------------------------------------
// CUDA pipeline helpers (compiled only when HAVE_CUDA)
// ---------------------------------------------------------------------------

#ifdef HAVE_CUDA

static constexpr int S2_FILTER_LEN = 128;
static constexpr int DEFAULT_L     = 2000;   // input steps per GPU batch

bool cuda_channelizer_impl::alloc_pipeline(const std::string& config_path)
{
    // Load JSON config
    std::ifstream f(config_path);
    if (!f.is_open()) {
        fprintf(stderr, "cuda_channelizer: cannot open %s\n", config_path.c_str());
        return false;
    }
    json j;
    try { j = json::parse(f); }
    catch (const std::exception& e) {
        fprintf(stderr, "cuda_channelizer: JSON parse error: %s\n", e.what());
        return false;
    }

    // Accept either top-level keys or wrapped under "channelizer"
    if (j.contains("channelizer")) j = j["channelizer"];

    ChannelizerConfig cfg{};
    cfg.sdr_sample_rate_hz = j.value("sample_rate_hz",  20000000.0f);
    cfg.sdr_center_freq_hz = j.value("center_freq_hz",  851000000.0f);
    cfg.channel_bw_hz      = j.value("channel_bw_hz",   12500.0f);
    cfg.max_channels       = j.value("max_channels",    20);
    cfg.taps_per_phase     = j.value("taps_per_phase",  12);
    cfg.control_port       = j.value("control_port",    23457);
    cfg.status_port        = j.value("status_port",     23458);
    int s2_len             = j.value("s2_filter_len",   S2_FILTER_LEN);

    char err[256];
    if (!compute_config(cfg, err)) {
        fprintf(stderr, "cuda_channelizer: compute_config: %s\n", err);
        return false;
    }

    d_M               = cfg.num_phases;
    d_input_len       = DEFAULT_L * d_M;
    d_sdr_center_freq_hz = cfg.sdr_center_freq_hz;
    d_channel_bw_hz   = cfg.channel_bw_hz;

    d_state.config    = cfg;
    d_state.input_len = d_input_len;

    // Design filters
    std::vector<float> proto, poly, s2_h;
    design_proto_filter(cfg, proto);
    polyphase_decompose(cfg, proto, poly);
    design_s2_filter(cfg, s2_len, s2_h);

    // Allocate each pipeline stage
    if (!channelizer_alloc(d_state, proto, poly, d_input_len)) return false;
    if (!s2_filter_alloc(d_state, s2_h))                       return false;
    if (!fm_demod_alloc(d_state))                              return false;
    if (!mm_alloc(d_state))                                    return false;

    // Initialise M&M mu to sps/2 (sample at symbol centre of first symbol)
    {
        float sps = cfg.channel_bw_hz / 4800.0f;
        std::vector<float> h_mu(d_M, sps / 2.0f);
        cudaMemcpy(d_state.d_mm_mu, h_mu.data(),
                   d_M * sizeof(float), cudaMemcpyHostToDevice);
    }

    // GPU staging buffers
    cudaMalloc(&d_gpu_in,    static_cast<size_t>(d_input_len) * sizeof(cufftComplex));
    cudaMalloc(&d_s1_scratch, static_cast<size_t>(d_input_len) * sizeof(cufftComplex));

    if (d_debug)
        fprintf(stderr, "cuda_channelizer: M=%d  input_len=%d  s2_len=%d\n",
                d_M, d_input_len, s2_len);

    return true;
}

void cuda_channelizer_impl::run_gpu_batch(const void* cpu_iq_buf)
{
    // Upload IQ samples to GPU
    cudaMemcpy(d_gpu_in, cpu_iq_buf,
               static_cast<size_t>(d_input_len) * sizeof(cufftComplex),
               cudaMemcpyHostToDevice);

    // Run the full pipeline
    channelizer_process(d_state, d_gpu_in, d_s1_scratch);
    int out_steps = s2_filter_process(d_state, d_s1_scratch);
    fm_demod_process(d_state, out_steps);
    mm_process(d_state, out_steps);

    // Download results
    cudaMemcpy(d_h_counts.data(), d_state.d_mm_sym_count,
               d_M * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_h_dibits.data(), d_state.d_mm_dibits,
               static_cast<size_t>(MM_MAX_SYM) * d_M * sizeof(int8_t),
               cudaMemcpyDeviceToHost);

    // Push dibits to per-slot output queues
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    for (int slot = 0; slot < d_max_channels; slot++) {
        int bin = d_slot_to_bin[slot];
        if (bin < 0) continue;
        int n = d_h_counts[bin];
        for (int s = 0; s < n; s++)
            d_out_queues[slot].push_back(
                static_cast<uint8_t>(d_h_dibits[s * d_M + bin]));
    }
}

void cuda_channelizer_impl::free_pipeline()
{
    mm_free(d_state);
    fm_demod_free(d_state);
    s2_filter_free(d_state);
    channelizer_free(d_state);
    if (d_gpu_in)     { cudaFree(d_gpu_in);     d_gpu_in     = nullptr; }
    if (d_s1_scratch) { cudaFree(d_s1_scratch); d_s1_scratch = nullptr; }
}

#endif // HAVE_CUDA

} // namespace op25_repeater
} // namespace gr
