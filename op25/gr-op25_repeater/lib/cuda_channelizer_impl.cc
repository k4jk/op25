/* -*- c++ -*- */
/*
 * Copyright 2024 OP25 Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_channelizer_impl.h"

#include <gnuradio/io_signature.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

// POSIX socket headers for UDP control listener
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/select.h>

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
      d_slot_to_mode(max_channels, 1),
      d_slot_allocated(max_channels, false),
      d_out_queues(max_channels),
      d_ctrl_port(0),
      d_status_port(0)
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
    d_accum_buf.resize(d_input_len);

    set_tag_propagation_policy(TPP_DONT);

    // Start UDP control listener
    d_ctrl_running = true;
    d_ctrl_thread  = std::thread(&cuda_channelizer_impl::ctrl_loop, this);
#endif
}

cuda_channelizer_impl::~cuda_channelizer_impl()
{
    // Stop the control listener thread gracefully
    d_ctrl_running = false;
    if (d_ctrl_thread.joinable())
        d_ctrl_thread.join();

#ifdef HAVE_CUDA
    free_pipeline();
#endif
}

// ---------------------------------------------------------------------------
// GR scheduling
// ---------------------------------------------------------------------------

void cuda_channelizer_impl::forecast(int /*noutput_items*/,
                                     gr_vector_int& ninput_items_required)
{
    // Always accept whatever GR delivers.  Accumulation to d_input_len happens
    // inside general_work(), so we never need to ask for more than 1 item here.
    // Asking for d_input_len (up to 3.2 M at M=1600) exceeded the GR ring-buffer
    // size and caused a busy-loop: general_work was called repeatedly with
    // insufficient data, consuming nothing and burning a full CPU core.
    ninput_items_required[0] = 1;
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
    // ── Accumulate incoming IQ samples ──────────────────────────────────────
    // GR delivers whatever fits in its ring buffer (~32 K items typically).
    // We copy into d_accum_buf and fire a GPU batch once d_input_len is full.
    const auto* in   = static_cast<const gr_complex*>(input_items[0]);
    int         n_in = ninput_items[0];
    int         consumed = 0;

    while (consumed < n_in) {
        size_t space = static_cast<size_t>(d_input_len) - d_accum_fill;
        size_t take  = std::min(space, static_cast<size_t>(n_in - consumed));
        std::memcpy(d_accum_buf.data() + d_accum_fill, in + consumed,
                    take * sizeof(gr_complex));
        d_accum_fill += take;
        consumed     += static_cast<int>(take);

        if (static_cast<int>(d_accum_fill) >= d_input_len) {
            run_gpu_batch(d_accum_buf.data());
            d_accum_fill = 0;
        }
    }

    // ── Drain per-slot queues to GR output buffers ───────────────────────────
    // Hold the mutex so set_channel()/clear_channel() can't clear a deque
    // while we are iterating it.
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    for (int slot = 0; slot < d_max_channels; slot++) {
        auto&    q   = d_out_queues[slot];
        int      n   = std::min(static_cast<int>(q.size()), noutput_items);
        uint8_t* out = static_cast<uint8_t*>(output_items[slot]);
        for (int i = 0; i < n; i++) { out[i] = q.front(); q.pop_front(); }
        produce(slot, n);
    }

    consume_each(consumed);
    return WORK_CALLED_PRODUCE;
#endif
}

// ---------------------------------------------------------------------------
// Channel control — public API (all thread-safe)
// ---------------------------------------------------------------------------

void cuda_channelizer_impl::set_channel(int slot, float center_freq_hz,
                                        int tdma_slot, int p25_mode)
{
    if (slot < 0 || slot >= d_max_channels) return;
#ifdef HAVE_CUDA
    int bin = freq_to_bin(center_freq_hz);
    const int8_t mode = static_cast<int8_t>((p25_mode == 2) ? 2 : 1);
    if (d_debug)
        fprintf(stderr, "cuda_channelizer: slot %d → bin %d  (%.0f Hz)  mode=%d tdma=%d\n",
                slot, bin, (double)center_freq_hz, (int)mode, tdma_slot);
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    const bool bin_changed  = (bin  != d_slot_to_bin[slot]);
    const bool mode_changed = (mode != d_slot_to_mode[slot]);
    d_slot_to_bin[slot]  = bin;
    d_slot_to_mode[slot] = mode;
    d_bin_to_mode[bin]   = mode;
    if (bin_changed || mode_changed) {
        // Only discard buffered dibits when the channel frequency or mode changes.
        // Re-confirming the same slot (repeated grants) must NOT clear the queue —
        // doing so drops mid-LDU dibits, shifts d_rx_count in rx_sync, and causes
        // "resync at wrong time" every time the trunking controller rebroadcasts a grant.
        d_out_queues[slot].clear();
        d_bin_reset_pending[bin] = true;
    }
#endif
}

void cuda_channelizer_impl::clear_channel(int slot)
{
    if (slot < 0 || slot >= d_max_channels) return;
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    d_slot_to_bin[slot] = -1;
    d_out_queues[slot].clear();
}

int cuda_channelizer_impl::alloc_slot(float center_freq_hz)
{
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    for (int s = 0; s < d_max_channels; s++) {
        if (!d_slot_allocated[s]) {
            d_slot_allocated[s] = true;
#ifdef HAVE_CUDA
            d_slot_to_bin[s] = freq_to_bin(center_freq_hz);
#endif
            d_out_queues[s].clear();
            if (d_debug)
                fprintf(stderr, "cuda_channelizer: alloc_slot → slot %d  (%.0f Hz)\n",
                        s, (double)center_freq_hz);
            return s;
        }
    }
    return -1;   // all slots occupied
}

void cuda_channelizer_impl::free_slot(int slot)
{
    if (slot < 0 || slot >= d_max_channels) return;
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    d_slot_allocated[slot] = false;
    d_slot_to_bin[slot]    = -1;
    d_out_queues[slot].clear();
    if (d_debug)
        fprintf(stderr, "cuda_channelizer: free_slot %d\n", slot);
}

// ---------------------------------------------------------------------------
// UDP control listener
//
// Protocol (JSON on d_ctrl_port, reply to sender):
//
//   {"cmd":"add_channel",    "slot":N, "freq_hz":F}  → {"cmd":"add_channel",    "slot":N, "ok":true}
//   {"cmd":"remove_channel", "slot":N}               → {"cmd":"remove_channel", "slot":N, "ok":true}
//   {"cmd":"alloc_channel",  "freq_hz":F}            → {"cmd":"alloc_channel",  "slot":N, "ok":true|false}
//   {"cmd":"free_channel",   "slot":N}               → {"cmd":"free_channel",   "slot":N, "ok":true}
//   {"cmd":"status"}                                 → {"cmd":"status", "slots":[...], "M":M}
// ---------------------------------------------------------------------------

void cuda_channelizer_impl::ctrl_loop()
{
    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        fprintf(stderr, "cuda_channelizer: ctrl socket: %s\n", strerror(errno));
        return;
    }

    // Allow quick restart
    int yes = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(d_ctrl_port));

    if (::bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "cuda_channelizer: ctrl bind port %d: %s\n",
                d_ctrl_port, strerror(errno));
        ::close(sock);
        return;
    }

    if (d_debug)
        fprintf(stderr, "cuda_channelizer: UDP control listener on port %d\n",
                d_ctrl_port);

    char buf[4096];
    while (d_ctrl_running) {
        // Non-blocking poll so we can check d_ctrl_running periodically
        struct timeval tv{0, 200000};  // 200 ms timeout
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(sock, &fds);
        if (::select(sock + 1, &fds, nullptr, nullptr, &tv) <= 0) continue;

        struct sockaddr_in sender{};
        socklen_t sender_len = sizeof(sender);
        ssize_t n = ::recvfrom(sock, buf, sizeof(buf) - 1, 0,
                               reinterpret_cast<struct sockaddr*>(&sender),
                               &sender_len);
        if (n <= 0) continue;
        buf[n] = '\0';
        handle_command(std::string(buf, n), sock, sender);
    }
    ::close(sock);
}

void cuda_channelizer_impl::handle_command(const std::string& json_in,
                                           int sock,
                                           const struct sockaddr_in& sender)
{
#ifndef HAVE_CUDA
    (void)json_in; (void)sock; (void)sender;
#else
    json reply;
    try {
        json j = json::parse(json_in);
        std::string cmd = j.value("cmd", "");
        reply["cmd"] = cmd;

        if (cmd == "add_channel") {
            int   slot     = j.value("slot",      -1);
            float freq_hz  = j.value("freq_hz",   0.0f);
            int   tdma_sl  = j.value("tdma_slot", 0);
            int   p25_mode = j.value("p25_mode",  1);
            set_channel(slot, freq_hz, tdma_sl, p25_mode);
            reply["slot"] = slot;
            reply["ok"]   = true;

        } else if (cmd == "remove_channel") {
            int slot = j.value("slot", -1);
            clear_channel(slot);
            reply["slot"] = slot;
            reply["ok"]   = true;

        } else if (cmd == "alloc_channel") {
            float freq_hz = j.value("freq_hz", 0.0f);
            int   slot    = alloc_slot(freq_hz);
            reply["slot"] = slot;
            reply["ok"]   = (slot >= 0);

        } else if (cmd == "free_channel") {
            int slot = j.value("slot", -1);
            free_slot(slot);
            reply["slot"] = slot;
            reply["ok"]   = true;

        } else if (cmd == "status") {
            json slots_arr = json::array();
            {
                std::lock_guard<std::mutex> lk(d_slot_mutex);
                for (int s = 0; s < d_max_channels; s++) {
                    json entry;
                    entry["slot"]      = s;
                    entry["active"]    = (d_slot_to_bin[s] >= 0);
                    entry["allocated"] = d_slot_allocated[s];
                    entry["bin"]       = d_slot_to_bin[s];
                    float freq = (d_slot_to_bin[s] >= 0)
                        ? d_sdr_center_freq_hz + d_slot_to_bin[s] * d_channel_bw_hz
                        : 0.0f;
                    entry["freq_hz"] = freq;
                    slots_arr.push_back(entry);
                }
            }
            reply["slots"] = slots_arr;
            reply["M"]     = d_M;
            reply["ok"]    = true;

        } else {
            reply["ok"]    = false;
            reply["error"] = "unknown command";
        }
    } catch (const std::exception& e) {
        reply["ok"]    = false;
        reply["error"] = e.what();
    }

    std::string reply_str = reply.dump();
    ::sendto(sock, reply_str.c_str(), reply_str.size(), 0,
             reinterpret_cast<const struct sockaddr*>(&sender), sizeof(sender));
#endif
}

// ---------------------------------------------------------------------------
// CUDA pipeline helpers
// ---------------------------------------------------------------------------

#ifdef HAVE_CUDA

static constexpr int S2_FILTER_LEN      = 128;
static constexpr int S2_OUT_STEPS_TARGET = 80; // desired stage-2 output steps per GPU batch
                                               // (~30 P25 symbols/batch at 4800 baud, ≈6.4 ms)

int cuda_channelizer_impl::freq_to_bin(float freq_hz) const
{
    float offset = freq_hz - d_sdr_center_freq_hz;
    int bin = static_cast<int>(std::round(offset / d_channel_bw_hz));
    return ((bin % d_M) + d_M) % d_M;
}

bool cuda_channelizer_impl::alloc_pipeline(const std::string& config_path)
{
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

    d_M                   = cfg.num_phases;
    d_bin_reset_pending.assign(d_M, false);
    d_bin_to_mode.assign(d_M, 1);  // default all bins to Phase 1
    {
        // L must be divisible by D (stage2_decimation).  Round OUT_STEPS_TARGET
        // up to the next multiple of D, then multiply by M to get total input samples.
        // L = stage-1 steps per batch = S2_OUT_STEPS_TARGET * D, which is
        // always divisible by D by construction.
        const int D   = cfg.stage2_decimation;
        const int L   = S2_OUT_STEPS_TARGET * D;   // e.g. 80*25=2000 for 20MHz/12.5kHz
        d_input_len   = L * d_M;                  // e.g. 2000*64=128000 samples
    }
    d_sdr_center_freq_hz  = cfg.sdr_center_freq_hz;
    d_channel_bw_hz       = cfg.channel_bw_hz;
    d_ctrl_port           = cfg.control_port;
    d_status_port         = cfg.status_port;

    d_state.config    = cfg;
    d_state.input_len = d_input_len;

    // Allocate CUDA streams — one dedicated pipeline stream + one per slot
    cudaStreamCreate(&d_pipeline_stream);
    d_slot_streams.resize(d_max_channels);
    for (int s = 0; s < d_max_channels; s++)
        cudaStreamCreate(&d_slot_streams[s]);

    // Design filters and initialise all pipeline stages
    std::vector<float> proto, poly, s2_h;
    design_proto_filter(cfg, proto);
    polyphase_decompose(cfg, proto, poly);
    design_s2_filter(cfg, s2_len, s2_h);

    if (!channelizer_alloc(d_state, proto, poly, d_input_len)) return false;
    if (!s2_filter_alloc(d_state, s2_h))                       return false;
    if (!fm_demod_alloc(d_state))                              return false;
    if (!c4fm_filter_alloc(d_state))                           return false;
    if (!mm_alloc(d_state))                                    return false;

    // Initialise M&M mu to sps/2 so first strobe lands at symbol centre
    {
        float sps = cfg.channel_bw_hz / 4800.0f;
        std::vector<float> h_mu(d_M, sps / 2.0f);
        cudaMemcpy(d_state.d_mm_mu, h_mu.data(),
                   d_M * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&d_gpu_in,     static_cast<size_t>(d_input_len) * sizeof(cufftComplex));
    cudaMalloc(&d_s1_scratch, static_cast<size_t>(d_input_len) * sizeof(cufftComplex));

    if (d_debug)
        fprintf(stderr,
                "cuda_channelizer: M=%d  input_len=%d  s2_len=%d  "
                "ctrl_port=%d  status_port=%d\n",
                d_M, d_input_len, s2_len, d_ctrl_port, d_status_port);

    return true;
}

void cuda_channelizer_impl::run_gpu_batch(const void* cpu_iq_buf)
{
    // Upload IQ to GPU (uses pipeline stream for H2D overlap with CPU work)
    cudaMemcpyAsync(d_gpu_in, cpu_iq_buf,
                    static_cast<size_t>(d_input_len) * sizeof(cufftComplex),
                    cudaMemcpyHostToDevice, d_pipeline_stream);
    cudaStreamSynchronize(d_pipeline_stream);

    // Run pipeline on the default stream (cuFFT plan was created for it)
    channelizer_process(d_state, d_gpu_in, d_s1_scratch);
    int out_steps = s2_filter_process(d_state, d_s1_scratch);
    fm_demod_process(d_state, out_steps);
    c4fm_filter_process(d_state, out_steps);
    apply_pending_mm_resets();
    mm_process(d_state, out_steps);

    // Download results using the pipeline stream
    cudaMemcpyAsync(d_h_counts.data(), d_state.d_mm_sym_count,
                    d_M * sizeof(int32_t), cudaMemcpyDeviceToHost, d_pipeline_stream);
    cudaMemcpyAsync(d_h_dibits.data(), d_state.d_mm_dibits,
                    static_cast<size_t>(MM_MAX_SYM) * d_M * sizeof(int8_t),
                    cudaMemcpyDeviceToHost, d_pipeline_stream);
    cudaStreamSynchronize(d_pipeline_stream);

    // Push dibits to per-slot output queues (slot_streams reserved for
    // future per-channel async work that can overlap across slots)
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

// Reset M&M timing state for any bins flagged by set_channel() since the last batch.
// Must be called from the GR work thread (default CUDA stream) after fm_demod and
// before mm_process, so the reset is serialized with both kernels.
void cuda_channelizer_impl::apply_pending_mm_resets()
{
    const float   zero          = 0.0f;
    const int32_t fast_ctr_init = MM_FAST_SYMBOLS;

    // Phase 1 nominal sps: channel_bw / 4800 baud
    // Phase 2 nominal sps: channel_bw / 6000 baud
    const float sps_p1 = d_channel_bw_hz / 4800.0f;
    const float sps_p2 = d_channel_bw_hz / 6000.0f;

    const cufftComplex zero_c = {0.0f, 0.0f};

    std::lock_guard<std::mutex> lk(d_slot_mutex);
    for (int k = 0; k < d_M; k++) {
        if (!d_bin_reset_pending[k]) continue;

        const int8_t mode    = d_bin_to_mode[k];
        const float sps_nom  = (mode == 2) ? sps_p2 : sps_p1;
        const float sps_init = sps_nom / 2.0f;  // mu = sps/2: first strobe at symbol centre

        // Write per-bin mode to GPU so the Gardner kernel uses the correct path
        cudaMemcpy(d_state.d_channel_mode + k, &mode,         sizeof(int8_t),  cudaMemcpyHostToDevice);

        cudaMemcpy(d_state.d_mm_mu        + k, &sps_init,     sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_state.d_mm_omega     + k, &sps_nom,      sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_state.d_mm_dc_est    + k, &zero,         sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_state.d_mm_fast_ctr  + k, &fast_ctr_init, sizeof(int32_t), cudaMemcpyHostToDevice);

        if (mode == 2) {
            // Phase 2: reset complex IQ history and last decoded symbol
            cudaMemcpy(d_state.d_p2_iq_last  + k,        &zero_c, sizeof(cufftComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_p2_iq_last  + d_M + k,  &zero_c, sizeof(cufftComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_p2_iq_last  + 2*d_M + k,&zero_c, sizeof(cufftComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_p2_last_sym + k,         &zero_c, sizeof(cufftComplex), cudaMemcpyHostToDevice);
        } else {
            // Phase 1: reset real FM history and last raw symbol
            cudaMemcpy(d_state.d_mm_fm_last     + k,           &zero, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_mm_fm_last     + d_M + k,    &zero, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_mm_fm_last     + 2*d_M + k,  &zero, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_mm_last_interp + k, &zero,    sizeof(float),        cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_mm_last_dec    + k, &zero,    sizeof(float),        cudaMemcpyHostToDevice);
            // Clear C4FM filter history so stale FM from a prior channel
            // does not corrupt the first block's filtered output.
            for (int h = 0; h < 18; h++)
                cudaMemcpy(d_state.d_c4fm_history + h * d_M + k, &zero,
                           sizeof(float), cudaMemcpyHostToDevice);
        }

        d_bin_reset_pending[k] = false;
    }
}

void cuda_channelizer_impl::free_pipeline()
{
    mm_free(d_state);
    c4fm_filter_free(d_state);
    fm_demod_free(d_state);
    s2_filter_free(d_state);
    channelizer_free(d_state);

    if (d_gpu_in)     { cudaFree(d_gpu_in);     d_gpu_in     = nullptr; }
    if (d_s1_scratch) { cudaFree(d_s1_scratch); d_s1_scratch = nullptr; }

    for (auto& s : d_slot_streams) cudaStreamDestroy(s);
    d_slot_streams.clear();
    if (d_pipeline_stream) {
        cudaStreamDestroy(d_pipeline_stream);
        d_pipeline_stream = nullptr;
    }
}

#endif // HAVE_CUDA

} // namespace op25_repeater
} // namespace gr
