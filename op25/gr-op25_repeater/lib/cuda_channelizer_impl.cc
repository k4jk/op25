/* -*- c++ -*- */
/*
 * Copyright 2026, James Kirkham K4JK
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_channelizer_impl.h"
#include "frame_sync_magics.h"

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
      d_fft_size(0),
      d_input_len(0),
      d_slot_to_bin(max_channels, -1),
      d_slot_to_mode(max_channels, 1),
      d_slot_allocated(max_channels, false),
      d_out_queues(max_channels),
      d_sync_sr(max_channels, 0),
      d_sync_count(max_channels, 0),
      d_sync_rev_count(max_channels, 0),
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

    d_h_counts.resize(d_fft_size);
    d_h_dibits.resize(static_cast<size_t>(MM_MAX_SYM) * d_fft_size);
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
    const int8_t mode = static_cast<int8_t>((p25_mode >= 1 && p25_mode <= 3) ? p25_mode : 1);

    // Compute per-bin IQ derotation step for modes 2 and 3.
    // Δf = channel_freq - bin_center_freq; derot_step = -2π×Δf/channel_bw_hz.
    // Uses the nearest-bin center (same rounding as freq_to_bin) so Δf is exact.
    float derot_step = 0.0f;
    if (mode != 1) {
        const float bin_spacing = d_channel_bw_hz * d_M / static_cast<float>(d_fft_size);
        const float offset      = center_freq_hz - d_sdr_center_freq_hz;
        const int   bin_nearest = static_cast<int>(std::round(offset / bin_spacing));
        const float bin_center  = d_sdr_center_freq_hz + static_cast<float>(bin_nearest) * bin_spacing;
        const float delta_f     = center_freq_hz - bin_center;
        derot_step = -2.0f * 3.14159265358979f * delta_f / d_channel_bw_hz;
    }

    if (d_debug)
        fprintf(stderr, "cuda_channelizer: slot %d → bin %d  (%.0f Hz)  mode=%d tdma=%d  derot_step=%.6f rad/samp\n",
                slot, bin, (double)center_freq_hz, (int)mode, tdma_slot, (double)derot_step);
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    const bool bin_changed  = (bin  != d_slot_to_bin[slot]);
    const bool mode_changed = (mode != d_slot_to_mode[slot]);
    d_slot_to_bin[slot]  = bin;
    d_slot_to_mode[slot] = mode;
    d_bin_to_mode[bin]   = mode;
    d_bin_derot_step[bin] = derot_step;
    if (bin_changed || mode_changed) {
        // Only discard buffered dibits when the channel frequency or mode changes.
        // Re-confirming the same slot (repeated grants) must NOT clear the queue —
        // doing so drops mid-LDU dibits, shifts d_rx_count in rx_sync, and causes
        // "resync at wrong time" every time the trunking controller rebroadcasts a grant.
        d_out_queues[slot].clear();
        d_sync_sr[slot]        = 0;
        d_sync_count[slot]     = 0;
        d_sync_rev_count[slot] = 0;
        d_bin_reset_pending[bin] = true;
        // Arm warmup counter immediately so that if set_channel fires between
        // apply_pending_mm_resets() and mm_process() in the same batch the counter
        // is already positive — the pre-mm_process sync scan (in general_work) will
        // then push d_mm_in_warmup=1 to the GPU before mm_process sees the bin.
        d_bin_warmup_steps[bin] = d_state.s2_filter_len - 1;

        // Mode 3 CQPSK: seed Costas phase from any fully-converged mode 3 slot.
        //
        // All mode 3 channels share the same residual carrier offset (the global SDR
        // LO error, which persists after per-channel derotation).  Starting costas_phase
        // at 0 for a fresh voice channel can put the 4th-order PD in the wrong
        // attraction basin: even 1 ppm LO error at 850 MHz ≈ 850 Hz residual →
        // 2π×850/4800 ≈ 64° differential phase bias → Costas settles 90° off → all
        // dibits wrong → P25 sync never detected → 1 s p25p1_fdma timeout → bounce.
        //
        // Using the CC's already-converged phase as the seed avoids this entirely.
        // Priority: iterate slots in order (CC is typically slot 0); take the first
        // mode 3 slot with warmup==0 and fast_ctr==0 (fully locked, not this slot).
        if (mode == 3) {
            float costas_seed = 0.0f;
            for (int s = 0; s < d_max_channels && costas_seed == 0.0f; s++) {
                int ref_bin = d_slot_to_bin[s];
                if (ref_bin < 0 || ref_bin == bin) continue;
                if (d_bin_to_mode[ref_bin] != 3) continue;
                if (d_bin_warmup_steps[ref_bin] > 0 || d_bin_reset_pending[ref_bin]) continue;
                // Read fast_ctr from GPU to confirm Gardner is fully locked (fast_ctr==0).
                int32_t fast_ctr = 1;
                cudaMemcpy(&fast_ctr, d_state.d_mm_fast_ctr + ref_bin,
                           sizeof(int32_t), cudaMemcpyDeviceToHost);
                if (fast_ctr != 0) continue;
                // Seed from this converged bin.
                cudaMemcpy(&costas_seed, d_state.d_costas_phase + ref_bin,
                           sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stderr,
                    "cuda_channelizer: Costas seed bin=%d slot=%d"
                    " from ref bin=%d slot=%d  phase=%.4f rad\n",
                    bin, slot, ref_bin, s, costas_seed);
            }
            d_bin_costas_seed[bin] = costas_seed;
        } else {
            d_bin_costas_seed[bin] = 0.0f;
        }
    }
#endif
}

void cuda_channelizer_impl::clear_channel(int slot)
{
    if (slot < 0 || slot >= d_max_channels) return;
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    d_slot_to_bin[slot]    = -1;
    d_out_queues[slot].clear();
    d_sync_sr[slot]        = 0;
    d_sync_count[slot]     = 0;
    d_sync_rev_count[slot] = 0;
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
    d_sync_sr[slot]        = 0;
    d_sync_count[slot]     = 0;
    d_sync_rev_count[slot] = 0;
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
                const float bin_hz = d_channel_bw_hz * d_M / static_cast<float>(d_fft_size);
                for (int s = 0; s < d_max_channels; s++) {
                    json entry;
                    entry["slot"]      = s;
                    entry["active"]    = (d_slot_to_bin[s] >= 0);
                    entry["allocated"] = d_slot_allocated[s];
                    entry["bin"]       = d_slot_to_bin[s];
                    float freq = (d_slot_to_bin[s] >= 0)
                        ? d_sdr_center_freq_hz + d_slot_to_bin[s] * bin_hz
                        : 0.0f;
                    entry["freq_hz"] = freq;
                    slots_arr.push_back(entry);
                }
            }
            reply["slots"]    = slots_arr;
            reply["M"]        = d_M;
            reply["fft_size"] = d_fft_size;
            reply["ok"]       = true;

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

static constexpr int S2_FILTER_LEN      = 127; // odd-length: H(Nyquist)!=-0, signal passes even if decarrier not yet working
static constexpr int S2_OUT_STEPS_TARGET = 80; // desired stage-2 output steps per GPU batch
                                               // (~30 P25 symbols/batch at 4800 baud, ≈6.4 ms)

int cuda_channelizer_impl::freq_to_bin(float freq_hz) const
{
    float bin_hz = d_channel_bw_hz * d_M / static_cast<float>(d_fft_size);
    float offset = freq_hz - d_sdr_center_freq_hz;
    int bin = static_cast<int>(std::round(offset / bin_hz));
    return ((bin % d_fft_size) + d_fft_size) % d_fft_size;
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
    cfg.fft_oversample     = j.value("fft_oversample",  1);
    cfg.control_port       = j.value("control_port",    23457);
    cfg.status_port        = j.value("status_port",     23458);
    int s2_len             = j.value("s2_filter_len",   S2_FILTER_LEN);

    char err[256];
    if (!compute_config(cfg, err)) {
        fprintf(stderr, "cuda_channelizer: compute_config: %s\n", err);
        return false;
    }

    d_M                   = cfg.num_phases;
    d_fft_size            = cfg.fft_size;
    d_bin_reset_pending.assign(d_fft_size, false);
    d_bin_to_mode.assign(d_fft_size, 1);  // default all bins to Phase 1
    d_bin_warmup_steps.assign(d_fft_size, 0);
    d_bin_derot_step.assign(d_fft_size, 0.0f);
    d_bin_costas_seed.assign(d_fft_size, 0.0f);
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
    if (!dc_blocker_alloc(d_state))                            return false;
    if (!mm_alloc(d_state))                                    return false;

    // Initialise M&M mu to sps/2 so first strobe lands at symbol centre
    {
        float sps = cfg.channel_bw_hz / 4800.0f;
        std::vector<float> h_mu(d_fft_size, sps / 2.0f);
        cudaMemcpy(d_state.d_mm_mu, h_mu.data(),
                   d_fft_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&d_gpu_in,     static_cast<size_t>(d_input_len) * sizeof(cufftComplex));
    // Stage-1 output: L steps × fft_size channels (fft_size >= num_phases when oversampled)
    const size_t s1_sz = static_cast<size_t>(d_input_len / d_M) * d_fft_size;
    cudaMalloc(&d_s1_scratch, s1_sz * sizeof(cufftComplex));

    if (d_debug)
        fprintf(stderr,
                "cuda_channelizer: M=%d  fft_size=%d  input_len=%d  s2_len=%d  "
                "ctrl_port=%d  status_port=%d\n",
                d_M, d_fft_size, d_input_len, s2_len, d_ctrl_port, d_status_port);

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

#ifdef CUDA_DIAG
    // -----------------------------------------------------------------------
    // Timing-aware pipeline trace for bin 733 (Orange CC: 770.60625 MHz).
    // Scans all L steps at each stage; fires when amplitude > 0.05, at most
    // 5 times total.  Compares pre/post-decarrier IQ, S2 IQ, and actual vs
    // computed FM to isolate where carrier corruption enters (if any).
    // -----------------------------------------------------------------------
    static int  t_fires  = 0;
    const bool  do_trace = (t_fires < 5 && d_fft_size == 960);
    const int   TBIN     = 733;
    const int   L_s1     = d_input_len / d_M;

    // (A) Pre-decarrier peak scan: find step with maximum amplitude at TBIN
    int          pre_hot_s   = -1;
    float        pre_hot_amp = 0.0f;
    cufftComplex pre_iq_hot  = {0.0f, 0.0f};
    cufftComplex pre_iq_prv  = {0.0f, 0.0f};

    if (do_trace) {
        for (int s = 0; s < L_s1; s++) {
            cufftComplex v;
            cudaMemcpy(&v, d_s1_scratch + static_cast<size_t>(s) * d_fft_size + TBIN,
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            float a = sqrtf(v.x * v.x + v.y * v.y);
            if (a > pre_hot_amp) { pre_hot_amp = a; pre_hot_s = s; pre_iq_hot = v; }
        }
        if (pre_hot_amp > 0.05f && pre_hot_s > 0)
            cudaMemcpy(&pre_iq_prv,
                       d_s1_scratch + static_cast<size_t>(pre_hot_s - 1) * d_fft_size + TBIN,
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    }

    // (A2) Save pre-decarrier S1 IQ for steps 0..17 at TBIN.
    // S2 steps 63..79 read S1 steps 0..16 of this batch (via 63-step delay);
    // saving step 17 too covers the "prev" sample needed if a spike falls at s=80.
    // Must happen before decarrier_process overwrites d_s1_scratch in-place.
    cufftComplex pre_dc_s1[18] = {};
    if (d_fft_size == 960) {
        const int n_save = std::min(18, L_s1);
        for (int s = 0; s < n_save; s++)
            cudaMemcpy(&pre_dc_s1[s],
                       d_s1_scratch + static_cast<size_t>(s) * d_fft_size + TBIN,
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    }

    // -----------------------------------------------------------------------
    // DCB_v11: decarrier smoking-gun diagnostic.
    //
    // Captures 8 consecutive S1 steps pre- and post-decarrier for:
    //   bin 733 (odd, Orange CC) — decarrier should NEGATE at odd steps
    //   bin 734 (even, adjacent) — decarrier must leave UNCHANGED (all steps)
    //
    // Trigger conditions (both required):
    //   1. Peak amplitude at bin 733 steps 0-7 > 0.05  (real signal present)
    //   2. d_bin_warmup_steps[733] == 0                (S2 history fully populated)
    // Fires for the first 3 qualifying batches, then silently disabled.
    // -----------------------------------------------------------------------
    const int   DC_BINS[2] = {733, 734};
    const int   DC_NSTEPS  = 8;
    cufftComplex pre_dc_cmp[2][8] = {};

    // Always read bin 733 steps 0-7 before decarrier overwrites d_s1_scratch.
    float dc_cmp_peak733 = 0.0f;
    if (d_fft_size == 960) {
        for (int vs = 0; vs < DC_NSTEPS && vs < L_s1; vs++) {
            cudaMemcpy(&pre_dc_cmp[0][vs],
                       d_s1_scratch + static_cast<size_t>(vs) * d_fft_size + DC_BINS[0],
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            float a = sqrtf(pre_dc_cmp[0][vs].x * pre_dc_cmp[0][vs].x +
                            pre_dc_cmp[0][vs].y * pre_dc_cmp[0][vs].y);
            if (a > dc_cmp_peak733) dc_cmp_peak733 = a;
        }
    }
    static int  dc_cmp_signal_ctr = 0;
    const bool  dc_cmp_ready = (d_fft_size == 960 &&
                                 dc_cmp_peak733 > 0.05f &&
                                 d_bin_warmup_steps[DC_BINS[0]] == 0);
    if (dc_cmp_ready) ++dc_cmp_signal_ctr;
    const bool  do_dc_cmp = (dc_cmp_ready && dc_cmp_signal_ctr <= 3);

    // Read bin 734 only when we will actually print the comparison.
    if (do_dc_cmp) {
        for (int vs = 0; vs < DC_NSTEPS && vs < L_s1; vs++)
            cudaMemcpy(&pre_dc_cmp[1][vs],
                       d_s1_scratch + static_cast<size_t>(vs) * d_fft_size + DC_BINS[1],
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    }

#endif // CUDA_DIAG

    // Remove the Nyquist carrier from odd-bin outputs before the S2 filter.
    // For fft_oversample>1, odd FFT bins carry (-1)^step; negating odd-step
    // samples here (before S2) lets the filter and FM demod operate on
    // carrier-free IQ.  No-op when fft_size == num_phases (critically sampled).
    decarrier_process(d_state, d_s1_scratch, d_input_len / d_M);

#ifdef CUDA_DIAG
    // (B_v10) Post-decarrier: read back same steps and compare to expected.
    if (do_dc_cmp) {
        cufftComplex post_dc_cmp[2][8] = {};
        for (int bi = 0; bi < 2; bi++)
            for (int vs = 0; vs < DC_NSTEPS && vs < L_s1; vs++)
                cudaMemcpy(&post_dc_cmp[bi][vs],
                           d_s1_scratch
                               + static_cast<size_t>(vs) * d_fft_size + DC_BINS[bi],
                           sizeof(cufftComplex), cudaMemcpyDeviceToHost);

        const char* BLBL[2] = {"733(odd)", "734(even)"};
        fprintf(stderr,
            "DC_CMP fire=%d  amp733=%.4f  (EXPECTED: bin733 odd-steps negated, bin734 all unchanged)\n",
            dc_cmp_signal_ctr, dc_cmp_peak733);
        for (int bi = 0; bi < 2; bi++) {
            const bool is_odd_bin = (DC_BINS[bi] & 1);
            fprintf(stderr, "  bin %s:\n", BLBL[bi]);
            for (int vs = 0; vs < DC_NSTEPS; vs++) {
                float sign  = (is_odd_bin && (vs & 1)) ? -1.0f : 1.0f;
                float exp_x = pre_dc_cmp[bi][vs].x * sign;
                float exp_y = pre_dc_cmp[bi][vs].y * sign;
                float dx    = post_dc_cmp[bi][vs].x - exp_x;
                float dy    = post_dc_cmp[bi][vs].y - exp_y;
                bool  ok    = (fabsf(dx) < 1e-4f && fabsf(dy) < 1e-4f);
                fprintf(stderr,
                    "    s%d: pre=(%+.4f,%+.4f) post=(%+.4f,%+.4f)"
                    " exp=(%+.4f,%+.4f) %s\n",
                    vs,
                    pre_dc_cmp[bi][vs].x, pre_dc_cmp[bi][vs].y,
                    post_dc_cmp[bi][vs].x, post_dc_cmp[bi][vs].y,
                    exp_x, exp_y,
                    ok ? "OK" : "MISMATCH!");
            }
        }
    }

    // (B) Post-decarrier capture: read same step after carrier removal
    cufftComplex post_iq_hot = {0.0f, 0.0f};
    cufftComplex post_iq_prv = {0.0f, 0.0f};

    if (do_trace && pre_hot_amp > 0.05f) {
        cudaMemcpy(&post_iq_hot,
                   d_s1_scratch + static_cast<size_t>(pre_hot_s) * d_fft_size + TBIN,
                   sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        if (pre_hot_s > 0)
            cudaMemcpy(&post_iq_prv,
                       d_s1_scratch + static_cast<size_t>(pre_hot_s - 1) * d_fft_size + TBIN,
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    }

#endif // CUDA_DIAG

    int out_steps = s2_filter_process(d_state, d_s1_scratch);

#ifdef CUDA_DIAG
    // (C) S2 output peak scan at TBIN (independent — S2 is a 63-step delay, so
    // the hot step from s1 appears at s2[pre_hot_s+63] if pre_hot_s < 17, or at
    // s2[pre_hot_s-17] in the NEXT batch if pre_hot_s >= 17).
    int          s2_hot_s   = -1;
    float        s2_hot_amp = 0.0f;
    cufftComplex s2_iq_hot  = {0.0f, 0.0f};
    cufftComplex s2_iq_prv  = {0.0f, 0.0f};

    if (do_trace && pre_hot_amp > 0.05f) {
        for (int s = 0; s < out_steps; s++) {
            cufftComplex v;
            cudaMemcpy(&v, d_state.d_s2_output + static_cast<size_t>(s) * d_fft_size + TBIN,
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            float a = sqrtf(v.x * v.x + v.y * v.y);
            if (a > s2_hot_amp) { s2_hot_amp = a; s2_hot_s = s; s2_iq_hot = v; }
        }
        if (s2_hot_s > 0)
            cudaMemcpy(&s2_iq_prv,
                       d_state.d_s2_output + static_cast<size_t>(s2_hot_s - 1) * d_fft_size + TBIN,
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        else
            cudaMemcpy(&s2_iq_prv, d_state.d_fm_prev + TBIN,
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    }

#endif // CUDA_DIAG

    fm_demod_process(d_state, out_steps);

#ifdef CUDA_DIAG
    // (B_v10 cont.) FM output for steps 0..7 at bins 733 and 734.
    // Fires for the same first-3-batch window as the pre/post-DC comparison.
    if (do_dc_cmp) {
        const char* BLBL[2] = {"733(odd)", "734(even)"};
        fprintf(stderr, "DC_CMP fire=%d  FM output (post-S2, post-demod):\n",
                dc_cmp_signal_ctr);
        for (int bi = 0; bi < 2; bi++) {
            fprintf(stderr, "  bin %s:", BLBL[bi]);
            for (int vs = 0; vs < DC_NSTEPS && vs < out_steps; vs++) {
                float fm_val;
                cudaMemcpy(&fm_val,
                           d_state.d_fm_output
                               + static_cast<size_t>(vs) * d_fft_size + DC_BINS[bi],
                           sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stderr, " s%d=%.3f", vs, fm_val);
            }
            fprintf(stderr, "\n");
        }
    }

    // (D) FM output: actual value vs expected computed from S2 IQ, full report
    if (do_trace && pre_hot_amp > 0.05f) {
        ++t_fires;

        float actual_fm   = 0.0f;
        float expected_fm = 0.0f;
        if (s2_hot_s >= 0) {
            cudaMemcpy(&actual_fm,
                       d_state.d_fm_output + static_cast<size_t>(s2_hot_s) * d_fft_size + TBIN,
                       sizeof(float), cudaMemcpyDeviceToHost);
            float exp_re = s2_iq_hot.x * s2_iq_prv.x + s2_iq_hot.y * s2_iq_prv.y;
            float exp_im = s2_iq_hot.y * s2_iq_prv.x - s2_iq_hot.x * s2_iq_prv.y;
            expected_fm  = atan2f(exp_im, exp_re);
        }

        float s1_exp_fm = 0.0f;
        if (pre_hot_s > 0) {
            float re = post_iq_hot.x * post_iq_prv.x + post_iq_hot.y * post_iq_prv.y;
            float im = post_iq_hot.y * post_iq_prv.x - post_iq_hot.x * post_iq_prv.y;
            s1_exp_fm = atan2f(im, re);
        }

        fprintf(stderr,
            "pipeline_trace#%d bin=%d(odd=%d):\n"
            "  s1_pre:  s=%d amp=%.4f IQ=(%.4f,%.4f) prev=(%.4f,%.4f)\n"
            "  s1_post: s=%d IQ=(%.4f,%.4f) prev=(%.4f,%.4f)  s1_exp_fm=%.4f\n",
            t_fires, TBIN, TBIN & 1,
            pre_hot_s, pre_hot_amp, pre_iq_hot.x, pre_iq_hot.y, pre_iq_prv.x, pre_iq_prv.y,
            pre_hot_s, post_iq_hot.x, post_iq_hot.y, post_iq_prv.x, post_iq_prv.y, s1_exp_fm);

        if (s2_hot_s >= 0)
            fprintf(stderr,
                "  s2:      s=%d amp=%.4f IQ=(%.4f,%.4f) prev=(%.4f,%.4f)\n"
                "  fm:      s=%d actual=%.4f  exp_from_s2=%.4f  |delta|=%.4f\n",
                s2_hot_s, s2_hot_amp, s2_iq_hot.x, s2_iq_hot.y, s2_iq_prv.x, s2_iq_prv.y,
                s2_hot_s, actual_fm, expected_fm, fabsf(actual_fm - expected_fm));
        else
            fprintf(stderr,
                "  s2: (peak in next batch; pre_hot_s=%d >= 17, maps to s2[%d])\n",
                pre_hot_s, pre_hot_s - 17);
    }

    // -----------------------------------------------------------------------
    // FM spike scanner — fires for any S2 step where |FM| > 1.5 rad at TBIN.
    // Reports S2 IQ (curr/prev), S1 IQ pre- and post-decarrier, and the FM
    // computed from each pair, to isolate whether the spike originates in the
    // polyphase+FFT, the decarrier, or the S2 filter.
    // Covers S2 steps 63..79 (S1 steps 0..16 of this batch) where pre_dc_s1[]
    // is available.  At most 10 spikes total, then silent.
    // -----------------------------------------------------------------------
    static int fm_spike_fires = 0;
    if (d_fft_size == 960 && fm_spike_fires < 10) {
        for (int s = 0; s < out_steps && fm_spike_fires < 10; s++) {
            float fm_val;
            cudaMemcpy(&fm_val,
                       d_state.d_fm_output + static_cast<size_t>(s) * d_fft_size + TBIN,
                       sizeof(float), cudaMemcpyDeviceToHost);
            if (fabsf(fm_val) <= 1.5f) continue;
            ++fm_spike_fires;

            cufftComplex s2c = {}, s2p = {};
            cudaMemcpy(&s2c, d_state.d_s2_output + static_cast<size_t>(s) * d_fft_size + TBIN,
                       sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            if (s > 0)
                cudaMemcpy(&s2p, d_state.d_s2_output + static_cast<size_t>(s-1) * d_fft_size + TBIN,
                           sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            else
                cudaMemcpy(&s2p, d_state.d_fm_prev + TBIN,
                           sizeof(cufftComplex), cudaMemcpyDeviceToHost);

            // S1 steps in this batch that produced s2[s] and s2[s-1]:
            //   s2[s]   ← S1 step (s-63) for s>=63, or history for s<63
            //   s2[s-1] ← S1 step (s-64) for s>=64, or history for s<64
            int s1c = (s   >= 63) ? (s   - 63) : -1;
            int s1p = (s-1 >= 63) ? (s-1 - 63) : -1;

            cufftComplex s1c_post = {}, s1p_post = {}, s1c_pre = {}, s1p_pre = {};
            if (s1c >= 0 && s1c < L_s1)
                cudaMemcpy(&s1c_post, d_s1_scratch + static_cast<size_t>(s1c) * d_fft_size + TBIN,
                           sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            if (s1p >= 0 && s1p < L_s1)
                cudaMemcpy(&s1p_post, d_s1_scratch + static_cast<size_t>(s1p) * d_fft_size + TBIN,
                           sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            if (s1c >= 0 && s1c < 18) s1c_pre = pre_dc_s1[s1c];
            if (s1p >= 0 && s1p < 18) s1p_pre = pre_dc_s1[s1p];

            auto fm2 = [](cufftComplex a, cufftComplex b) -> float {
                float re = a.x*b.x + a.y*b.y;
                float im = a.y*b.x - a.x*b.y;
                return atan2f(im, re);
            };
            float pre_fm  = (s1c>=0 && s1p>=0) ? fm2(s1c_pre,  s1p_pre)  : 999.f;
            float post_fm = (s1c>=0 && s1p>=0) ? fm2(s1c_post, s1p_post) : 999.f;

            fprintf(stderr,
                "FM_SPIKE#%d bin=%d(odd) s2_step=%d actual_fm=%.4f\n"
                "  s2: curr=(%.4f,%.4f) prev=(%.4f,%.4f)\n"
                "  s1[%d]: pre_dc=(%.4f,%.4f) post_dc=(%.4f,%.4f)\n"
                "  s1[%d]: pre_dc=(%.4f,%.4f) post_dc=(%.4f,%.4f)\n"
                "  pre_dc_fm=%.4f  post_dc_fm=%.4f  [for valid C4FM: |fm|<=0.905]\n",
                fm_spike_fires, TBIN, s, fm_val,
                s2c.x, s2c.y, s2p.x, s2p.y,
                s1c, s1c_pre.x, s1c_pre.y, s1c_post.x, s1c_post.y,
                s1p, s1p_pre.x, s1p_pre.y, s1p_post.x, s1p_post.y,
                pre_fm, post_fm);
        }
    }

#endif // CUDA_DIAG

    c4fm_filter_process(d_state, out_steps);
    dc_blocker_process(d_state, out_steps);
    apply_pending_mm_resets();
    // Pre-mm_process sync: push d_mm_in_warmup=1 to GPU for every bin that has
    // an active warmup counter, including bins where set_channel fired after
    // apply_pending_mm_resets() returned (race window: trunking thread vs GR work thread).
    {
        std::lock_guard<std::mutex> lk(d_slot_mutex);
        const int8_t one8 = 1;
        for (int s = 0; s < d_max_channels; s++) {
            int b = d_slot_to_bin[s];
            if (b >= 0 && d_bin_warmup_steps[b] > 0)
                cudaMemcpy(d_state.d_mm_in_warmup + b, &one8, sizeof(int8_t), cudaMemcpyHostToDevice);
        }
    }
    mm_process(d_state, out_steps, d_debug);

    // Download results using the pipeline stream
    cudaMemcpyAsync(d_h_counts.data(), d_state.d_mm_sym_count,
                    d_fft_size * sizeof(int32_t), cudaMemcpyDeviceToHost, d_pipeline_stream);
    cudaMemcpyAsync(d_h_dibits.data(), d_state.d_mm_dibits,
                    static_cast<size_t>(MM_MAX_SYM) * d_fft_size * sizeof(int8_t),
                    cudaMemcpyDeviceToHost, d_pipeline_stream);
    cudaStreamSynchronize(d_pipeline_stream);

    print_iq_diagnostics(out_steps);

#ifdef CUDA_DIAG
    static int diag_counter = 0;
    const bool print_diag = (++diag_counter % 150 == 1);

    // Gardner state diagnostic: bins 733 (Orange CC) vs 498 (Amelia CC)
    if (print_diag && d_fft_size == 960) {
        static const int   DIAG_BINS[] = {733, 498};
        static const char* DIAG_LBLS[] = {"[Orange/733]", "[Amelia/498]"};
        static float prev_filt_79[2]   = {0.0f, 0.0f};

        for (int di = 0; di < 2; di++) {
            const int bin = DIAG_BINS[di];
            float   h_mu, h_omega, h_dc;
            int32_t h_fast, h_cnt;
            cudaMemcpy(&h_mu,    d_state.d_mm_mu       + bin, sizeof(float),   cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_omega, d_state.d_mm_omega    + bin, sizeof(float),   cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_dc,    d_state.d_mm_dc_est   + bin, sizeof(float),   cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_fast,  d_state.d_mm_fast_ctr + bin, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_cnt,   d_state.d_mm_sym_count + bin, sizeof(int32_t), cudaMemcpyDeviceToHost);

            float filt_0, filt_1, filt_78, filt_79;
            cudaMemcpy(&filt_0,  d_state.d_fm_filtered +  0*d_fft_size + bin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&filt_1,  d_state.d_fm_filtered +  1*d_fft_size + bin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&filt_78, d_state.d_fm_filtered + 78*d_fft_size + bin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&filt_79, d_state.d_fm_filtered + 79*d_fft_size + bin, sizeof(float), cudaMemcpyDeviceToHost);

            fprintf(stderr,
                "gardner %s: mu=%.3f omega=%.4f dc=%.4f fast_ctr=%d sym_count=%d\n"
                "  filt boundary: prev_79=%.3f | s0=%.3f s1=%.3f .. s78=%.3f s79=%.3f\n",
                DIAG_LBLS[di], h_mu, h_omega, h_dc, h_fast, h_cnt,
                prev_filt_79[di],
                filt_0, filt_1, filt_78, filt_79);

            prev_filt_79[di] = filt_79;

            if (h_cnt > 0) {
                int n = (h_cnt < 30) ? h_cnt : 30;
                fprintf(stderr, "  dibits:");
                for (int s = 0; s < n; s++) {
                    int8_t dbit;
                    cudaMemcpy(&dbit, d_state.d_mm_dibits + (size_t)s * d_fft_size + bin,
                               sizeof(int8_t), cudaMemcpyDeviceToHost);
                    fprintf(stderr, " %d", (int)dbit);
                }
                fprintf(stderr, "\n");
            }
        }
    }
#endif // CUDA_DIAG

    // Push dibits to per-slot output queues
    std::lock_guard<std::mutex> lk(d_slot_mutex);
    for (int slot = 0; slot < d_max_channels; slot++) {
        int bin = d_slot_to_bin[slot];
        if (bin < 0) continue;

        // S2 filter warmup gate: suppress dibit output for the first
        // (s2_filter_len-1) S2 steps after a channel reset.  During this
        // window the padded history still contains old-channel data, producing
        // FM transients that would corrupt the freshly-reset Gardner state and
        // prevent P25 sync acquisition.
        if (d_bin_warmup_steps[bin] > 0) {
            d_bin_warmup_steps[bin] -= out_steps;
            if (d_bin_warmup_steps[bin] < 0) d_bin_warmup_steps[bin] = 0;
            if (d_bin_warmup_steps[bin] == 0) {
                const int8_t z8 = 0;
                cudaMemcpy(d_state.d_mm_in_warmup + bin, &z8, sizeof(int8_t), cudaMemcpyHostToDevice);
                fprintf(stderr, "cuda_channelizer: warmup EXPIRED bin=%d slot=%d\n", bin, slot);
            }
            continue;
        }

        int n = d_h_counts[bin];
        for (int s = 0; s < n; s++) {
            uint8_t dbit = static_cast<uint8_t>(d_h_dibits[s * d_fft_size + bin]);
            d_out_queues[slot].push_back(dbit);

            // Sliding P25 frame sync correlator — checks both normal polarity
            // (MAGIC) and inverted polarity (REV_P) to detect dibit sign errors.
            // rx_sync requires exact 48-bit match for initial acquisition, and
            // does NOT recognize REV_P — so inverted polarity would produce
            // zero sync hits in the decoder while REV_P hits appear here.
            d_sync_sr[slot] = ((d_sync_sr[slot] << 2) | (dbit & 3)) & P25_FRAME_SYNC_MASK;
            if (d_sync_sr[slot] == P25_FRAME_SYNC_MAGIC) {
                ++d_sync_count[slot];
                if (d_sync_count[slot] == 1)
                    fprintf(stderr,
                        "SYNC_MAGIC FIRST HIT slot=%d bin=%d(odd=%d)\n",
                        slot, bin, bin & 1);
            } else if (d_sync_sr[slot] == P25_FRAME_SYNC_REV_P) {
                ++d_sync_rev_count[slot];
                if (d_sync_rev_count[slot] == 1)
                    fprintf(stderr,
                        "SYNC_REV_P FIRST HIT slot=%d bin=%d(odd=%d)"
                        " — INVERTED POLARITY: rx_sync will never match!\n",
                        slot, bin, bin & 1);
            }
        }
    }
}

// Reset M&M timing state for any bins flagged by set_channel() since the last batch.
// Must be called from the GR work thread (default CUDA stream) after fm_demod and
// before mm_process, so the reset is serialized with both kernels.
void cuda_channelizer_impl::apply_pending_mm_resets()
{
    const float   zero          = 0.0f;
    const int8_t  one8          = 1;
    const int32_t fast_ctr_init = MM_FAST_SYMBOLS;

    // Phase 1 nominal sps: channel_bw / 4800 baud
    // Phase 2 nominal sps: channel_bw / 6000 baud
    const float sps_p1 = d_channel_bw_hz / 4800.0f;
    const float sps_p2 = d_channel_bw_hz / 6000.0f;

    const cufftComplex zero_c = {0.0f, 0.0f};

    std::lock_guard<std::mutex> lk(d_slot_mutex);
    for (int k = 0; k < d_fft_size; k++) {
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

        // Gate dibit output AND Gardner timing updates for (s2_filter_len-1) S2
        // steps: the padded history in d_s2_padded mixes old and new channel data
        // for that many steps, producing FM transients that would corrupt the
        // freshly-reset Gardner state (mu/omega) before sync can be acquired.
        d_bin_warmup_steps[k] = d_state.s2_filter_len - 1;
        cudaMemcpy(d_state.d_mm_in_warmup + k, &one8, sizeof(int8_t), cudaMemcpyHostToDevice);
        fprintf(stderr, "cuda_channelizer: warmup ARMED bin=%d mode=%d steps=%d fft=%d\n",
                k, (int)mode, d_bin_warmup_steps[k], d_fft_size);

        if (mode != 1) {
            // Phase 2 H-DQPSK (mode 2) and Phase 1 CQPSK (mode 3):
            // reset complex IQ history, last decoded complex symbol, and derotation state.
            for (int h = 0; h < MM_HIST; h++)
                cudaMemcpy(d_state.d_p2_iq_last + h * d_fft_size + k, &zero_c,
                           sizeof(cufftComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_p2_last_sym + k, &zero_c, sizeof(cufftComplex), cudaMemcpyHostToDevice);

            // Push derotation step and zero accumulator so the first batch
            // rotates from phase=0 (the rotation is continuous across batches
            // via d_iq_derot_acc, which the kernel advances each batch).
            const float dstep = d_bin_derot_step[k];
            cudaMemcpy(d_state.d_iq_derot_step + k, &dstep, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_iq_derot_acc  + k, &zero,  sizeof(float), cudaMemcpyHostToDevice);

            // Apply Costas seed computed in set_channel() (mode 3) or 0 (mode 2).
            // Mode 3 voice channels are seeded from the CC's converged phase so
            // the Costas PD starts in the correct 4th-order lock basin immediately.
            // costas_freq always resets to 0 — the frequency integrator starts fresh
            // each call so it doesn't carry stale drift from the previous channel.
            const float costas_init = d_bin_costas_seed[k];
            cudaMemcpy(d_state.d_costas_phase + k, &costas_init, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_costas_freq  + k, &zero,        sizeof(float), cudaMemcpyHostToDevice);
        } else {
            // Phase 1 FM/FSK4 (mode 1): reset real FM history and last raw symbol
            for (int h = 0; h < MM_HIST; h++)
                cudaMemcpy(d_state.d_mm_fm_last + h * d_fft_size + k, &zero,
                           sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_mm_last_interp + k, &zero,    sizeof(float),        cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_mm_last_dec    + k, &zero,    sizeof(float),        cudaMemcpyHostToDevice);
            // Clear C4FM filter history so stale FM from a prior channel
            // does not corrupt the first block's filtered output.
            for (int h = 0; h < 18; h++)
                cudaMemcpy(d_state.d_c4fm_history + h * d_fft_size + k, &zero,
                           sizeof(float), cudaMemcpyHostToDevice);
            // Clear DC blocker state so the new channel's carrier offset is
            // tracked from scratch rather than inheriting the old channel's DC.
            cudaMemcpy(d_state.d_dcb_x_prev + k, &zero, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_state.d_dcb_y_prev + k, &zero, sizeof(float), cudaMemcpyHostToDevice);
            // Discard step=0 of d_fm_filtered: c4fm_filter and dc_blocker have
            // already run with stale (pre-reset) history/state, so the first
            // cell of the filtered output may reflect the old channel's signal.
            // Zeroing it before mm_process prevents a spurious Gardner input.
            cudaMemcpy(d_state.d_fm_filtered + k, &zero, sizeof(float), cudaMemcpyHostToDevice);
        }

        d_bin_reset_pending[k] = false;
    }
}

// ---------------------------------------------------------------------------
// IQ path health diagnostics
//
// Fires at batch 50 (~0.3 s) then every 500 batches (~3 s) for all active
// slots in mode 2 or 3.  Reads three GPU arrays per slot:
//   d_s2_output     → IQ amplitude stats before Gardner (pre-MMSE FIR)
//   d_mm_symbols    → DC-corrected phase values after differential decode
//   per-slot scalars: dc_est, fast_ctr, mu, omega, in_warmup
//
// Phase histogram uses a π/8-shifted bucket grid so the four P25 symbol
// points (±π/4, ±3π/4) land at bucket CENTERS rather than boundaries.
// Valid P25: odd buckets (1,3,5,7) >> even buckets (0,2,4,6).
// Random noise or bad derotation: all 8 buckets roughly equal.
// ---------------------------------------------------------------------------
void cuda_channelizer_impl::print_iq_diagnostics(int out_steps)
{
    ++d_diag_batch_ctr;
    if (d_diag_batch_ctr != 50 && d_diag_batch_ctr % 500 != 0) return;

    const int   C  = d_fft_size;
    const float PI = 3.14159265358979f;

    // Snapshot slot table without holding the mutex during GPU reads.
    struct SlotSnap {
        int   bin;
        int8_t mode;
        int   sync_magic;
        int   sync_revp;
        int   n_sym;        // symbols decoded in most-recent batch (from d_h_counts)
    };
    std::vector<SlotSnap> snaps;
    {
        std::lock_guard<std::mutex> lk(d_slot_mutex);
        for (int s = 0; s < d_max_channels; s++) {
            int b = d_slot_to_bin[s];
            if (b < 0) continue;
            SlotSnap ss;
            ss.bin        = b;
            ss.mode       = d_slot_to_mode[s];
            ss.sync_magic = d_sync_count[s];
            ss.sync_revp  = d_sync_rev_count[s];
            ss.n_sym      = d_h_counts[b];
            snaps.push_back(ss);
        }
    }
    if (snaps.empty()) return;

    fprintf(stderr, "\n[IQDIAG] ===== IQ path snapshot (batch %d) =====\n",
            d_diag_batch_ctr);

    for (auto& ss : snaps) {
        const int bin = ss.bin;

        // ── Gardner scalars (single-element copies) ──────────────────────────
        float   h_dc, h_mu, h_omega;
        int32_t h_fast;
        int8_t  h_warmup;
        cudaMemcpy(&h_dc,     d_state.d_mm_dc_est    + bin, sizeof(float),   cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_mu,     d_state.d_mm_mu        + bin, sizeof(float),   cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_omega,  d_state.d_mm_omega     + bin, sizeof(float),   cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_fast,   d_state.d_mm_fast_ctr  + bin, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_warmup, d_state.d_mm_in_warmup + bin, sizeof(int8_t),  cudaMemcpyDeviceToHost);
        // derot_step and costas_seed are CPU-side; costas_phase/freq need GPU reads.
        const float h_derot = d_bin_derot_step[bin];
        float h_costas = 0.0f, h_costas_freq = 0.0f;
        if (ss.mode == 3) {
            cudaMemcpy(&h_costas,      d_state.d_costas_phase + bin, sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_costas_freq, d_state.d_costas_freq  + bin, sizeof(float),
                       cudaMemcpyDeviceToHost);
        }

        fprintf(stderr,
            "[IQDIAG] bin=%-4d mode=%d | sync_magic=%d sync_revp=%d\n"
            "         dc_est=%.4f  fast_ctr=%d  warmup=%d  mu=%.3f  omega=%.4f"
            "  sym/batch=%d  derot=%.6f rad/samp"
            "  costas_phase=%.4f rad  costas_freq=%.6f rad/sym\n",
            bin, (int)ss.mode, ss.sync_magic, ss.sync_revp,
            h_dc, h_fast, (int)h_warmup, h_mu, h_omega, ss.n_sym, h_derot,
            h_costas, h_costas_freq);

        if (ss.mode == 1) {
            fprintf(stderr, "         (mode 1 FM — no IQ analysis)\n");
            continue;
        }

        // ── Diag 1: S2 IQ amplitude (pre-Gardner, invariant to derotation) ──
        // |exp(jθ)×x| = |x|, so raw d_s2_output amplitude == post-derotation amplitude.
        {
            std::vector<cufftComplex> h_s2(out_steps);
            cudaMemcpy2D(
                h_s2.data(),
                sizeof(cufftComplex),
                d_state.d_s2_output + bin,
                static_cast<size_t>(C) * sizeof(cufftComplex),
                sizeof(cufftComplex),
                static_cast<size_t>(out_steps),
                cudaMemcpyDeviceToHost);

            float a_sum = 0, a_sq = 0, a_min = 1e10f, a_max = 0;
            for (int s = 0; s < out_steps; s++) {
                float a = sqrtf(h_s2[s].x * h_s2[s].x + h_s2[s].y * h_s2[s].y);
                a_sum += a;  a_sq += a * a;
                if (a < a_min) a_min = a;
                if (a > a_max) a_max = a;
            }
            float mean = a_sum / out_steps;
            float var  = a_sq  / out_steps - mean * mean;
            fprintf(stderr,
                "  [Diag1] S2 |IQ|: mean=%.4f  std=%.4f  min=%.4f  max=%.4f  (%d steps)\n",
                mean, sqrtf(var > 0 ? var : 0), a_min, a_max, out_steps);
        }

        if (ss.n_sym <= 0) {
            fprintf(stderr, "  [Diag2] no symbols decoded this batch — phase histogram skipped\n");
            continue;
        }

        // ── Diag 2/5: Differential phase histogram ────────────────────────────
        // d_mm_symbols[sym*C+bin] = phase_dc (radians, DC-corrected).
        // π/8-shifted buckets so P25 symbol points (±π/4, ±3π/4) land at
        // bucket centers (odd buckets 1,3,5,7).  Even buckets 0,2,4,6 are the
        // inter-symbol regions at 0 and ±π/2.
        //
        // Bucket centers (after shift):
        //   0: ≈±π (wrap)   1: −3π/4   2: −π/2   3: −π/4
        //   4:  0            5: +π/4    6: +π/2    7: +3π/4
        //
        // Valid P25: odd >> even.  Noise / bad derotation: all ~equal.
        {
            const int n = std::min(ss.n_sym, (int)MM_MAX_SYM);
            std::vector<float> h_ph(n);
            cudaMemcpy2D(
                h_ph.data(),
                sizeof(float),
                d_state.d_mm_symbols + bin,
                static_cast<size_t>(C) * sizeof(float),
                sizeof(float),
                static_cast<size_t>(n),
                cudaMemcpyDeviceToHost);

            int   hist[8] = {};
            float ph_sum = 0, ph_sq = 0;
            for (int i = 0; i < n; i++) {
                float ph = h_ph[i];
                if (ph >  PI) ph -= 2 * PI;
                if (ph < -PI) ph += 2 * PI;
                // Shift by π/8 so P25 points land at bucket centers.
                // shifted=0 corresponds to ph = −π − π/8 (before wrapping).
                float shifted = ph + PI + PI / 8.0f;
                if (shifted >= 2 * PI) shifted -= 2 * PI;
                int b = static_cast<int>(shifted / (PI / 4.0f));
                if (b < 0) b = 0;
                if (b > 7) b = 7;
                hist[b]++;
                ph_sum += ph;
                ph_sq  += ph * ph;
            }
            float ph_mean = ph_sum / n;
            float ph_var  = ph_sq / n - ph_mean * ph_mean;
            float ph_std  = sqrtf(ph_var > 0 ? ph_var : 0);

            // Uniform noise → std ≈ π/√3 ≈ 1.81 rad.  Good P25 → std < 1.0 rad.
            const char* quality = (ph_std < 0.8f) ? "GOOD" :
                                  (ph_std < 1.2f) ? "MARGINAL" : "BAD/NOISE";

            fprintf(stderr,
                "  [Diag2] phase dist (%d syms, mean=%+.3f std=%.3f rad [%s]):\n"
                "    ±π   -3π/4  -π/2  -π/4    0   +π/4  +π/2  +3π/4\n"
                "    [%3d] [%3d] [%3d] [%3d] [%3d] [%3d] [%3d] [%3d]\n"
                "     ^0     ^1    ^2    ^3    ^4    ^5    ^6    ^7\n"
                "           ***         ***         ***         ***  ← P25 peaks expected at odd buckets\n",
                n, ph_mean, ph_std, quality,
                hist[0], hist[1], hist[2], hist[3],
                hist[4], hist[5], hist[6], hist[7]);
        }
    }
    fprintf(stderr, "[IQDIAG] ================================================\n\n");
}

void cuda_channelizer_impl::free_pipeline()
{
    mm_free(d_state);
    dc_blocker_free(d_state);
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
