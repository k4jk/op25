#include "proto_fir.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

bool compute_config(ChannelizerConfig& cfg, char* error_msg)
{
    auto fail = [&](const char* msg) {
        if (error_msg) std::strncpy(error_msg, msg, 256);
        return false;
    };

    if (cfg.channel_bw_hz <= 0.0f)
        return fail("channel_bw_hz must be positive");
    if (cfg.sdr_sample_rate_hz <= 0.0f)
        return fail("sdr_sample_rate_hz must be positive");

    float total_dec_f = cfg.sdr_sample_rate_hz / cfg.channel_bw_hz;
    int   total_dec   = static_cast<int>(std::round(total_dec_f));
    if (std::fabs(total_dec_f - total_dec) > 0.01f)
        return fail("sdr_sample_rate_hz / channel_bw_hz must be an integer");

    // Use the full total decimation as the polyphase count so that each FFT
    // bin maps to exactly one channel_bw_hz-wide channel.  Stage 2 becomes a
    // plain channel filter (D=1) with no further decimation.
    cfg.stage1_decimation     = total_dec;
    cfg.stage2_decimation     = 1;
    cfg.num_phases            = total_dec;
    cfg.prototype_filter_len  = total_dec * cfg.taps_per_phase;
    cfg.stage1_output_rate_hz = cfg.channel_bw_hz;
    cfg.stage2_output_rate_hz = cfg.channel_bw_hz;

    // fft_oversample: 1 = critically sampled (default), 2 = zero-padded 2M FFT
    // giving half-bin (channel_bw_hz/2) spacing without changing output sample rate.
    if (cfg.fft_oversample < 1) cfg.fft_oversample = 1;
    cfg.fft_size = cfg.num_phases * cfg.fft_oversample;

    return true;
}

// ---------------------------------------------------------------------------
// windowed-sinc helpers
// ---------------------------------------------------------------------------

static float sinc(float x)
{
    if (std::fabs(x) < 1e-9f) return 1.0f;
    return std::sin(static_cast<float>(M_PI) * x)
         / (static_cast<float>(M_PI) * x);
}

// 4-term Blackman-Harris window — ~92 dB minimum stopband attenuation.
static float blackman_harris(int n, int N)
{
    const float a0 = 0.35875f, a1 = 0.48829f, a2 = 0.14128f, a3 = 0.01168f;
    float t = 2.0f * static_cast<float>(M_PI) * n / (N - 1);
    return a0
         - a1 * std::cos(t)
         + a2 * std::cos(2.0f * t)
         - a3 * std::cos(3.0f * t);
}

// ---------------------------------------------------------------------------
// prototype filter design
// ---------------------------------------------------------------------------

void design_proto_filter(const ChannelizerConfig& cfg, std::vector<float>& h)
{
    const int   N      = cfg.prototype_filter_len;
    const float fc     = 1.0f / (2.0f * static_cast<float>(cfg.num_phases));
    const float center = (N - 1) / 2.0f;

    h.resize(N);
    for (int n = 0; n < N; n++) {
        float m = n - center;
        h[n] = 2.0f * fc * sinc(2.0f * fc * m) * blackman_harris(n, N);
    }
}

// ---------------------------------------------------------------------------
// polyphase decomposition
// ---------------------------------------------------------------------------

void polyphase_decompose(const ChannelizerConfig& cfg,
                         const std::vector<float>& h,
                         std::vector<float>& poly)
{
    const int M = cfg.num_phases;
    const int K = cfg.taps_per_phase;

    assert(static_cast<int>(h.size()) == M * K);
    poly.resize(M * K);

    // poly[phase * K + tap] = h[phase + tap * M]
    // Each row is one phase's sub-filter, in contiguous memory.
    for (int k = 0; k < M; k++)
        for (int m = 0; m < K; m++)
            poly[k * K + m] = h[k + m * M];
}

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

void print_filter_stats(const ChannelizerConfig& cfg, const std::vector<float>& h)
{
    const int   N  = cfg.prototype_filter_len;
    const float fc = 1.0f / (2.0f * static_cast<float>(cfg.num_phases));

    double sum = 0.0, sumsq = 0.0;
    float  peak = 0.0f;
    for (int i = 0; i < N; i++) {
        sum   += h[i];
        sumsq += static_cast<double>(h[i]) * h[i];
        if (std::fabs(h[i]) > peak) peak = std::fabs(h[i]);
    }

    printf("Prototype FIR filter:\n");
    printf("  Taps              : %d  (%d phases x %d taps/phase)\n",
           N, cfg.num_phases, cfg.taps_per_phase);
    printf("  Normalized cutoff : %.6f  (cycles/sample, Nyquist = 0.5)\n", fc);
    printf("  Cutoff (Hz)       : %.1f Hz\n", fc * cfg.sdr_sample_rate_hz);
    printf("  DC gain (sum)     : %+.8f  (ideal 1/M = %.8f)\n",
           static_cast<float>(sum), 1.0f / cfg.num_phases);
    printf("  L2 norm           : %.8f\n", static_cast<float>(std::sqrt(sumsq)));
    printf("  Peak tap          : %.8f\n", peak);
    printf("  Stage 1 dec       : %d  (num_phases / FFT size)\n",
           cfg.stage1_decimation);
    printf("  Stage 2 dec       : %d\n", cfg.stage2_decimation);
    printf("  Stage 1 out rate  : %.1f Hz\n", cfg.stage1_output_rate_hz);
    printf("  Stage 2 out rate  : %.1f Hz  (== channel_bw_hz)\n",
           cfg.stage2_output_rate_hz);
}
