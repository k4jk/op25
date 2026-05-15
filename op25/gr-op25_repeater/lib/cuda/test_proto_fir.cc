// Standalone prototype FIR filter design test.
// Build: g++ -std=c++17 -O2 -I../../include -o test_proto_fir proto_fir.cc test_proto_fir.cc
// No CUDA dependency.

#include "proto_fir.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static ChannelizerConfig default_config()
{
    ChannelizerConfig cfg{};
    cfg.sdr_sample_rate_hz = 20000000.0f;
    cfg.sdr_center_freq_hz = 851000000.0f;
    cfg.channel_bw_hz      = 12500.0f;
    cfg.max_channels       = 20;
    cfg.taps_per_phase     = 12;
    cfg.control_port       = 23457;
    cfg.status_port        = 23458;
    return cfg;
}

// Estimate stopband attenuation by evaluating the frequency response at a
// grid of frequencies in [fc_stop, 0.5] and returning the worst-case dB value.
static float stopband_attenuation_db(const std::vector<float>& h, float fc_stop)
{
    const int N      = static_cast<int>(h.size());
    const int nfreqs = 512;
    float     worst  = 0.0f; // worst = highest magnitude in stopband

    for (int k = 0; k < nfreqs; k++) {
        float f = fc_stop + (0.5f - fc_stop) * k / (nfreqs - 1);
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; n++) {
            double phase = -2.0 * M_PI * f * n;
            re += h[n] * std::cos(phase);
            im += h[n] * std::sin(phase);
        }
        float mag = static_cast<float>(std::sqrt(re * re + im * im));
        if (mag > worst) worst = mag;
    }

    if (worst < 1e-12f) return -999.0f;
    return 20.0f * std::log10(worst);
}

// Passband ripple: worst-case deviation from 1/num_phases in [0, fc_pass].
static float passband_ripple_db(const std::vector<float>& h, float fc_pass,
                                float ideal_gain)
{
    const int N      = static_cast<int>(h.size());
    const int nfreqs = 256;
    float     worst  = 0.0f;

    for (int k = 0; k < nfreqs; k++) {
        float f = fc_pass * k / (nfreqs - 1);
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; n++) {
            double phase = -2.0 * M_PI * f * n;
            re += h[n] * std::cos(phase);
            im += h[n] * std::sin(phase);
        }
        float mag = static_cast<float>(std::sqrt(re * re + im * im));
        float err = std::fabs(mag - ideal_gain);
        if (err > worst) worst = err;
    }

    if (worst < 1e-12f) return -999.0f;
    return 20.0f * std::log10(worst / ideal_gain);
}

// Verify polyphase matrix reconstructs the prototype filter exactly.
static bool verify_polyphase(const ChannelizerConfig& cfg,
                             const std::vector<float>& h,
                             const std::vector<float>& poly)
{
    const int M = cfg.num_phases;
    const int K = cfg.taps_per_phase;
    float max_err = 0.0f;

    for (int k = 0; k < M; k++) {
        for (int m = 0; m < K; m++) {
            float expected = h[k + m * M];
            float got      = poly[k * K + m];
            float err      = std::fabs(got - expected);
            if (err > max_err) max_err = err;
        }
    }

    printf("Polyphase reconstruction max error: %.2e  %s\n",
           max_err, max_err < 1e-7f ? "PASS" : "FAIL");
    return max_err < 1e-7f;
}

// Write prototype filter taps to a plain-text file for verify_fir.py.
static void write_taps(const std::vector<float>& h, const char* path)
{
    std::ofstream f(path);
    f.precision(10);
    for (float tap : h)
        f << tap << "\n";
    printf("Filter taps written to: %s\n", path);
}

int main()
{
    printf("=== OP25 CUDA Channelizer — Prototype FIR Test ===\n\n");

    ChannelizerConfig cfg = default_config();

    char err[256];
    if (!compute_config(cfg, err)) {
        fprintf(stderr, "compute_config failed: %s\n", err);
        return 1;
    }

    // --- design prototype filter ---
    std::vector<float> h;
    design_proto_filter(cfg, h);

    print_filter_stats(cfg, h);
    printf("\n");

    // --- frequency response checks ---
    // fc = 1/(2*M): prototype filter cutoff at the input sample rate.
    //   Passband of interest = [0, channel_bw/2] — the actual audio bandwidth
    //   at the input sample rate.  This is much smaller than fc; the filter
    //   should be essentially flat here.
    // Adjacent channels are spaced at 1/M = 2*fc from the channel of interest.
    // Stopband starts after the transition band: transition_bw ≈ 8/N.
    float fc            = 1.0f / (2.0f * static_cast<float>(cfg.num_phases));
    float f_chan_edge   = (cfg.channel_bw_hz / 2.0f) / cfg.sdr_sample_rate_hz;
    float f_adj_channel = 2.0f * fc;                  // first adjacent channel center
    float f_stop        = f_adj_channel + 0.5f * fc;  // safe into stopband

    // Point-by-point gains at key frequencies
    auto freq_gain = [&](float f) -> float {
        const int N = static_cast<int>(h.size());
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; n++) {
            double phase = -2.0 * M_PI * f * n;
            re += h[n] * std::cos(phase);
            im += h[n] * std::sin(phase);
        }
        return static_cast<float>(std::sqrt(re * re + im * im));
    };

    float gain_dc      = freq_gain(0.0f);
    float gain_ch_edge = freq_gain(f_chan_edge);
    float gain_fc      = freq_gain(fc);
    float gain_adj     = freq_gain(f_adj_channel);
    float atten_db     = stopband_attenuation_db(h, f_stop);

    auto db = [](float g) { return 20.0f * std::log10(g + 1e-15f); };

    // Channel passband ripple = variation from DC to channel edge
    float ch_ripple_db = db(gain_ch_edge) - db(gain_dc);

    printf("Frequency response:\n");
    printf("  Transition BW (8/N)    : ~%.5f  (%.1f Hz)\n",
           8.0f / cfg.prototype_filter_len,
           8.0f / cfg.prototype_filter_len * cfg.sdr_sample_rate_hz);
    printf("  Gain at DC             : %.6f  (%+.3f dB)\n", gain_dc,   db(gain_dc));
    printf("  Gain at ch edge (%.0f Hz) : %.6f  (%+.3f dB)\n",
           f_chan_edge * cfg.sdr_sample_rate_hz, gain_ch_edge, db(gain_ch_edge));
    printf("  Gain at cutoff fc      : %.6f  (%+.3f dB)  [expect ~-6 dB]\n",
           gain_fc, db(gain_fc));
    printf("  Gain at adj. ch (%.0f Hz): %.6f  (%+.3f dB)\n",
           f_adj_channel * cfg.sdr_sample_rate_hz, gain_adj, db(gain_adj));
    printf("  Channel passband ripple: %+.4f dB  %s\n",
           ch_ripple_db, std::fabs(ch_ripple_db) > 0.1f ? "FAIL" : "PASS");
    printf("  Stopband atten (>%.0f Hz): %+.2f dB  %s\n",
           f_stop * cfg.sdr_sample_rate_hz,
           atten_db, atten_db > -60.0f ? "FAIL" : "PASS");
    printf("\n");

    // --- polyphase decomposition ---
    std::vector<float> poly;
    polyphase_decompose(cfg, h, poly);
    printf("Polyphase matrix: %d phases x %d taps = %zu elements\n",
           cfg.num_phases, cfg.taps_per_phase, poly.size());

    bool poly_ok = verify_polyphase(cfg, h, poly);
    printf("\n");

    // --- write taps for Python verification ---
    write_taps(h, "/tmp/op25_proto_filter.txt");
    printf("\nRun  python3 verify_fir.py  to compare against scipy.\n");

    return poly_ok ? 0 : 1;
}
