#!/usr/bin/env python3
"""
Verify the C++ prototype FIR filter against a scipy reference design.

Usage:
    1. Build and run test_proto_fir first:
         g++ -std=c++17 -O2 -I../../include -o test_proto_fir proto_fir.cc test_proto_fir.cc
         ./test_proto_fir
    2. Then run this script:
         python3 verify_fir.py
"""

import sys
import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Parameters — must match the defaults in test_proto_fir.cc
# ---------------------------------------------------------------------------
SAMPLE_RATE    = 20_000_000.0   # Hz
CHANNEL_BW     = 12_500.0       # Hz
# 20e6 / 12.5e3 = 1600 = 64 × 25; largest power-of-2 factor = 64
NUM_PHASES     = 64             # stage-1 decimation
TAPS_PER_PHASE = 12
FILTER_LEN     = NUM_PHASES * TAPS_PER_PHASE   # 768

TAPS_FILE = "/tmp/op25_proto_filter.txt"

# ---------------------------------------------------------------------------
# Load C++ output
# ---------------------------------------------------------------------------
try:
    h_cpp = np.loadtxt(TAPS_FILE, dtype=np.float64)
except FileNotFoundError:
    sys.exit(f"ERROR: {TAPS_FILE} not found.\n"
             "Build and run test_proto_fir first:\n"
             "  g++ -std=c++17 -O2 -I../../include -o test_proto_fir "
             "proto_fir.cc test_proto_fir.cc && ./test_proto_fir")

if len(h_cpp) != FILTER_LEN:
    sys.exit(f"ERROR: expected {FILTER_LEN} taps, got {len(h_cpp)}")

print(f"Loaded {len(h_cpp)} taps from {TAPS_FILE}")

# ---------------------------------------------------------------------------
# Design a reference filter with scipy using the same windowed-sinc approach
# ---------------------------------------------------------------------------
fc_norm = 1.0 / (2.0 * NUM_PHASES)   # normalized cutoff (0–0.5 scale)
# scipy.signal.firwin uses cycles/sample normalized to Nyquist = 1.0,
# so fc for firwin = fc_norm * 2
h_scipy = signal.firwin(
    FILTER_LEN,
    cutoff=fc_norm * 2.0,           # scipy Nyquist-normalized: 1.0 = Nyquist
    window="blackmanharris",
    scale=False,                     # don't force unity gain (we want 1/M)
)

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
max_abs_err  = np.max(np.abs(h_cpp - h_scipy))
rms_err      = np.sqrt(np.mean((h_cpp - h_scipy) ** 2))
max_rel_peak = max_abs_err / np.max(np.abs(h_scipy))

print(f"\nComparison vs scipy firwin (blackmanharris):")
print(f"  Max absolute error : {max_abs_err:.3e}")
print(f"  RMS error          : {rms_err:.3e}")
print(f"  Max relative error : {max_rel_peak:.3e}  ({max_rel_peak*100:.4f}%)")

TOLERANCE = 1e-4   # allow small window-coefficient rounding differences
ok_match = max_abs_err < TOLERANCE
print(f"  Match (< {TOLERANCE:.0e})     : {'PASS' if ok_match else 'FAIL'}")

# ---------------------------------------------------------------------------
# Frequency response comparison
# ---------------------------------------------------------------------------
w_cpp,   H_cpp   = signal.freqz(h_cpp,   worN=4096, fs=1.0)
w_scipy, H_scipy = signal.freqz(h_scipy, worN=4096, fs=1.0)

H_cpp_db   = 20 * np.log10(np.abs(H_cpp)   + 1e-15)
H_scipy_db = 20 * np.log10(np.abs(H_scipy) + 1e-15)

fc_hz = fc_norm * SAMPLE_RATE
f_hz  = w_cpp * SAMPLE_RATE   # convert normalized frequency to Hz

# Relevant frequency regions (matching test_proto_fir.cc):
#   Channel passband:  0 to channel_bw/2  (12.5 kHz / 2 = 6.25 kHz)
#   Adjacent channel:  2*fc  (first adjacent channel center = 312.5 kHz)
#   Stopband:          2.5*fc onwards  (past transition band = 8/N at input rate)
channel_edge_hz   = CHANNEL_BW / 2.0
transition_bw_hz  = 8.0 / FILTER_LEN * SAMPLE_RATE
f_stop_hz         = 2.0 * fc_hz + 0.5 * fc_hz   # 2.5*fc, safely into stopband

passband_mask = f_hz < channel_edge_hz
stopband_mask = f_hz > f_stop_hz

pb_ripple_cpp  = np.max(H_cpp_db[passband_mask])  - np.min(H_cpp_db[passband_mask])
sb_atten_cpp   = -np.max(H_cpp_db[stopband_mask])

pb_ripple_sc   = np.max(H_scipy_db[passband_mask]) - np.min(H_scipy_db[passband_mask])
sb_atten_sc    = -np.max(H_scipy_db[stopband_mask])

# Gain at adjacent channel center
adj_idx = np.argmin(np.abs(f_hz - 2.0 * fc_hz))
adj_gain_cpp = H_cpp_db[adj_idx]
adj_gain_sc  = H_scipy_db[adj_idx]

print(f"\nFrequency response (C++ design):")
print(f"  Cutoff frequency       : {fc_hz:,.1f} Hz  ({fc_norm:.6f} normalized)")
print(f"  Transition BW (8/N)    : ~{transition_bw_hz:,.1f} Hz")
print(f"  Passband ripple        : {pb_ripple_cpp:.4f} dB  (0 to {channel_edge_hz:.0f} Hz = channel edge)")
print(f"  Adj. channel gain      : {adj_gain_cpp:.1f} dB  (at {2*fc_hz:,.0f} Hz)")
print(f"  Stopband attenuation   : {sb_atten_cpp:.1f} dB  (>{f_stop_hz:,.0f} Hz)")
print(f"\nFrequency response (scipy reference):")
print(f"  Passband ripple        : {pb_ripple_sc:.4f} dB")
print(f"  Adj. channel gain      : {adj_gain_sc:.1f} dB")
print(f"  Stopband attenuation   : {sb_atten_sc:.1f} dB")

# ---------------------------------------------------------------------------
# Polyphase sanity check
# ---------------------------------------------------------------------------
poly_cpp = h_cpp.reshape(NUM_PHASES, TAPS_PER_PHASE, order='F')  # h[phase + tap*M]
# Reconstruct: h[n] = poly[n % M, n // M]
h_reconstructed = poly_cpp.flatten(order='F')
recon_err = np.max(np.abs(h_reconstructed - h_cpp))
print(f"\nPolyphase round-trip error: {recon_err:.2e}  {'PASS' if recon_err < 1e-10 else 'FAIL'}")

# ---------------------------------------------------------------------------
# Optional plot (skip gracefully if matplotlib not available)
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(h_cpp,   label="C++ (windowed sinc)", lw=1.5)
    ax.plot(h_scipy, label="scipy firwin",         lw=1.0, linestyle="--", alpha=0.7)
    ax.set_title(f"Prototype FIR — {FILTER_LEN} taps, fc={fc_norm:.4f}")
    ax.set_xlabel("Tap index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(f_hz / 1e3, H_cpp_db,   label="C++",   lw=1.5)
    ax.plot(f_hz / 1e3, H_scipy_db, label="scipy", lw=1.0, linestyle="--", alpha=0.7)
    ax.axvline(fc_hz / 1e3, color="red", linestyle=":", label=f"fc = {fc_hz/1e3:.2f} kHz")
    ax.set_xlim(0, 3 * fc_hz / 1e3)
    ax.set_ylim(-120, 5)
    ax.set_title("Frequency Response")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/op25_proto_filter.png", dpi=150)
    print(f"\nPlot saved to /tmp/op25_proto_filter.png")
    plt.show()

except ImportError:
    print("\n(matplotlib not available — skipping plot)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
all_pass = ok_match and sb_atten_cpp > 60.0 and pb_ripple_cpp < 0.1
print(f"\n{'='*50}")
print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
if not ok_match:
    print(f"  FAIL: filter tap mismatch vs scipy (max err {max_abs_err:.2e})")
if sb_atten_cpp <= 60.0:
    print(f"  FAIL: stopband attenuation {sb_atten_cpp:.1f} dB < 60 dB minimum")
if pb_ripple_cpp >= 0.1:
    print(f"  FAIL: channel passband ripple {pb_ripple_cpp:.4f} dB >= 0.1 dB")
sys.exit(0 if all_pass else 1)
