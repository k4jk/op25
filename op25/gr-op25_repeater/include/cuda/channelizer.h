// Copyright 2026, James Kirkham K4JK
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <mutex>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#endif

// Runtime configuration — five primary fields set from channelizer.json,
// all derived fields populated by compute_config().
struct ChannelizerConfig {
    // Primary (from JSON)
    float    sdr_sample_rate_hz;
    float    sdr_center_freq_hz;
    float    channel_bw_hz;
    int      max_channels;
    int      taps_per_phase;
    int      control_port;
    int      status_port;

    // Primary (from JSON, optional)
    int      fft_oversample;         // 1 = critically sampled (default), 2 = half-bin resolution
                                     //
                                     // KNOWN LIMITATION — fft_oversample=2 odd-bin SNR penalty:
                                     // The 2M-point FFT is applied to M polyphase outputs zero-padded
                                     // to 2M.  Odd-indexed bins (k=1,3,5,...) sit at frequencies
                                     // halfway between the natural polyphase grid points.  With K=25
                                     // (odd taps/phase), odd bins receive -3.92 dB less signal power
                                     // than even bins — numerically verified, irreducible for any odd K.
                                     // Even K makes odd bins completely silent (-290+ dB); K=25 is
                                     // the best achievable.  fft_oversample=1 has no such penalty.
                                     //
                                     // Operator implication: choose sdr_center_freq_hz so that the
                                     // most critical system's channels land on even bins (offset from
                                     // center is an even multiple of channel_bw_hz/2).  Shifting
                                     // center by channel_bw_hz/2 flips all parities simultaneously.
                                     // See the "#center_freq_hz" note in channelizer.json for the
                                     // per-system calculation.

    // Derived — set by compute_config()
    int      stage1_decimation;      // == total_decimation == num_phases (stage2 is always 1)
    int      stage2_decimation;      // always 1: full decimation is done in stage 1
    int      num_phases;             // polyphase branch count == total_decimation
    int      fft_size;               // FFT length == num_phases * fft_oversample
                                     //   fft_oversample=1: fft_size==num_phases (critically sampled)
                                     //   fft_oversample=2: fft_size==2*num_phases, bin spacing halved
    int      prototype_filter_len;   // num_phases * taps_per_phase
    float    stage1_output_rate_hz;  // sdr_sample_rate_hz / stage1_decimation
    float    stage2_output_rate_hz;  // stage1_output_rate_hz / stage2_decimation (== channel_bw_hz)
};

// Populate all derived fields from the primary fields.
// Returns false (and writes a message to error_msg if non-null) if the
// parameters are invalid (e.g. non-integer decimation ratio).
bool compute_config(ChannelizerConfig& cfg, char* error_msg = nullptr);

// One voice (or control) channel slot.
struct Channel {
    bool      active;
    float     center_freq_hz;   // absolute frequency
    float     offset_hz;        // offset from SDR center frequency
    int       system_id;
    uint32_t  talkgroup;
    int       tdma_slot;        // 0 or 1 for Phase 2, always 0 for Phase 1
    int       p25_mode;         // 1=Phase1 (C4FM), 2=Phase2 (H-DQPSK TDMA)

#ifdef HAVE_CUDA
    cufftComplex*  d_stage1_out;    // polyphase channelizer output buffer
    cufftComplex*  d_stage2_out;    // stage-2 decimation output buffer
    float*         d_audio_out;     // FM-demodulated audio buffer
    cufftComplex*  d_filter_state;  // stage-2 FIR filter state
    cudaStream_t   stream;          // independent CUDA stream for this slot
#endif
};

// Top-level channelizer state, one instance per SDR.
struct ChannelizerState {
    ChannelizerConfig           config;
    std::array<Channel, 20>     channels;
    std::mutex                  channel_mutex;

#ifdef HAVE_CUDA
    // ---- Stage 1: polyphase channelizer ----
    // d_input is a padded buffer: [(K-1)*M + input_len] complex samples.
    // The first (K-1)*M samples are the FIR history from the previous block;
    // the next input_len samples are overwritten with new IQ on each call.
    cufftComplex*   d_input;          // padded IQ buffer on GPU
    int             input_len;        // new samples per block (multiple of M); L = input_len/M
    float*          d_proto_filter;   // prototype FIR coefficients on GPU (reference copy)
    float*          d_poly_phases;    // polyphase matrix [M * K], row-major (phase × tap)
    cufftComplex*   d_phase_out;      // intermediate + stage-1 output: [L * fft_size], step-major
                                      // zeroed at alloc; poly kernel only writes [0..num_phases-1]
                                      // per step, leaving the padding zeros for the oversampled FFT
    cufftHandle     fft_plan;         // cuFFT fft_size-point forward FFT, batch = L

    // ---- Stage 2: per-channel decimating FIR ----
    // Padded input: [(N2-1 + L) * M] step-major.  First (N2-1)*M are history.
    float*          d_s2_coeff;       // stage-2 FIR coefficients [s2_filter_len]
    int             s2_filter_len;    // N2 (number of stage-2 FIR taps)
    cufftComplex*   d_s2_padded;      // padded stage-1 output for all M channels
    cufftComplex*   d_s2_output;      // decimated output [(L/D) * M], step-major

    // ---- FM demodulator ----
    cufftComplex*   d_fm_prev;        // previous sample per channel [M] for ∆θ discriminator
    float*          d_fm_output;      // demodulated frequency [(L/D) * M], step-major (radians/sample)

    // ---- C4FM matched receive filter ----
    float*          d_c4fm_history;   // cross-block FIR history [18 * M] (last 18 FM samples of prev batch)
    float*          d_fm_filtered;    // filtered FM output [(L/D) * M], step-major (input to DC blocker)

    // ---- DC blocker (applied to d_fm_filtered in-place before Gardner) ----
    // IIR highpass: y[n] = x[n] - x[n-1] + alpha*y[n-1], alpha=0.99
    // cutoff ≈ 20 Hz at 12500 sps; settles in ~100 samples after reset.
    // State zeroed on channel reset (apply_pending_mm_resets).
    float*          d_dcb_x_prev;     // previous FM input sample [C]
    float*          d_dcb_y_prev;     // previous DC-blocked output [C]

    // ---- Symbol timing recovery (Mueller-Müller clock recovery) ----
    float*          d_mm_mu;          // fractional timing position [M], range ≈ [-(sps-1), sps)
    float*          d_mm_fm_last;     // last 8 FM samples from previous block [8*M]: [k]=fm(-1), [M+k]=fm(-2), ..., [7M+k]=fm(-8)
    float*          d_mm_last_interp; // previous interpolated symbol value [M] (M&M error term)
    float*          d_mm_last_dec;    // previous slicer decision [M] (M&M error term)
    float*          d_mm_symbols;     // interpolated symbol values [MM_MAX_SYM * M], step-major
    int8_t*         d_mm_dibits;      // decoded dibits 0-3 [MM_MAX_SYM * M], step-major
    int32_t*        d_mm_sym_count;   // valid symbol count per channel [M]
    float*          d_mm_dc_est;      // IIR carrier DC estimate per channel [M]
    float*          d_mm_omega;       // samples-per-symbol estimate per channel [M] (Gardner rate tracking)
    int32_t*        d_mm_fast_ctr;   // fast-gain countdown per channel [M] (symbols remaining)

    // ---- Phase 2 TDMA (H-DQPSK) complex Gardner timing recovery ----
    // Phase 2 channels bypass the FM discriminator and C4FM filter.  The
    // complex Gardner loop reads directly from d_s2_output and performs
    // differential decoding at the symbol level (curr × conj(prev)) so that
    // the decoded phase transitions are exactly ±π/4 or ±3π/4 regardless of
    // carrier offset — DC removed via d_mm_dc_est (shared with Phase 1).
    int8_t*         d_channel_mode;   // per-bin mode [M]: 1=Phase1 FM/FSK4, 2=Phase2 H-DQPSK, 3=Phase1 CQPSK
    cufftComplex*   d_p2_iq_last;    // cross-block complex IQ history [8*M]: indices -1 through -8
    cufftComplex*   d_p2_last_sym;   // previous decoded complex symbol [M] for differential decode

    // Per-bin Gardner warmup gate: 1=suppress Gardner updates (S2 history still
    // filling after channel reset), 0=run normally.  Set to 1 in
    // apply_pending_mm_resets, cleared to 0 when d_bin_warmup_steps[k] expires.
    int8_t*         d_mm_in_warmup;  // [M]

    // ---- IQ derotation (modes 2 and 3: complex Gardner path) ----
    // Corrects the residual carrier offset Δf = (channel_freq - bin_center_freq)
    // left by the polyphase channelizer.  Each sample read from d_s2_output is
    // rotated by exp(j × (derot_acc + n × derot_step)) before the Gardner sees it.
    //
    // derot_step = -2π × Δf / channel_bw_hz   [rad/sample, set on channel assignment]
    // derot_acc  = accumulated rotation phase at the start of the current batch [rad]
    //
    // Without this correction, a carrier offset of more than ~1 kHz creates a
    // 4-way phase ambiguity in the differential decode that dc_est cannot resolve,
    // causing systematic dibit errors and preventing P25 sync acquisition.
    // Mode 1 (FM/FSK4) bins leave these at 0 (FM discriminator handles offset natively).
    float*          d_iq_derot_step;  // [C] per-sample derotation phase step [rad/sample]
    float*          d_iq_derot_acc;   // [C] accumulated derotation phase at batch start [rad]

    // ---- Costas loop (mode 3 Phase 1 CQPSK only) ----
    // 4th-order phase detector: error = sgn(I)*Q - sgn(Q)*I applied to the
    // normalized differential phasor.  Breaks the Gardner S-curve symmetry
    // between symbol-center and symbol-midpoint zero-crossings.
    // alpha = 0.008 (matches digital.costas_loop_cc(alpha=0.008, order=4)).
    float*          d_costas_phase;     // [C] accumulated phase correction [rad]; 0 for modes 1/2
    float*          d_costas_freq;      // [C] frequency integrator [rad/symbol]; 0 for modes 1/2
    int32_t*        d_costas_slip_ctr;  // [C] signed slip-detection counter; snaps phase on ±SLIP_WIN
#endif
};
