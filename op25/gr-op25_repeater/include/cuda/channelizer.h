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

    // Derived — set by compute_config()
    int      stage1_decimation;      // largest power-of-2 factor of total decimation, max 64
    int      stage2_decimation;      // total_decimation / stage1_decimation
    int      num_phases;             // == stage1_decimation (also the FFT size)
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
    cufftComplex*   d_phase_out;      // intermediate + stage-1 output: [L * M], step-major
    cufftHandle     fft_plan;         // cuFFT M-point forward FFT, batch = L

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
    float*          d_fm_filtered;    // filtered FM output [(L/D) * M], step-major (input to Gardner)

    // ---- Symbol timing recovery (Mueller-Müller clock recovery) ----
    float*          d_mm_mu;          // fractional timing position [M], range ≈ [-(sps-1), sps)
    float*          d_mm_fm_last;     // last 3 FM samples from previous block [3*M]: [k]=fm(-1), [M+k]=fm(-2), [2M+k]=fm(-3)
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
    int8_t*         d_channel_mode;   // per-bin mode [M]: 1=Phase1, 2=Phase2, 0=unset
    cufftComplex*   d_p2_iq_last;    // cross-block complex IQ history [3*M]: indices -1,-2,-3
    cufftComplex*   d_p2_last_sym;   // previous decoded complex symbol [M] for differential decode
#endif
};
