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
    // d_input is a padded buffer: [(K-1)*M + input_len] complex samples.
    // The first (K-1)*M samples are the FIR history from the previous block;
    // the next input_len samples are overwritten with new IQ on each call.
    cufftComplex*   d_input;          // padded IQ buffer on GPU
    int             input_len;        // new samples per processing block (must be multiple of M)
    float*          d_proto_filter;   // prototype FIR coefficients on GPU (reference copy)
    float*          d_poly_phases;    // polyphase matrix [num_phases * taps_per_phase], row-major
    cufftComplex*   d_phase_out;      // intermediate: [input_len] phase filter outputs (L*M)
    cufftHandle     fft_plan;         // cuFFT M-point FFT, batch size = input_len / num_phases
#endif
};
