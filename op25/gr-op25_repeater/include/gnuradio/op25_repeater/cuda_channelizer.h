/* -*- c++ -*- */
/*
 * Copyright 2024 OP25 Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_H
#define INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_H

#include <gnuradio/op25_repeater/api.h>
#include <gnuradio/block.h>

namespace gr {
namespace op25_repeater {

/*!
 * \brief GPU-accelerated wideband polyphase analysis channelizer.
 *
 * Splits one wideband SDR IQ stream into up to max_channels P25 dibit streams
 * via: polyphase filter bank → stage-2 per-channel FIR decimation →
 *      FM discriminator → Mueller-Müller clock recovery → 4-level C4FM slicer.
 *
 * Input:  1 × complex float (gr_complex) at sdr_sample_rate_hz
 * Output: max_channels × uint8_t dibit streams (0-3), step-major per channel
 *
 * Channel assignment is dynamic; call set_channel() / clear_channel() at runtime
 * (thread-safe) to map output port indices to polyphase bin frequencies.
 *
 * Only available when compiled with -DENABLE_CUDA=ON.  Constructing the block
 * without CUDA support throws std::runtime_error.
 */
class OP25_REPEATER_API cuda_channelizer : virtual public gr::block
{
public:
    typedef std::shared_ptr<cuda_channelizer> sptr;

    /*!
     * \param config_path  Path to channelizer.json configuration file.
     * \param max_channels Number of simultaneous channel slots (output ports).
     * \param debug        Verbosity level (0 = silent).
     */
    static sptr make(const std::string& config_path,
                     int max_channels = 20,
                     int debug = 0);

    /*!
     * \brief Assign a channel slot to a polyphase bin.
     *
     * Maps output port \p slot to the polyphase bin closest to
     * \p center_freq_hz.  Dibits for that bin will appear on output port
     * \p slot.  Thread-safe.
     */
    virtual void set_channel(int slot, float center_freq_hz) = 0;

    /*!
     * \brief Deactivate a channel slot.
     *
     * Output port \p slot will emit zeros until set_channel() is called again.
     * Thread-safe.
     */
    virtual void clear_channel(int slot) = 0;

    /*!
     * \brief Allocate the next free slot and assign it to center_freq_hz.
     *
     * \returns Allocated slot index (0 … max_channels-1), or -1 if all slots
     *          are in use.  Thread-safe.
     */
    virtual int alloc_slot(float center_freq_hz) = 0;

    /*!
     * \brief Release a previously-allocated slot.  Thread-safe.
     */
    virtual void free_slot(int slot) = 0;
};

} // namespace op25_repeater
} // namespace gr

#endif /* INCLUDED_OP25_REPEATER_CUDA_CHANNELIZER_H */
