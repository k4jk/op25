/*
 * Copyright 2024 OP25 Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * pybind11 binding for gr::op25_repeater::cuda_channelizer.
 * Only available when the library was built with ENABLE_CUDA=ON.
 */

/* BINDTOOL_GEN_AUTOMATIC(0) */
/* BINDTOOL_USE_PYGCCXML(0) */
/* BINDTOOL_HEADER_FILE(cuda_channelizer.h) */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/op25_repeater/cuda_channelizer.h>

void bind_cuda_channelizer(py::module& m)
{
    using cuda_channelizer = ::gr::op25_repeater::cuda_channelizer;

    py::class_<cuda_channelizer, gr::block, gr::basic_block,
               std::shared_ptr<cuda_channelizer>>(m, "cuda_channelizer")

        .def(py::init(&cuda_channelizer::make),
             py::arg("config_path"),
             py::arg("max_channels") = 20,
             py::arg("debug") = 0)

        .def("set_channel", &cuda_channelizer::set_channel,
             py::arg("slot"), py::arg("center_freq_hz"),
             "Map output port slot to the polyphase bin closest to center_freq_hz.")

        .def("clear_channel", &cuda_channelizer::clear_channel,
             py::arg("slot"),
             "Deactivate a channel slot (output port emits zeros).");
}
