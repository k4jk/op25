# Building OP25 with the CUDA Polyphase Channelizer

The CUDA channelizer replaces the per-channel GNU Radio demodulation chain with a
single GPU pipeline that processes the full SDR bandwidth in one pass. A single
wideband capture (e.g. 10 MHz from a B200) is polyphase-filtered and FFT-binned on
the GPU; individual P25 channels are extracted, clock-recovered, and delivered as
dibit streams to the existing `frame_assembler` / `rx_sync` decoder stack.

---

## Hardware Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | Any CUDA-capable (compute 6.0+) | NVIDIA GeForce GTX 1080 (compute 6.1) |
| SDR | Wideband receiver (≥ 5 MHz) | USRP B200 Mini |
| Driver | NVIDIA driver 525+ | — |
| CUDA Toolkit | 11.8+ | 12.8 |

The GPU does not need to be powerful — the pipeline is memory-bandwidth bound at
modest channel counts. A GTX 1080 handles 20+ simultaneous P25 channels with ease.

---

## Software Dependencies

### CUDA Toolkit

Install from the [NVIDIA CUDA repository](https://developer.nvidia.com/cuda-downloads).
The toolkit provides `nvcc`, `libcufft`, and the CUDA runtime:

```bash
# Ubuntu 22.04 / 24.04 — add NVIDIA repo then:
sudo apt-get install cuda-toolkit-12-8
```

Verify the install:

```bash
nvcc --version
nvidia-smi
```

### GNU Radio and OP25 Base Dependencies

The CUDA build requires the same base dependencies as the standard OP25 build:

```bash
sudo apt-get install \
    gnuradio gnuradio-dev \
    gr-osmosdr \
    libuhd-dev uhd-host \
    cmake build-essential pkg-config \
    python3-pybind11 python3-numpy \
    python3-waitress python3-requests \
    libsndfile1-dev libspdlog-dev \
    gcc-12 g++-12
```

> **Note:** The CUDA host compiler must be compatible with your CUDA toolkit version.
> CUDA 12.x requires GCC ≤ 13. On Ubuntu 24.04, `gcc-12` is recommended and is what
> the working build uses (`CMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12`).

---

## Build

The CUDA channelizer is built as part of `gr-op25_repeater`. It is **disabled by
default** and enabled with `-DENABLE_CUDA=ON`. Use a separate build directory from
the standard build to avoid mixing artifacts.

### 1. Create a build directory

```bash
cd op25/gr-op25_repeater
mkdir build_cuda && cd build_cuda
```

### 2. Configure with CMake

```bash
cmake .. \
    -DENABLE_CUDA=ON \
    -DENABLE_PYTHON=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_INSTALL_PREFIX=/usr/local
```

| CMake Option | Default | Description |
|---|---|---|
| `ENABLE_CUDA` | `OFF` | Build the GPU channelizer. Requires CUDA toolkit. |
| `ENABLE_PYTHON` | — | Build Python bindings (required for `multi_rx.py`). |
| `CMAKE_BUILD_TYPE` | `Release` | Use `Release` for `-O3` CUDA kernels. |
| `CMAKE_CUDA_ARCHITECTURES` | — | `native` compiles for the installed GPU. Use a specific value (e.g. `61`) for cross-compilation or to avoid the `native` detection overhead. |
| `ENABLE_CUDA_DIAG` | `OFF` | Adds per-batch device-to-host reads for pipeline diagnostics. **Do not enable in production** — it serializes the GPU pipeline and significantly reduces throughput. |
| `P25_ONLY` | `OFF` | Restrict the frame sync correlator to P25 only (skips DMR, YSF, D-STAR). Minor CPU savings if you only run P25 sites. |
| `CMAKE_INSTALL_PREFIX` | `/usr/local` | Install root. Must match where GNU Radio looks for OOT modules. |

`CMAKE_CUDA_ARCHITECTURES=native` is the easiest choice — CMake queries the
installed GPU and compiles for exactly that architecture. If you are building on a
different machine than the one running OP25, set the architecture explicitly:

```bash
# Compute 6.1 (GTX 1080, GTX 1080 Ti, Quadro P5000, etc.)
-DCMAKE_CUDA_ARCHITECTURES=61

# Compute 8.6 (RTX 3080, RTX 3090, etc.)
-DCMAKE_CUDA_ARCHITECTURES=86
```

### 3. Build and install

```bash
make -j$(nproc)
sudo make install
sudo ldconfig
```

Only `libgnuradio-op25_repeater` needs to be rebuilt when modifying the C++/CUDA
source. To rebuild just that target after a source change:

```bash
make -j$(nproc) gnuradio-op25_repeater
sudo make install
sudo ldconfig
```

---

## Verify the Build

After installation, confirm the CUDA block is registered with GNU Radio:

```bash
python3 -c "import gnuradio.op25_repeater as op25r; print(dir(op25r))" | tr ',' '\n' | grep cuda
```

You should see `cuda_channelizer` in the output.

---

## Configuration

### channelizer.json

The GPU pipeline is configured by `channelizer.json` (located in the `apps/`
directory alongside your run script). The path is referenced by the `"channelizer"`
key in your main config file.

```json
{
    "channelizer": {
        "channel_bw_hz":  12500,
        "taps_per_phase": 25,
        "control_port":   23457,
        "status_port":    23458
    }
}
```

| Field | Description |
|---|---|
| `channel_bw_hz` | Per-channel bandwidth in Hz. 12500 for standard P25. |
| `taps_per_phase` | Polyphase filter taps per phase (per FFT bin). **Must be an odd integer.** See note below. |
| `control_port` | UDP port for the runtime channel-control interface. |
| `status_port` | UDP port for status queries. |

> **`taps_per_phase` must be odd.** The K-parity theorem for oversampled polyphase
> banks (when `fft_oversample` > 1) states that an even K produces zero response at
> all odd FFT bins, silencing every other channel. K=25 gives odd bins −3.92 dB
> relative to even bins — the best achievable for any odd K. Never round down to an
> even number.

### Main config file (device section)

Add `"cuda": true` and `"fft_oversample"` to your device entry to route it through
the GPU pipeline:

```json
{
    "devices": [
        {
            "name": "sdr0",
            "args": "uhd,nchan=1,subdev=A:A,num_recv_frames=64,recv_frame_size=16360",
            "gains": "PGA:67",
            "frequency": 855587500,
            "rate": 10000000,
            "usable_bw_pct": 0.95,
            "cuda": true,
            "fft_oversample": 1,
            "ppm": 2.3,
            "tunable": false
        }
    ],
    "channelizer": {
        "config_path": "channelizer.json",
        "max_channels": 20
    }
}
```

| Field | Description |
|---|---|
| `cuda` | `true` to route this device through the GPU channelizer. |
| `fft_oversample` | FFT oversampling factor. `1` = critically sampled (no odd-bin penalty, recommended for most deployments). `2` = 2× oversampled (improves channel isolation at the cost of the odd-bin SNR penalty and higher GPU memory usage). |
| `max_channels` | Maximum simultaneous voice channels the channelizer will allocate slots for. |

### fft_oversample guidance

- **`fft_oversample: 1`** — Recommended for most deployments. No odd-bin SNR
  penalty. Each FFT bin corresponds to exactly one 12.5 kHz channel.
- **`fft_oversample: 2`** — Doubles GPU memory and compute. Channels that land on
  odd FFT bins suffer −3.92 dB SNR. Useful if channel isolation is a concern and
  you can tolerate the SNR tradeoff. Requires careful center frequency selection
  to place control channels on even bins.

---

## Running

Launch `multi_rx.py` exactly as for the standard build — the CUDA path is activated
automatically based on the `"cuda": true` device flag:

```bash
cd op25/gr-op25_repeater/apps
python3 multi_rx.py -c cfg_alb_b200_cuda.json -v 0 2>stderr.log
```

On startup you should see lines like:

```
cuda_channelizer: M=800  fft_size=800  input_len=160000  s2_len=127  ctrl_port=23457  status_port=23458
cuda_channelizer: slot 0 → bin 219  (855587500 Hz)  mode=3 tdma=0  derot_step=0.000000 rad/samp
cuda_channelizer: warmup ARMED bin=219 mode=3 steps=126 fft=800
cuda_channelizer: warmup EXPIRED bin=219 slot=0
```

---

## Troubleshooting

**`libgnuradio-op25_repeater.so: undefined symbol: cudaMemcpy`**
The library was installed but the CUDA runtime is not on the linker path. Run
`sudo ldconfig` and confirm `/usr/local/cuda/lib64` (or equivalent) is in
`/etc/ld.so.conf.d/`.

**`cuda_channelizer: not compiled with HAVE_CUDA`**
The block was built without `-DENABLE_CUDA=ON`, or the wrong installed library
is being loaded. Check `ldconfig -p | grep op25`.

**`cuda_channelizer: GPU pipeline initialisation failed`**
Check that `channelizer.json` exists at the path specified by `"config_path"` in
your main config, and that the JSON is valid.

**`SYNC_REV_P FIRST HIT slot=N — INVERTED POLARITY: rx_sync will never match!`**
The dibit polarity coming out of the GPU is inverted for that slot. This usually
means the center frequency choice placed the channel on an odd FFT bin with
`fft_oversample=1` and an odd `taps_per_phase`. Verify your `taps_per_phase` is
odd (e.g. 25) and consider adjusting center frequency.

**Channels time out immediately after grant with no audio**
Confirm `rx_q` in `multi_rx.py` is set to `gr.msg_queue(1000)` (not 100). The
smaller queue silently drops timeout messages during control channel floods.
