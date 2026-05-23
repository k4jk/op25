# Building OP25 with the CUDA Polyphase Channelizer

The CUDA channelizer replaces the per-channel GNU Radio demodulation chain with a
single GPU pipeline that processes full SDR bandwidths in one pass. One or more
wideband captures are polyphase-filtered and FFT-binned on the GPU; individual P25 channels are extracted, clock-recovered, and delivered as dibit streams to the existing `frame_assembler` / `rx_sync` decoder stack.

---

## Hardware Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | Any CUDA-capable (compute 6.0+) | NVIDIA GeForce GTX 1080 (compute 6.1) |
| SDR | Wideband receiver (≥ 5 MHz) | USRP B200 Mini, Airspy R2 |
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

### 1. Checkout CUDA branch & create a build directory

```bash
git checkout feature/cuda-channelizer
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

The GPU pipeline core is configured by `channelizer.json` (located in the `apps/`
directory alongside your other configs). The path is referenced by the `"channelizer"`
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
            "args": "uhd,nchan=1,subdev=A:A,num_recv_frames=64 recv_frame_size=16360",
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
| `fft_oversample` | FFT oversampling factor. `1` = critically sampled 12.5 khz bins (no odd-bin penalty, recommended for most deployments). `2` = 2× oversampled (improves channel isolation at the cost of the odd-bin SNR penalty and higher GPU memory usage). See section below on FFT oversampling.|
| `max_channels` | Maximum simultaneous voice channels the channelizer will allocate slots for. |

Note: You may need to play with device buffering in the device args to prevent USB overflow, depending on your setup.

Example for UHD Devices: Something like num_recv_frames=64,recv_frame_size=16360

### Channel entries: CC designation and voice slot count

Each entry in the `"channels"` array that shares a `trunking_sysname` with a CUDA
device becomes either the **control channel** slot or a **voice pool** slot,
depending on whether it carries `"cc": true`.

#### Why `"cc": true` matters

When `multi_rx.py` starts, `configure_cuda_voice_pool()` inspects all channel
entries on each CUDA device and assigns roles:

- The channel with `"cc": true` for a given `trunking_sysname` is marked
  **CC-dedicated** (`cc_dedicated = True`). This slot is permanently locked to
  control channel duty — the trunking layer will never reassign it to decode a
  voice call, even if the system is quiet.
- All other channels on the same device and sysname are marked **voice pool**
  (`cc_capable = False`, `tuner_idle = True`). These slots sit idle until the
  trunking layer assigns them a grant.

Without `"cc": true`, the code falls back to using the lowest-indexed slot for
that sysname as the CC and logs a warning:

```
WARNING: no cc:true channel found for sysname=BIGSYSTEM on device=sdr0,
defaulting to slot 0 (msgq_id=0) — add "cc": true to your CC channel config
```

The fallback works for single-system configs, as long as you also have a voice channel. But it becomes unreliable when you run multiple trunked systems on a single wideband SDR. In that case, each system needs its own dedicated CC slot, and without explicit `"cc": true` markers the code has no way to know which slot belongs to which system's control channel.

Marking the CC explicitly also makes the config self-documenting and prevents
subtle bugs if channel list order ever changes.

```json
{
    "channels": [
        {
            "name": "BIGSYSTEM CC",
            "device": "sdr0",
            "trunking_sysname": "BIGSYSTEM",
            "cc": true,
            "demod_type": "cqpsk"
        },
        {
            "name": "BIGSYSTEM Law Enforcement",
            "device": "sdr0",
            "trunking_sysname": "BIGSYSTEM",
            "destination": "ws://0.0.0.0:9009",
            "demod_type": "cqpsk",
            "whitelist": "BIGSYSTEM_LE.wlist"
        },
        {
            "name": "BIGSYSTEM Fire-Rescue",
            "device": "sdr0",
            "trunking_sysname": "BIGSYSTEM",
            "destination": "ws://0.0.0.0:9010",
            "demod_type": "cqpsk",
            "whitelist": "BIGSYSTEM_FR.wlist"
        }
    ]
}
```

#### Choosing the number of voice slots

The number of voice pool slots is simply the number of non-CC channel entries you
add for a given sysname. There is no separate counter — each channel entry becomes
one decoder slot. The right number depends on the busy-hour traffic of the system
you are monitoring:

- **One slot per simultaneous call you want to hear.** A system that routinely has
  3–4 simultaneous voice grants needs at least 3–4 voice entries, if you want to hear them all.
- **Slots consume GPU bins continuously** (once a grant is assigned, the GPU keeps
  that FFT bin active until the call ends). More slots means more GPU load, but on
  any modern GPU the per-slot cost is very small.
- **`max_channels`** in the channelizer config sets the hard GPU-side ceiling on
  how many bins can be active at once. The number of channel entries you configure
  should not exceed `max_channels - (number of CC slots)`.
- **Unassigned slots sit completely idle** — a voice pool slot that hasn't received
  a grant does not process any samples and adds no load.

A reasonable starting point is 4–6 voice entries for a moderately busy P25 system.
You can add more entries (and rebuild nothing — it is pure config) if you observe
grants being dropped because all slots are occupied.

### fft_oversample guidance

- **`fft_oversample: 1`** — Recommended for most deployments. No odd-bin SNR
  penalty. Each FFT bin corresponds to exactly one 12.5 kHz channel.
- **`fft_oversample: 2`** — Can be used when 6.25 khz channelization is needed. Doubles GPU memory and compute. Additionally, channels that land on ODD FFT bins suffer −3.92 dB SNR. Mainly useful on 700mhz when using a wideband device to monitor multiple systems. Essentially, oversampling for 6.25 mhz channelization can allow you to monitor multiple frequencies on a single device when they do not fall neatly in multiples of 12.5 khz.

 Before resorting to oversampling, you may be able to place all channels on even 12.5 khz bins with careful center frequency selection in your SDR device config. It depends on your exact situation, and the systems you are trying to monitor. With critical sampling, the channelizer will create 12.5 khz bins by default to cover the entire sample bandwidth, based on the the center frequency & offset set in the device config.

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

**Architecture Diagram**

══════════════════════════════════════════════════════════════
                        GPU SIDE
══════════════════════════════════════════════════════════════

SDR wideband IQ streams (any GNU Radio compatible devices)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│            CUDA Polyphase Channelizer               │
│                                                     │
│  • Channel extraction (polyphase filter bank)       │
│  • Nyquist carrier removal (decarrier — odd bins)   │
│  • Per-bin frequency derotation (Δf correction)     │
│  • Per-channel S2 FIR filtering + decimation        │
│                                                     │
│  MODE 1 — FM/FSK4 path (fsk4 demod_type):          │
│  • IIR DC blocker (alpha=0.99)                      │
│  • FM demodulation                                  │
│  • C4FM matched filter                              │
│  • Gardner TED (real FM samples, raw IQ S-curve)    │
│  • FSK4 slicer → dibits                             │
│                                                     │
│  MODE 2/3 — CQPSK path (cqpsk demod_type):         │
│  • IQ Gardner TED (complex, raw IQ S-curve)         │
│  • MMSE FIR interpolator (GNU Radio coefficients)   │
│  • Differential IQ decoder (curr × conj(prev))      │
│  • Costas loop (4th-order, α=0.008)                 │
│  • DQPSK slicer → dibits                            │
│                                                     │
│  Both paths:                                        │
│  • 8-tap MMSE FIR interpolator                      │
│  • MM_HIST=12 history samples                       │
│  • Warmup gate (126-step freeze on channel assign)  │
│  • Raw P25 dibit stream output                      │
└─────────────────────────────────────────────────────┘
        │
        │  raw P25 symbols (tiny bandwidth — ~24 KB/s total)
        ▼
══════════════════════════════════════════════════════════════
                        CPU SIDE  (OP25 — unchanged)
══════════════════════════════════════════════════════════════
        │
        ▼
op25_repeater.frame_assembler — frame assembly, error correction
        │
trunking.py              — protocol state machine, grants/releases
        │
audio output             — IMBE/AMBE vocoder, playback
══════════════════════════════════════════════════════════════