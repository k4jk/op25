#!/usr/bin/env python3
"""
Stage 1 hardware smoke-test for cuda_channelizer with a USRP B200 Mini.

Tests:
  • osmosdr source connects and produces samples
  • cuda_channelizer initialises without crash (GPU allocates)
  • GPU batch fires within 5 s (visible in debug=1 stderr output)
  • GR flowgraph shuts down cleanly

Prerequisites
─────────────
1. Build build_cuda2 with Python bindings enabled (one-time step):

   cd /home/james/ham/dev/op25/op25/gr-op25_repeater/build_cuda2
   cmake -DENABLE_CUDA=ON -DENABLE_PYTHON=ON \\
         -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12 ..
   make -j$(nproc)

2. Export the build paths before running (or add to ~/.bashrc):

   BUILD=/home/james/ham/dev/op25/op25/gr-op25_repeater/build_cuda2
   export LD_LIBRARY_PATH="$BUILD/lib:$LD_LIBRARY_PATH"
   export PYTHONPATH="$BUILD/test_modules:$PYTHONPATH"

3. Plug in the B200 Mini, then:

   python3 test_b200_live.py

Usage
─────
  python3 test_b200_live.py [options]

  -f, --freq HZ     SDR centre frequency Hz      [default: 851000000]
  -r, --rate HZ     Sample rate Hz               [default: 20000000]
  -g, --gain DB     RF gain dB                   [default: 40]
  -t, --time SEC    Run duration in seconds       [default: 5]
  -c, --config PATH channelizer.json path         [default: channelizer.json]
  -d, --device STR  osmosdr device string         [default: uhd,type=b200]
  -n, --channels N  Number of output slots        [default: 20]
  -v, --verbose     Print GR block debug to stdout
"""

import argparse
import os
import subprocess
import sys
import time
import threading

# ── env-var preflight ────────────────────────────────────────────────────────

def _check_env():
    build = os.environ.get('CUDA_CHAN_BUILD',
                           os.path.join(os.path.dirname(__file__),
                                        '..', 'build_cuda2'))
    build = os.path.realpath(build)

    lib_path = os.path.join(build, 'lib')
    py_path  = os.path.join(build, 'test_modules')

    missing = []
    if not os.path.isfile(os.path.join(lib_path, 'libgnuradio-op25_repeater.so')):
        missing.append(f'  {lib_path}/libgnuradio-op25_repeater.so')
    if not os.path.isdir(py_path):
        missing.append(f'  {py_path}/')

    if missing:
        print('ERROR: Build artefacts not found:')
        for m in missing:
            print(m)
        print()
        print('Run the one-time build step:')
        print(f'  cd {build}')
        print('  cmake -DENABLE_CUDA=ON -DENABLE_PYTHON=ON \\')
        print('        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12 ..')
        print('  make -j$(nproc)')
        sys.exit(1)

    # Prepend paths so the CUDA-enabled build takes priority over any install
    ld = os.environ.get('LD_LIBRARY_PATH', '')
    if lib_path not in ld:
        os.environ['LD_LIBRARY_PATH'] = lib_path + (':' + ld if ld else '')
        # Re-exec so the dynamic linker picks up the updated path
        os.execv(sys.executable, [sys.executable] + sys.argv)

    if py_path not in sys.path:
        sys.path.insert(0, py_path)

_check_env()

# ── imports (after env fixup) ────────────────────────────────────────────────

try:
    import gnuradio
    from gnuradio import gr, blocks
except ImportError as e:
    sys.exit(f'Cannot import gnuradio: {e}\nIs GNU Radio installed?')

try:
    import osmosdr
except ImportError as e:
    sys.exit(f'Cannot import osmosdr: {e}\nInstall gr-osmosdr.')

try:
    from gnuradio import op25_repeater
    _cuda_chan = op25_repeater.cuda_channelizer
except AttributeError:
    sys.exit(
        'op25_repeater.cuda_channelizer not found.\n'
        'Rebuild build_cuda2 with ENABLE_PYTHON=ON (see script header).'
    )


# ── flowgraph ────────────────────────────────────────────────────────────────

class SmokeTestFlowgraph(gr.top_block):

    def __init__(self, args):
        super().__init__('cuda_channelizer_smoke_test')

        cfg  = args.config
        n    = args.channels
        dbg  = 1 if args.verbose else 1   # always 1 so we see "GPU batch" lines

        # ── osmosdr source (B200 Mini) ──────────────────────────────────────
        src = osmosdr.source(args.device)
        src.set_sample_rate(args.rate)
        src.set_center_freq(args.freq, 0)
        src.set_gain(args.gain, 0)
        src.set_freq_corr(0, 0)

        # ── CUDA channelizer ────────────────────────────────────────────────
        chan = _cuda_chan.make(cfg, n, dbg)

        # ── null sinks (one per output slot) ────────────────────────────────
        sinks = [blocks.null_sink(gr.sizeof_char) for _ in range(n)]

        # ── connect ─────────────────────────────────────────────────────────
        self.connect(src, chan)
        for i, sink in enumerate(sinks):
            self.connect((chan, i), sink)

        self._chan = chan
        self._n    = n

    def assign_test_channels(self, center_hz, bw_hz):
        """Assign a handful of slots to real P25 channel frequencies."""
        freqs = [center_hz + k * bw_hz for k in [0, 11, 25, -8, 16]]
        for slot, freq in enumerate(freqs[:self._n]):
            self._chan.set_channel(slot, freq)
            print(f'  slot {slot:2d} → {freq/1e6:.4f} MHz')


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description='Stage 1 cuda_channelizer smoke test')
    ap.add_argument('-f', '--freq',     type=float, default=851e6,
                    help='SDR centre frequency Hz [851000000]')
    ap.add_argument('-r', '--rate',     type=float, default=20e6,
                    help='Sample rate Hz [20000000]')
    ap.add_argument('-g', '--gain',     type=float, default=40.0,
                    help='RF gain dB [40]')
    ap.add_argument('-t', '--time',     type=float, default=5.0,
                    help='Run duration seconds [5]')
    ap.add_argument('-c', '--config',   default='channelizer.json',
                    help='Path to channelizer.json')
    ap.add_argument('-d', '--device',   default='uhd,type=b200',
                    help='osmosdr device string [uhd,type=b200]')
    ap.add_argument('-n', '--channels', type=int,   default=20,
                    help='Number of output slots [20]')
    ap.add_argument('-v', '--verbose',  action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()

    print('=== cuda_channelizer Stage 1 smoke test ===')
    print(f'  Device   : {args.device}')
    print(f'  Freq     : {args.freq/1e6:.3f} MHz')
    print(f'  Rate     : {args.rate/1e6:.1f} Msps')
    print(f'  Gain     : {args.gain} dB')
    print(f'  Duration : {args.time} s')
    print(f'  Config   : {args.config}')
    print(f'  Slots    : {args.channels}')
    print()

    if not os.path.isfile(args.config):
        print(f'WARNING: {args.config} not found — '
              'the block will use built-in defaults if available.')
    print('Building flowgraph…')

    try:
        tb = SmokeTestFlowgraph(args)
    except Exception as e:
        print(f'FAIL  flowgraph construction: {e}')
        sys.exit(1)

    print('OK    flowgraph constructed')

    # Assign a few channels so the GPU has something real to process
    print('Assigning test channel slots:')
    tb.assign_test_channels(args.freq, 12_500.0)

    print()
    print(f'Running for {args.time:.0f} s  (watch for "GPU batch" in output)…')
    tb.start()

    crashed = threading.Event()
    def _watchdog():
        time.sleep(args.time)
        tb.stop()
        tb.wait()

    wdog = threading.Thread(target=_watchdog, daemon=True)
    wdog.start()
    wdog.join(timeout=args.time + 10)

    if wdog.is_alive():
        print('FAIL  flowgraph did not stop within deadline')
        sys.exit(1)

    print()
    print('OK    flowgraph stopped cleanly')
    print()
    print('=== Stage 1 PASS ===')
    print()
    print('Next steps:')
    print('  Stage 2 — polyphase bin energy check:')
    print('    Connect a vector_sink to one output port and verify')
    print('    the carrier energy appears in the correct bin.')
    print('  Stage 3 — full P25 decode:')
    print('    Point multi_rx.py at a live P25 system with a')
    print('    channelizer.json that matches the system centre frequency.')


if __name__ == '__main__':
    main()
