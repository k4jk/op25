#!/usr/bin/env python3
# Milestone 5 — UDP control protocol round-trip test
#
# Starts a minimal cuda_channelizer (via gnuradio) on localhost, then sends
# each control command and verifies the JSON responses.
#
# Usage (from the apps/ directory after installing the library):
#   python3 test_m5_control.py [--config channelizer.json] [--port 23457]
#
# The test can also be run in 'offline' mode against a standalone UDP echo
# server for CI pipelines that have no GPU:
#   python3 test_m5_control.py --offline

import argparse
import json
import socket
import sys
import time
import threading

# ---------------------------------------------------------------------------
# Lightweight UDP client helper
# ---------------------------------------------------------------------------

class CtrlClient:
    def __init__(self, host='127.0.0.1', port=23457, timeout=2.0):
        self.addr    = (host, port)
        self.timeout = timeout
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def send(self, cmd_dict):
        payload = json.dumps(cmd_dict).encode()
        self.sock.sendto(payload, self.addr)
        try:
            data, _ = self.sock.recvfrom(65536)
            return json.loads(data.decode())
        except socket.timeout:
            return None

    def close(self):
        self.sock.close()


# ---------------------------------------------------------------------------
# Standalone offline echo server (for CI without a GPU)
# ---------------------------------------------------------------------------

class OfflineServer:
    """Minimal echo server that mimics cuda_channelizer's control protocol."""

    def __init__(self, port=23457):
        self.port    = port
        self._sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('127.0.0.1', port))
        self._sock.settimeout(0.2)
        self._slots  = [-1] * 20          # slot → bin, -1 = inactive
        self._alloc  = [False] * 20        # slot allocated by alloc_slot?
        self._M      = 64
        self._center = 851_000_000.0
        self._bw     = 12_500.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._run    = True
        self._thread.start()

    def _freq_to_bin(self, freq_hz):
        off = freq_hz - self._center
        return int(round(off / self._bw)) % self._M

    def _handle(self, j):
        cmd = j.get('cmd', '')
        r   = {'cmd': cmd}
        if cmd == 'add_channel':
            s = j.get('slot', -1); f = j.get('freq_hz', 0.0)
            if 0 <= s < 20:
                self._slots[s] = self._freq_to_bin(f)
            r.update({'slot': s, 'ok': True})
        elif cmd == 'remove_channel':
            s = j.get('slot', -1)
            if 0 <= s < 20:
                self._slots[s] = -1
            r.update({'slot': s, 'ok': True})
        elif cmd == 'alloc_channel':
            f = j.get('freq_hz', 0.0)
            slot = next((i for i, a in enumerate(self._alloc) if not a), -1)
            if slot >= 0:
                self._alloc[slot] = True
                self._slots[slot] = self._freq_to_bin(f)
            r.update({'slot': slot, 'ok': slot >= 0})
        elif cmd == 'free_channel':
            s = j.get('slot', -1)
            if 0 <= s < 20:
                self._alloc[s] = False
                self._slots[s] = -1
            r.update({'slot': s, 'ok': True})
        elif cmd == 'status':
            entries = [{'slot': i, 'active': self._slots[i] >= 0,
                        'allocated': self._alloc[i], 'bin': self._slots[i],
                        'freq_hz': self._center + self._slots[i] * self._bw
                                   if self._slots[i] >= 0 else 0.0}
                       for i in range(20)]
            r.update({'slots': entries, 'M': self._M, 'ok': True})
        else:
            r.update({'ok': False, 'error': 'unknown'})
        return r

    def _loop(self):
        while self._run:
            try:
                data, addr = self._sock.recvfrom(65536)
                reply = self._handle(json.loads(data.decode()))
                self._sock.sendto(json.dumps(reply).encode(), addr)
            except socket.timeout:
                pass

    def stop(self):
        self._run = False
        self._thread.join(timeout=1.0)
        self._sock.close()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_tests(client):
    SDR_CENTER = 851_000_000.0
    CH_BW      = 12_500.0
    FREQ_1     = SDR_CENTER + 11 * CH_BW   # 851137500 Hz  (bin 11)
    FREQ_2     = SDR_CENTER + 25 * CH_BW   # bin 25

    results = []

    def check(name, resp, **expect):
        ok = resp is not None and all(
            resp.get(k) == v for k, v in expect.items()
        )
        status = 'PASS' if ok else 'FAIL'
        results.append(ok)
        detail = json.dumps(resp) if resp else '<timeout>'
        print(f'  {status}  [{name}]  {detail}')
        return ok

    # 1. add_channel: explicit slot assignment
    r = client.send({'cmd': 'add_channel', 'slot': 3, 'freq_hz': FREQ_1})
    check('add_channel slot=3', r, cmd='add_channel', slot=3, ok=True)

    # 2. status: verify slot 3 is active
    r = client.send({'cmd': 'status'})
    slot3_active = r and any(
        s['slot'] == 3 and s['active'] for s in r.get('slots', []))
    results.append(slot3_active)
    print(f"  {'PASS' if slot3_active else 'FAIL'}  [status: slot 3 active]  "
          f"{json.dumps(r) if r else '<timeout>'}")

    # 3. alloc_channel: automatic slot assignment
    r = client.send({'cmd': 'alloc_channel', 'freq_hz': FREQ_2})
    alloc_ok = r is not None and r.get('ok') and r.get('slot', -1) >= 0
    alloc_slot = r['slot'] if alloc_ok else -1
    results.append(alloc_ok)
    print(f"  {'PASS' if alloc_ok else 'FAIL'}  [alloc_channel → slot={alloc_slot}]  "
          f"{json.dumps(r) if r else '<timeout>'}")

    # 4. status: both slots active
    r = client.send({'cmd': 'status'})
    n_active = sum(1 for s in r.get('slots', []) if s['active']) if r else 0
    both_ok = n_active >= 2
    results.append(both_ok)
    print(f"  {'PASS' if both_ok else 'FAIL'}  [status: {n_active} active slots (expect ≥2)]")

    # 5. remove_channel: explicit deactivate
    r = client.send({'cmd': 'remove_channel', 'slot': 3})
    check('remove_channel slot=3', r, cmd='remove_channel', slot=3, ok=True)

    # 6. free_channel: return alloc'd slot to pool
    if alloc_slot >= 0:
        r = client.send({'cmd': 'free_channel', 'slot': alloc_slot})
        check(f'free_channel slot={alloc_slot}', r,
              cmd='free_channel', slot=alloc_slot, ok=True)

    # 7. status: all slots inactive again
    r = client.send({'cmd': 'status'})
    n_active = sum(1 for s in r.get('slots', []) if s['active']) if r else 99
    clear_ok = n_active == 0
    results.append(clear_ok)
    print(f"  {'PASS' if clear_ok else 'FAIL'}  [status: {n_active} active slots (expect 0)]")

    # 8. alloc_channel exhaustion: fill all slots, verify -1 on overflow
    for _ in range(20):
        client.send({'cmd': 'alloc_channel', 'freq_hz': FREQ_1})
    r = client.send({'cmd': 'alloc_channel', 'freq_hz': FREQ_1})
    full_ok = r is not None and not r.get('ok') and r.get('slot') == -1
    results.append(full_ok)
    print(f"  {'PASS' if full_ok else 'FAIL'}  [alloc overflow → slot=-1]  "
          f"{json.dumps(r) if r else '<timeout>'}")

    # 9. unknown command → ok=False
    r = client.send({'cmd': 'bogus_cmd'})
    check('unknown command', r, ok=False)

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Milestone 5 control protocol test')
    ap.add_argument('--host',    default='127.0.0.1')
    ap.add_argument('--port',    type=int, default=23457)
    ap.add_argument('--offline', action='store_true',
                    help='Run against a built-in offline echo server (no GPU needed)')
    args = ap.parse_args()

    server = None
    if args.offline:
        print('Starting offline echo server on port %d …' % args.port)
        server = OfflineServer(args.port)
        time.sleep(0.1)  # let thread bind
    else:
        print('Connecting to cuda_channelizer on %s:%d …' % (args.host, args.port))
        print('(ensure a GR flowgraph with cuda_channelizer is running)')

    client = CtrlClient(args.host, args.port)
    print()
    print('=== Milestone 5 UDP control protocol test ===')
    print()

    results = run_tests(client)
    client.close()
    if server:
        server.stop()

    passed = sum(results)
    total  = len(results)
    print()
    print(f'Results: {passed}/{total} PASS')
    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
