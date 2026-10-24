"""
dpa.py — Generic DPA (difference-of-means).

Works on any single-S-box transformation of the form
        output = SBOX[input ^ key]
by attacking one bit of `output` at a time.

Typical use cases:
  * AES-128 round 1 (SBOX, plaintext)
  * AES-256 round 14 decryption (INV_SBOX, ciphertext)
  * Your own toy S-box (e.g. XOR + lookup)

Usage:
    python dpa.py traces.csv --bit 0 --inv-sbox    # AES-256 round 14
    python dpa.py traces.csv --bit 0                # AES-128 round 1
    python dpa.py --demo

Output: recovered key (hex) for each byte.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from common import (DEVICE, SBOX, INV_SBOX, load_csv, synth_aes128_traces)

KEYS_256 = torch.arange(256, device=DEVICE)


def dpa_attack(traces, inputs, sbox, bit=0, verbose=True):
    """
    traces: (D, S) float32
    inputs: (D, 16) uint8
    sbox:   bytes(256) — either SBOX or INV_SBOX
    bit:    which bit of sbox output to split on (0..7)

    Returns: key bytes (16,), peaks (16,)
    """
    traces = traces.to(DEVICE)
    sbox_t = torch.tensor(list(sbox), dtype=torch.long, device=DEVICE)

    keys_out = np.zeros(16, dtype=np.uint8)
    peaks = np.zeros(16, dtype=np.float32)

    for b in range(16):
        t0 = time.time()
        inp = torch.from_numpy(inputs[:, b]).long().to(DEVICE)

        # Hypothetical bit for each of 256 key guesses: (256, D)
        inter = sbox_t[inp.unsqueeze(0) ^ KEYS_256.unsqueeze(1)]
        bits = ((inter >> bit) & 1).float()

        n1 = bits.sum(dim=1, keepdim=True).clamp(min=1)
        n0 = (traces.shape[0] - n1).clamp(min=1)

        sum1 = bits @ traces
        sum0 = (1 - bits) @ traces
        dom = sum1 / n1 - sum0 / n0                   # (256, S)

        pk = dom.abs().max(dim=1).values              # (256,)
        best = int(pk.argmax())
        keys_out[b] = best
        peaks[b] = float(pk[best])

        if verbose:
            print(f"  byte {b:2d}: K=0x{best:02x}  |DoM|={peaks[b]:.4f}  "
                  f"({(time.time()-t0)*1000:.0f}ms)")

    return bytes(keys_out), peaks


def main():
    ap = argparse.ArgumentParser(description="Generic DPA")
    ap.add_argument('csv', nargs='?')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--bit', type=int, default=0, help="Bit of S-box output to split on")
    ap.add_argument('--inv-sbox', action='store_true',
                    help="Use INV_SBOX (AES-256 decryption)")
    ap.add_argument('--n-pt-bytes', type=int, default=None)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--out', default='dpa_key.bin')
    args = ap.parse_args()

    # ---- Load ----
    if args.demo:
        print("[*] Demo mode: synthetic AES-128 traces")
        traces, pt, true_key = synth_aes128_traces(n_traces=3000, n_samples=4000)
        print(f"    True key: {true_key.hex()}")
    else:
        if not args.csv: sys.exit("Provide CSV or --demo")
        traces, meta = load_csv(args.csv, n_pt_bytes=args.n_pt_bytes,
                                cache=not args.no_cache)
        # Explicit None check — never use `or` on NumPy arrays
        pt = meta.get('plaintext')
        if pt is None:
            pt = meta.get('ciphertext')
        if pt is None:
            sys.exit("No input column ('plaintext' or 'ciphertext') found in CSV.")
        true_key = None

    sbox = INV_SBOX if args.inv_sbox else SBOX
    print(f"[*] Device: {DEVICE}   traces: {tuple(traces.shape)}")
    print(f"[*] DPA: sbox={'INV' if args.inv_sbox else 'FWD'}  bit={args.bit}")

    key, peaks = dpa_attack(traces, pt, sbox, bit=args.bit)
    print(f"\n[+] Recovered key: {key.hex()}")
    if true_key is not None:
        print(f"[+] Matches ground truth: {key == true_key}")

    Path(args.out).write_bytes(key)
    print(f"[+] Saved to {args.out}")


if __name__ == '__main__':
    main()