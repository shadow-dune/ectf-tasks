"""
cpa_aes128.py — Correlation Power Analysis on AES-128 round 1.

Usage:
    python cpa_aes128.py traces.csv
    python cpa_aes128.py traces.csv --n-pt-bytes 16 --no-cache
    python cpa_aes128.py --demo

Input: CSV with plaintext bytes and trace samples (see common.py loader).
Output: recovered 16-byte key, printed + saved to key_recovered.bin
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from common import (DEVICE, SBOX, HW_TABLE, pearson_corr, load_csv,
                    synth_aes128_traces)

SBOX_T = torch.tensor(list(SBOX), dtype=torch.long, device=DEVICE)
KEYS_256 = torch.arange(256, device=DEVICE)


def cpa_aes128_round1(traces, plaintexts, target_byte=None, verbose=True):
    """
    Attack one or all 16 AES-128 first-round bytes.

    traces:     (D, S)  float32
    plaintexts: (D, 16) uint8 (or (D,) uint8 for single-byte attack)
    target_byte: int or None (None = all bytes)

    Returns dict: {byte: {'key': int, 'peak': float, 'top5': [(k, r), ...]}}
    """
    traces = traces.to(DEVICE)
    if plaintexts.ndim == 1:
        plaintexts = plaintexts.reshape(-1, 1)

    D = traces.shape[0]
    n_bytes = 16 if target_byte is None else 1
    byte_range = range(16) if target_byte is None else [target_byte]

    results = {}
    for b in byte_range:
        t0 = time.time()
        pt_b = torch.from_numpy(plaintexts[:, b]).long().to(DEVICE)

        # Build hypothetical leakage: (256, D) = HW(SBOX[pt ^ k])
        inter = SBOX_T[pt_b.unsqueeze(0) ^ KEYS_256.unsqueeze(1)]
        L = HW_TABLE[inter]

        # Correlation: (256, S)
        corr = pearson_corr(traces, L)
        peak = corr.abs().max(dim=1).values          # (256,)

        best_k = int(peak.argmax())
        top5 = peak.topk(5)
        results[b] = {
            'key': best_k,
            'peak': float(peak[best_k]),
            'top5': [(int(k), float(v)) for k, v in zip(top5.indices, top5.values)],
        }

        if verbose:
            dt = time.time() - t0
            margin = results[b]['top5'][0][1] - results[b]['top5'][1][1]
            print(f"  byte {b:2d}: K=0x{best_k:02x}  |r|={results[b]['peak']:.4f}  "
                  f"margin={margin:+.4f}  ({dt*1000:.0f}ms)")

    return results


def main():
    ap = argparse.ArgumentParser(description="CPA on AES-128 first round")
    ap.add_argument('csv', nargs='?', help="Trace CSV (omit if --demo)")
    ap.add_argument('--demo', action='store_true',
                    help="Generate synthetic traces from a random key")
    ap.add_argument('--n-pt-bytes', type=int, default=None,
                    help="Number of plaintext byte columns (auto-detect by default)")
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--out', default='key_recovered.bin',
                    help="Output file for recovered key")
    ap.add_argument('--byte', type=int, default=None,
                    help="Attack only this byte (0-15), else all 16")
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    if args.cpu:
        import common; common.DEVICE = 'cpu'
        globals()['DEVICE'] = 'cpu'

    # ---------------- Load or synthesize ----------------
    if args.demo:
        print("[*] Demo mode: generating synthetic AES-128 traces")
        traces, pt, true_key = synth_aes128_traces(n_traces=3000, n_samples=4000)
        print(f"    True key: {true_key.hex()}")
    else:
        if not args.csv:
            sys.exit("Provide a CSV or use --demo")
        traces, meta = load_csv(args.csv, n_pt_bytes=args.n_pt_bytes,
                                max_rows=args.max_rows, cache=not args.no_cache)
        if 'plaintext' not in meta:
            sys.exit("CSV has no plaintext column.")
        pt = meta['plaintext']
        true_key = None

    # ---------------- Attack ----------------
    print(f"[*] Device: {DEVICE}   traces: {tuple(traces.shape)}")
    print(f"[*] Running CPA on AES-128 round 1 ...")
    t0 = time.time()
    results = cpa_aes128_round1(traces, pt, target_byte=args.byte)
    dt = time.time() - t0

    key = bytes(results.get(b, {'key': 0})['key'] for b in range(16)
                if b in results or args.byte is None)
    if args.byte is not None:
        key = bytes([results[args.byte]['key']])

    print(f"\n[+] Recovered key: {key.hex()}")
    print(f"[+] Elapsed: {dt:.2f}s")

    if true_key is not None:
        ok = key == true_key[:len(key)]
        print(f"[+] Matches ground truth: {ok}")

    Path(args.out).write_bytes(key)
    print(f"[+] Saved to {args.out}")


if __name__ == '__main__':
    main()