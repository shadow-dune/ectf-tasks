"""
align_traces.py — Resynchronize misaligned power traces via SAD.

Inputs:
    - CSV of traces (same loader as common.py)
    - A reference window inside the traces (either auto-picked from a
      reference trace or user-specified --window start:end)

Outputs:
    - aligned_traces.npz with {traces, shifts} for downstream CPA/DPA

Usage:
    python align_traces.py traces.csv
    python align_traces.py traces.csv --window 1500:2000 --max-shift 100
    python align_traces.py traces.csv --reference-row 0
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from common import load_csv, DEVICE

def pick_reference_window(traces, window=None, ref_row=0):
    """
    Return (reference_window tensor, window_start, window_end).
    If window is None, auto-pick a high-variance region.
    """
    if window is not None:
        lo, hi = window
        return traces[ref_row, lo:hi].clone(), lo, hi

    # Auto-pick: rolling variance, choose 200-sample window with max var
    ref = traces[ref_row]
    win = 200
    rolled = ref.unfold(0, win, 1)              # (S-win+1, win)
    var = rolled.var(dim=1)
    start = int(var.argmax())
    return ref[start:start + win].clone(), start, start + win


def align_sad(traces, ref_window, max_shift=50, batch_size=512):
    """
    Align all traces to ref_window using SAD minimization.

    traces:     (D, S) float32
    ref_window: (L,) float32 — pattern to align against
    max_shift:  int, search ± this many samples
    Returns:    aligned (D, S) float32, shifts (D,) int64
    """
    traces = traces.to(DEVICE)
    ref_window = ref_window.to(DEVICE)
    D, S = traces.shape
    L = ref_window.shape[0]
    center = (S - L) // 2

    aligned = torch.empty_like(traces)
    shifts = torch.zeros(D, dtype=torch.long, device=DEVICE)

    lo = max(0, center - max_shift)
    hi = min(S - L + 1, center + max_shift + 1)

    t0 = time.time()
    for s in range(0, D, batch_size):
        e = min(s + batch_size, D)
        batch = traces[s:e]                      # (b, S)
        # Sliding windows: (b, S-L+1, L)
        win = batch.unfold(1, L, 1)
        # SAD against reference: (b, S-L+1)
        sad = (win - ref_window.unsqueeze(0).unsqueeze(0)).abs().sum(dim=2)
        # Restrict to search range
        sad_sub = sad[:, lo:hi]
        best_local = sad_sub.argmin(dim=1) + lo   # (b,)
        shifts[s:e] = best_local - center

        # Apply shift via gather (pad with edge values)
        idx = torch.arange(S, device=DEVICE).unsqueeze(0) - \
              shifts[s:e].unsqueeze(1)            # (b, S)
        idx = idx.clamp(0, S - 1)
        aligned[s:e] = torch.gather(batch, 1, idx)

        if s % (batch_size * 8) == 0:
            print(f"    {s}/{D} traces aligned ...")

    print(f"[+] Alignment took {time.time()-t0:.2f}s "
          f"(shift range [{int(shifts.min())}, {int(shifts.max())}])")
    return aligned, shifts


def main():
    ap = argparse.ArgumentParser(description="SAD trace alignment")
    ap.add_argument('csv')
    ap.add_argument('--window', type=str, default=None,
                    help="Reference window as 'start:end' (default: auto)")
    ap.add_argument('--reference-row', type=int, default=0)
    ap.add_argument('--max-shift', type=int, default=50,
                    help="Search ± this many samples for best shift")
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--out', default=None,
                    help="Output NPZ (default: <csv>_aligned.npz)")
    ap.add_argument('--n-pt-bytes', type=int, default=None)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    traces, meta = load_csv(args.csv, n_pt_bytes=args.n_pt_bytes,
                            cache=not args.no_cache)
    print(f"[*] Loaded {tuple(traces.shape)} traces")

    window = None
    if args.window:
        lo, hi = map(int, args.window.split(':'))
        window = (lo, hi)
    ref_window, lo, hi = pick_reference_window(traces, window, args.reference_row)
    print(f"[*] Reference window: samples {lo}:{hi} (length {hi-lo})")

    aligned, shifts = align_sad(traces, ref_window,
                                 max_shift=args.max_shift,
                                 batch_size=args.batch_size)

    out = args.out or str(Path(args.csv).with_suffix('')) + '_aligned.npz'
    save_dict = {'traces': aligned.cpu().numpy(),
                 'shifts': shifts.cpu().numpy()}
    save_dict.update(meta)
    np.savez_compressed(out, **save_dict)
    print(f"[+] Saved aligned traces to {out}")

    # Print summary
    uniq, counts = np.unique(shifts.cpu().numpy(), return_counts=True)
    print(f"[+] Most common shifts: "
          f"{dict(zip(uniq[np.argsort(counts)[::-1][:5]].tolist(), counts[np.argsort(counts)[::-1][:5]].tolist()))}")


if __name__ == '__main__':
    main()