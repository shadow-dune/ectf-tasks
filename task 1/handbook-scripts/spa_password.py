#!/usr/bin/env python3
"""
spa_password.py — SPA on a leaky string comparison (password checker).

The book's example: a program compares an entered password to a stored
password byte-by-byte with early exit. Traces that share a longer correct
prefix take longer (more power) than traces with a wrong first byte.

Approach
--------
Given one trace per candidate character at the CURRENT position:
  - Take a baseline (default: mean of all traces, or the trace at --baseline)
  - Compute SAD of each trace against the baseline
  - The trace with the LARGEST SAD is the correct character
    (because a correct prefix causes a longer / more divergent trace)

Usage:
    # One CSV per position (36 traces: a-z, 0-9)
    python spa_password.py pos0.csv
    python spa_password.py pos0.csv --baseline-row 35    # explicit baseline

    # Or all positions in one CSV with a 'position' column
    python spa_password.py all.csv --position-col position

CSV format:
    Header: <guess_col>, s0, s1, ..., sN
    Each row: guess (str), trace samples
    guess_col auto-detected from ['guess','char','plaintext','pt']
"""
import argparse, sys, csv
from pathlib import Path
import numpy as np
import torch
from common import DEVICE

GUESS_COL_CANDIDATES = ['guess', 'char', 'plaintext', 'pt', 'input']


def _find_col(cols, candidates):
    lower = [c.lower() for c in cols]
    for cand in candidates:
        if cand in lower:
            return cols[lower.index(cand)]
    return None


def load_spa_csv(path, guess_col=None, pos_col=None, max_rows=None):
    """Returns (traces (N,S) float32, guesses list[str], positions list[int]|None)."""
    import pandas as pd
    header = pd.read_csv(path, nrows=0).columns.tolist()

    if guess_col is None:
        guess_col = _find_col(header, GUESS_COL_CANDIDATES)
    if guess_col is None:
        raise ValueError(f"No guess column found. Expected one of "
                         f"{GUESS_COL_CANDIDATES}.")

    trace_cols = [c for c in header if c not in (guess_col, pos_col)]
    dtypes = {c: np.float32 for c in trace_cols}
    df = pd.read_csv(path, dtype=dtypes, nrows=max_rows)

    guesses = df[guess_col].astype(str).tolist()
    positions = df[pos_col].astype(int).tolist() if pos_col else None
    traces = torch.from_numpy(df[trace_cols].values.astype(np.float32))

    return traces, guesses, positions


def sad_matrix(traces):
    """Compute pairwise SAD distance matrix. (N,N)"""
    N = traces.shape[0]
    d = torch.zeros(N, N, device=traces.device)
    for i in range(N):
        d[i] = (traces - traces[i].unsqueeze(0)).abs().sum(dim=1)
    return d


def recover_char(traces, guesses, baseline_idx=None, sample_window=None,
                 verbose=True):
    """
    For one position: find the guess with max SAD vs the baseline.

    traces:        (N, S) torch tensor
    guesses:       list[str] length N
    baseline_idx:  index of the "known wrong" baseline (None → use mean trace)
    sample_window: (start, end) to restrict analysis, or None for full trace

    Returns: (best_guess: str, sad_scores: np.ndarray)
    """
    t = traces.to(DEVICE)

    if sample_window:
        lo, hi = sample_window
        t = t[:, lo:hi]

    if baseline_idx is None:
        baseline = t.mean(dim=0, keepdim=True)   # mean trace as baseline
        label = "mean"
    else:
        baseline = t[baseline_idx:baseline_idx + 1]
        label = f"row {baseline_idx} ({guesses[baseline_idx]!r})"

    # SAD of each trace vs baseline
    sad = (t - baseline).abs().sum(dim=1).cpu().numpy()

    # Highest SAD = most divergent = correct guess
    order = np.argsort(sad)[::-1]
    best = guesses[order[0]]

    if verbose:
        print(f"  Baseline: {label}")
        print(f"  Top 5 by SAD:")
        for rank, idx in enumerate(order[:5]):
            print(f"    {rank+1}. {guesses[idx]!r:<20} SAD={sad[idx]:.4f}")

    return best, sad


def main():
    ap = argparse.ArgumentParser(description="SPA on leaky password comparison")
    ap.add_argument('csv')
    ap.add_argument('--guess-col', default=None)
    ap.add_argument('--position-col', default=None,
                    help="Column with position index (for multi-position CSV)")
    ap.add_argument('--baseline-row', type=int, default=None,
                    help="Row index to use as baseline (default: mean)")
    ap.add_argument('--window', type=int, nargs=2, default=None,
                    metavar=('LO', 'HI'), help="Sample window to compare (default: whole trace)")
    ap.add_argument('--max-rows', type=int, default=None)
    args = ap.parse_args()

    traces, guesses, positions = load_spa_csv(args.csv, args.guess_col,
                                              args.position_col, args.max_rows)
    print(f"[*] Loaded {len(guesses)} traces, shape={tuple(traces.shape)}")

    if positions is None:
        # Single position: attack directly
        best, sad = recover_char(traces, guesses, args.baseline_row, args.window)
        print(f"\n[+] Best guess: {best!r}")
    else:
        # Multi-position CSV: group, attack each
        unique_pos = sorted(set(positions))
        full_guess = ""
        for pos in unique_pos:
            mask = np.array([p == pos for p in positions])
            idxs = np.where(mask)[0]
            sub_traces = traces[idxs]
            sub_guesses = [guesses[i] for i in idxs]

            print(f"\n[*] Position {pos}:")
            best, sad = recover_char(sub_traces, sub_guesses,
                                     args.baseline_row, args.window)
            full_guess += best
            print(f"  → char {pos} = {best!r}")
        print(f"\n[+] Full guess: {full_guess!r}")


if __name__ == '__main__':
    main()