"""
common.py — AES tables, CSV loaders, vectorized Pearson correlation,
GF(2^8) arithmetic, and demo trace generation.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import binascii

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SBOX = bytes([
 0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
 0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
 0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
 0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
 0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
 0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
 0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
 0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
 0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
 0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
 0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
 0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
 0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
 0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
 0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
 0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
])

INV_SBOX = bytes(256)
_inv = [0]*256
for _i, _v in enumerate(SBOX):
    _inv[_v] = _i
INV_SBOX = bytes(_inv)

RCON = [None,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36]

# Permutations for InvShiftRows / ShiftRows on 16-byte column-major state
INV_SHIFT_ROWS = [0,13,10,7, 4,1,14,11, 8,5,2,15, 12,9,6,3]
SHIFT_ROWS     = [0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11]

# =============================================================
# Hamming weight lookup
# =============================================================
def build_hw(device=DEVICE):
    return torch.tensor([bin(i).count('1') for i in range(256)],
                        dtype=torch.float32, device=device)

HW_TABLE = build_hw()

# =============================================================
# GF(2^8) arithmetic
# =============================================================
def _gf_mult(a, b):
    p = 0
    for _ in range(8):
        if b & 1: p ^= a
        hi = a & 0x80
        a = ((a << 1) & 0xFF) ^ (0x1b if hi else 0)
        b >>= 1
    return p

GF_MUL = np.zeros((256, 256), dtype=np.uint8)
for _a in range(256):
    for _b in range(256):
        GF_MUL[_a, _b] = _gf_mult(_a, _b)

INV_MIXCOLS = np.array([[14,11,13,9],[9,14,11,13],[13,9,14,11],[11,13,9,14]], dtype=np.uint8)
MIXCOLS     = np.array([[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]], dtype=np.uint8)

def _mix_columns_impl(state, matrix):
    """state: (..., 16) uint8. Returns same shape."""
    s = state.reshape(*state.shape[:-1], 4, 4)  # (..., col, row)
    out = np.zeros_like(s)
    for i in range(4):
        for j in range(4):
            acc = np.zeros(s.shape[:-2], dtype=np.uint8)
            for k in range(4):
                acc = acc ^ GF_MUL[matrix[i, k], s[..., j, k]]
            out[..., j, i] = acc
    return out.reshape(*state.shape)

def inv_mix_columns(state):   return _mix_columns_impl(state, INV_MIXCOLS)
def mix_columns(state):       return _mix_columns_impl(state, MIXCOLS)

def inv_shift_rows(state):    return state[..., INV_SHIFT_ROWS]
def shift_rows(state):        return state[..., SHIFT_ROWS]

def sub_bytes(state):         return np.array(SBOX, dtype=np.uint8)[state]
def inv_sub_bytes(state):     return np.array(INV_SBOX, dtype=np.uint8)[state]

# =============================================================
# AES-256 inverse key schedule
# =============================================================
def _sub_word(w):
    return bytes(SBOX[b] for b in w)

def _rot_word(w):
    return w[1:] + w[:1]

def _xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

def aes256_inverse_key_schedule(k14_bytes, k13_bytes):
    """
    Given two consecutive 16-byte round keys K14 (last round) and K13,
    recover the full 32-byte AES-256 key by walking the key schedule back.
    """
    W = [None]*60
    for i in range(4):
        W[56 + i] = k14_bytes[i*4:(i+1)*4]
    for i in range(4):
        W[52 + i] = k13_bytes[i*4:(i+1)*4]

    for i in range(51, -1, -1):
        if (i + 8) % 8 == 0:
            rcon = bytes([RCON[(i + 8) // 8], 0, 0, 0])
            t = _xor_bytes(_sub_word(_rot_word(W[i + 7])), rcon)
            W[i] = _xor_bytes(W[i + 8], t)
        elif (i + 8) % 8 == 4:
            t = _sub_word(W[i + 7])
            W[i] = _xor_bytes(W[i + 8], t)
        else:
            W[i] = _xor_bytes(W[i + 8], W[i + 7])

    return b''.join(W[:8])

# =============================================================
# CSV Loader
# =============================================================
PT_COL_CANDIDATES = ['plaintext', 'pt', 'input', 'p', 'msg']
CT_COL_CANDIDATES = ['ciphertext', 'ct', 'output', 'c']

def _find_col(cols, candidates):
    lower = [c.lower() for c in cols]
    for cand in candidates:
        if cand in lower:
            return cols[lower.index(cand)]
    return None

def _parse_hex_column(series, n_bytes):
    arr = np.zeros((len(series), n_bytes), dtype=np.uint8)
    for i, s in enumerate(series.astype(str)):
        s = s.strip().replace(' ', '').replace('0x', '')
        try:
            arr[i] = np.frombuffer(bytes.fromhex(s), dtype=np.uint8)
        except ValueError:
            # Fall back: treat as raw bytes string
            arr[i] = np.frombuffer(s.encode('latin-1')[:n_bytes], dtype=np.uint8)
    return arr

def load_csv(path, pt_col=None, ct_col=None, n_pt_bytes=None,
             max_rows=None, cache=True, verbose=True):
    """
    Load (traces, metadata) from CSV.
    Metadata dict contains 'plaintext' and/or 'ciphertext' arrays.
    """
    path = Path(path)
    cache_path = path.with_suffix('.npz')

    if cache and cache_path.exists() and \
       cache_path.stat().st_mtime > path.stat().st_mtime:
        if verbose: print(f"[+] Cache hit: {cache_path.name}")
        d = np.load(cache_path)
        meta = {k: d[k] for k in d.files if k != 'traces'}
        return torch.from_numpy(d['traces']), meta

    if verbose: print(f"[+] Loading {path.name} ...")
    header = pd.read_csv(path, nrows=0).columns.tolist()

    if pt_col is None: pt_col = _find_col(header, PT_COL_CANDIDATES)
    if ct_col is None: ct_col = _find_col(header, CT_COL_CANDIDATES)

    trace_cols = [c for c in header if c not in (pt_col, ct_col)]
    dtypes = {c: np.float32 for c in trace_cols}
    df = pd.read_csv(path, dtype=dtypes, nrows=max_rows)
    if verbose: print(f"[+] Loaded {len(df)} rows, {len(header)} columns")

    meta = {}
    if pt_col:
        raw = df[pt_col].astype(str).iloc[0]
        n_bytes = len(raw.replace(' ', '')) // 2
        meta['plaintext'] = _parse_hex_column(df[pt_col], n_bytes)
        if verbose: print(f"[+] plaintext: column={pt_col!r}, {n_bytes} bytes")

    if ct_col:
        raw = df[ct_col].astype(str).iloc[0]
        n_bytes = len(raw.replace(' ', '')) // 2
        meta['ciphertext'] = _parse_hex_column(df[ct_col], n_bytes)
        if verbose: print(f"[+] ciphertext: column={ct_col!r}, {n_bytes} bytes")

    if not meta:
        # Fallback: first N integer columns in [0,255] are plaintext bytes
        if n_pt_bytes is None:
            n_pt_bytes = 0
            for c in trace_cols:
                vals = df[c].head(100)
                if vals.dtype == np.float32:
                    if (vals % 1 == 0).all() and vals.between(0, 255).all():
                        n_pt_bytes += 1
                    else:
                        break
                else:
                    break
            if n_pt_bytes == 0:
                raise ValueError(
                    "Cannot auto-detect plaintext columns. "
                    "Add a column named 'plaintext' or pass --n-pt-bytes.")
        meta['plaintext'] = df[trace_cols[:n_pt_bytes]].values.astype(np.uint8)
        trace_cols = trace_cols[n_pt_bytes:]
        if verbose: print(f"[+] positional plaintext: first {n_pt_bytes} columns")

    traces_np = df[trace_cols].values.astype(np.float32)
    traces = torch.from_numpy(traces_np)
    if verbose: print(f"[+] traces: {tuple(traces.shape)}  {traces.dtype}")

    if cache:
        np.savez_compressed(cache_path, traces=traces_np, **meta)
        if verbose: print(f"[+] Cached → {cache_path.name}")

    return traces, meta

# =============================================================
# Vectorized Pearson correlation
# =============================================================
def pearson_corr(traces, leakage):
    """
    traces:   (D, S) float32
    leakage:  (K, D) float32
    Returns:  (K, S) Pearson r matrix (one row per key guess)
    """
    T = traces - traces.mean(dim=0, keepdim=True)   # (D, S)
    L = leakage - leakage.mean(dim=1, keepdim=True) # (K, D)

    num = L @ T                                     # (K, S)
    Ln = torch.sqrt((L * L).sum(dim=1)).clamp(min=1e-10)  # (K,)
    Tn = torch.sqrt((T * T).sum(dim=0)).clamp(min=1e-10)  # (S,)
    den = Ln.unsqueeze(1) * Tn.unsqueeze(0)

    return num / den

# =============================================================
# Demo synthetic traces (verify scripts without hardware)
# =============================================================
def synth_leak_positions(n_samples, margin_frac=0.10):
    """
    Return the 16 sample positions where synth_aes128_traces() places
    its byte leaks, so demos and windows can reference them.
    """
    margin = max(20, int(n_samples * margin_frac))
    usable = n_samples - 2 * margin - 2          # room for [pos, pos+1]
    stride = max(2, usable // 16)
    positions = [margin + b * stride for b in range(16)]
    assert positions[-1] + 1 < n_samples, (
        f"n_samples={n_samples} too small for 16 leaks "
        f"(need ≥ {positions[-1] + 2})")
    return positions


def synth_aes128_traces(n_traces=2000, n_samples=4000, key=None,
                        noise=1.0, seed=42, leaky=True):
    """
    Synthetic AES-128 traces with Hamming-weight leakage on round 1.

    Leak positions auto-scale to fit n_samples (see synth_leak_positions).

    Parameters
    ----------
    key     : bytes(16) or None (random)
    leaky   : if False, return pure-noise traces (for TVLA "clean" tests)
    """
    rng = np.random.default_rng(seed)
    if key is None:
        key = rng.integers(0, 256, 16, dtype=np.uint8)
    pt = rng.integers(0, 256, (n_traces, 16), dtype=np.uint8)
    traces = rng.normal(0, noise, (n_traces, n_samples)).astype(np.float32)

    if not leaky:
        return torch.from_numpy(traces), pt, bytes(key)

    positions = synth_leak_positions(n_samples)
    sbox_arr = np.frombuffer(SBOX, dtype=np.uint8)
    for b in range(16):
        inter = sbox_arr[pt[:, b] ^ key[b]]
        hw = np.array([bin(x).count('1') for x in inter], dtype=np.float32)
        pos = positions[b]
        traces[:, pos] += hw * 3.0
        traces[:, pos + 1] += hw * 2.0

    return torch.from_numpy(traces), pt, bytes(key)

# ---- Forward AES-256 key schedule (needed by demo generator) ----
def _xor4(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

def aes256_forward_key_schedule(full_key_32):
    """Returns list of 15 round keys (bytes(16)), indices 0..14."""
    assert len(full_key_32) == 32
    W = [full_key_32[i*4:(i+1)*4] for i in range(8)]
    for i in range(8, 60):
        if i % 8 == 0:
            t = _xor_bytes(_sub_word(_rot_word(W[i-1])),
                           bytes([RCON[i // 8], 0, 0, 0]))
        elif i % 8 == 4:
            t = _sub_word(W[i-1])
        else:
            t = W[i-1]
        W.append(_xor_bytes(W[i-8], t))
    return [b''.join(W[r*4:(r+1)*4]) for r in range(15)]

def hex_series_to_array(series):
    """Convert pandas Series of hex strings → (N, n_bytes) uint8 numpy array."""
    h = series.values
    if len(h) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    s0 = str(h[0]).strip().replace(' ', '')
    n_bytes = len(s0) // 2
    out = np.empty((len(h), n_bytes), dtype=np.uint8)
    for i, row in enumerate(h):
        out[i] = np.frombuffer(binascii.unhexlify(str(row).strip()), dtype=np.uint8)
    return out