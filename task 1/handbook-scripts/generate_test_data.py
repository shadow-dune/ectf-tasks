#!/usr/bin/env python3
"""
generate_test_data.py — Creates synthetic CSVs to test all 5 attack scripts.
Generates:
  1. test_aes128.csv    (for cpa_aes128.py, dpa.py, align_traces.py)
  2. test_aes256.csv    (for cpa_aes256.py)
  3. test_spa.csv       (for spa_password.py)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Import from your common.py
from common import (
    SBOX, INV_SBOX, synth_aes128_traces, synth_leak_positions,
    aes256_forward_key_schedule, inv_shift_rows, inv_mix_columns
)

def gen_aes128():
    print("[*] Generating test_aes128.csv (3000 traces, 1000 samples)...")
    traces, pt, key = synth_aes128_traces(n_traces=3000, n_samples=1000)
    
    # Save as CSV with hex plaintext
    df = pd.DataFrame(traces.numpy())
    df.insert(0, 'plaintext', [bytes(row).hex() for row in pt])
    df.to_csv("test_aes128.csv", index=False)
    print(f"    Saved! True key = {key.hex()}")

def gen_aes256():
    print("[*] Generating test_aes256.csv (2000 traces, 4000 samples)...")
    n_traces, n_samples = 2000, 4000
    rng = np.random.default_rng(42)
    
    # 1. Generate random key and derive round keys
    full_key = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    rks = aes256_forward_key_schedule(full_key)
    K13, K14 = np.frombuffer(rks[13], dtype=np.uint8), np.frombuffer(rks[14], dtype=np.uint8)
    K13_primed = inv_shift_rows(inv_mix_columns(K13))
    
    # 2. Random ciphertext
    ct = rng.integers(0, 256, (n_traces, 16), dtype=np.uint8)
    
    # 3. Noisy baseline traces
    traces = rng.normal(0, 1.0, (n_traces, n_samples)).astype(np.float32)
    sbox_inv = np.frombuffer(INV_SBOX, dtype=np.uint8)
    
    # 4. Add Round 14 leakage (samples 100..420)
    step1 = ct ^ K14[None, :]
    X14 = sbox_inv[inv_shift_rows(step1)]
    for b in range(16):
        hw = np.array([bin(x).count('1') for x in X14[:, b]], dtype=np.float32)
        traces[:, 100 + 20*b] += hw * 3.0
        
    # 5. Add Round 13 leakage (samples 2000..3500)
    Y = inv_shift_rows(inv_mix_columns(X14))
    X13 = sbox_inv[Y ^ K13_primed[None, :]]
    for b in range(16):
        hw = np.array([bin(x).count('1') for x in X13[:, b]], dtype=np.float32)
        traces[:, 2000 + 100*b] += hw * 3.0
        
    # 6. Save as CSV (ciphertext goes in 'plaintext' column for decryption attack)
    df = pd.DataFrame(traces)
    df.insert(0, 'plaintext', [bytes(row).hex() for row in ct])
    df.to_csv("test_aes256.csv", index=False)
    print(f"    Saved! True 256-bit key = {full_key.hex()}")

def gen_spa():
    print("[*] Generating test_spa.csv (36 guesses, 1000 samples)...")
    # 36 guesses (a-z, 0-9)
    guesses = list("abcdefghijklmnopqrstuvwxyz0123456789")
    n_samples = 1000
    traces = np.random.normal(0, 1.0, (len(guesses), n_samples)).astype(np.float32)
    
    # The correct character is 'h'. If correct, add a big spike 
    # (mimicking longer execution / more loops)
    correct_idx = guesses.index('h')
    traces[correct_idx, 500:550] += 5.0 
    traces[correct_idx, 700:750] += 5.0
    
    df = pd.DataFrame(traces)
    df.insert(0, 'guess', guesses)
    df.to_csv("test_spa.csv", index=False)
    print("    Saved! Correct character is 'h'")

if __name__ == '__main__':
    gen_aes128()
    gen_aes256()
    gen_spa()
    print("\n[+] All test CSVs generated successfully!")