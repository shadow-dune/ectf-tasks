import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
CSV_PATH = 'Round10_traces.csv'

# Original AES Inverse S-Box
INV_SBOX = [
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
]

# Inverse ShiftRows mapping
INV_SR = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]

def load_traces(filename):
    ciphertexts, traces = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0]: continue
            ct = [int(row[0][i:i+2], 16) for i in range(0, 32, 2)]
            ciphertexts.append(ct)
            traces.append([float(x) for x in row[1:]])
    return np.array(ciphertexts), np.array(traces)

def main():
    print("Loading traces into CPU memory...")
    # Load traces into standard numpy arrays first
    ciphertexts_cpu, traces_cpu = load_traces(CSV_PATH)
    num_samples = traces_cpu.shape[1]
    
    # Auto-detect GPU via PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Accelerating CPA using device: {device.type.upper()}")

    # 1. Move data to GPU tensors
    # ciphertexts: shape (Num_Traces, 16)
    ct_tensor = torch.tensor(ciphertexts_cpu, dtype=torch.long, device=device)
    # traces: shape (Num_Traces, Num_Samples)
    tr_tensor = torch.tensor(traces_cpu, dtype=torch.float32, device=device)
    
    # 2. Move lookup tables to GPU tensors
    inv_sbox_tensor = torch.tensor(INV_SBOX, dtype=torch.long, device=device)
    hw_tensor = torch.tensor([bin(n).count('1') for n in range(256)], dtype=torch.float32, device=device)
    
    # 3. Pre-compute trace standard deviations for the Pearson denominator
    tr_mean = tr_tensor.mean(dim=0, keepdim=True)
    tr_centered = tr_tensor - tr_mean
    # Sum of squares for traces: shape (1, Num_Samples)
    tr_ss = torch.sum(tr_centered**2, dim=0, keepdim=True) 

    # Generate an array of all 256 keys to vectorize the guesses
    K = torch.arange(256, dtype=torch.long, device=device)

    recovered_key = []
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'GPU Accelerated CPA Correlation Traces ({device.type.upper()})', fontsize=16)

    for b in range(16):
        print(f"Processing byte {b:02d}/15...")
        
        # Isolate ciphertext bytes for the current position, shape (Num_Traces, 1)
        c_i = ct_tensor[:, b].unsqueeze(1)
        c_inv_sr_i = ct_tensor[:, INV_SR[b]].unsqueeze(1)
        
        # Calculate hypothetical leakage for ALL 256 keys simultaneously
        # Broadcasting creates a (Num_Traces, 256) matrix of SBox outputs
        sbox_out = inv_sbox_tensor[c_i ^ K] 
        # Calculate Hamming Distance, final leakage shape: (Num_Traces, 256)
        leakage = hw_tensor[sbox_out ^ c_inv_sr_i]
        
        # Vectorized Pearson Correlation
        l_mean = leakage.mean(dim=0, keepdim=True)
        l_centered = leakage - l_mean
        
        # Sum of squares for leakages: shape (256, 1)
        l_ss = torch.sum(l_centered**2, dim=0, keepdim=True).T
        
        # Matrix multiplication computes correlations for all traces against all 256 keys instantly
        # numerator shape: (256, Num_Samples)
        numerator = torch.matmul(l_centered.T, tr_centered)
        # denominator shape: (256, Num_Samples)
        denominator = torch.sqrt(torch.matmul(l_ss, tr_ss))
        
        # Avoid division by zero with torch.clamp
        correlations = torch.abs(numerator / torch.clamp(denominator, min=1e-8))
        
        # Move the computed matrix back to the CPU to plot
        correlations_cpu = correlations.cpu().numpy()
        
        # Find the peak correlation and the corresponding key guess
        max_corrs_per_key = np.max(correlations_cpu, axis=1)
        best_candidate = int(np.argmax(max_corrs_per_key))
        max_corr_val = max_corrs_per_key[best_candidate]
        
        recovered_key.append(best_candidate)
        print(f"  -> Best guess: {hex(best_candidate)} (Peak Corr: {max_corr_val:.4f})")
        
        # Plotting
        ax = axes[b // 4, b % 4]
        for k in range(256):
            if k == best_candidate:
                ax.plot(correlations_cpu[k], color='red', linewidth=1.5, zorder=10)
            else:
                ax.plot(correlations_cpu[k], color='grey', alpha=0.3, linewidth=0.5, zorder=1)
                
        ax.set_title(f"Byte {b} (Key: {hex(best_candidate)})")
        if b >= 12: ax.set_xlabel("Sample Index")
        if b % 4 == 0: ax.set_ylabel("Absolute Correlation")
        ax.grid(True, alpha=0.3)

    hex_key = "".join([f"{k:02x}" for k in recovered_key])
    print("\n" + "="*50)
    print(f"RECOVERED FAKE KEY: {hex_key}")
    print("="*50)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('correlation_plots.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

