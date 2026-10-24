import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------- config ----------------
CSV_PATH = 'Round10_traces.csv'
KF = bytes.fromhex("91dec068e3face4599ce38f420863a30")   # fake key
N_USE = None          # e.g. 30000 to speed things up, None = all traces

# SR_IDX[i] = state byte j feeding ciphertext byte i (j = SR^-1(i))
SR_IDX = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

# ---------------- AES helpers ----------------
def gmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = ((a << 1) ^ 0x1b) & 0xff if a & 0x80 else a << 1
        b >>= 1
    return r

def make_sbox():
    sbox = [0] * 256
    for x in range(256):
        inv = 0 if x == 0 else next(y for y in range(1, 256) if gmul(x, y) == 1)
        s = r = inv
        for _ in range(4):
            r = ((r << 1) | (r >> 7)) & 0xff
            s ^= r
        sbox[x] = s ^ 0x63
    return sbox

SBOX = make_sbox()
INV_SBOX = [0] * 256
for i, v in enumerate(SBOX):
    INV_SBOX[v] = i

def prev_round_key(k, r):
    """Given round key r (16 bytes, column-major), return round key r-1."""
    w = [k[4 * c:4 * c + 4] for c in range(4)]
    p = [None] * 4
    p[3] = [a ^ b for a, b in zip(w[3], w[2])]
    p[2] = [a ^ b for a, b in zip(w[2], w[1])]
    p[1] = [a ^ b for a, b in zip(w[1], w[0])]
    g = [SBOX[p[3][1]] ^ RCON[r - 1], SBOX[p[3][2]], SBOX[p[3][3]], SBOX[p[3][0]]]
    p[0] = [a ^ b for a, b in zip(w[0], g)]
    return p[0] + p[1] + p[2] + p[3]

def master_from_k10(k10):
    k = list(k10)
    for r in range(10, 0, -1):
        k = prev_round_key(k, r)
    return k

# ---------------- data ----------------
def load_traces(filename):
    cts, trs = [], []
    with open(filename, 'r') as f:
        for row in csv.reader(f):
            if not row or not row[0]:
                continue
            cts.append([int(row[0][i:i + 2], 16) for i in range(0, 32, 2)])
            trs.append([float(x) for x in row[1:]])
    return np.array(cts), np.array(trs)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ct_np, tr_np = load_traces(CSV_PATH)
if N_USE:
    ct_np, tr_np = ct_np[:N_USE], tr_np[:N_USE]
N = ct_np.shape[0]
print(f"{N} traces, device = {dev.type}")

ct = torch.tensor(ct_np, dtype=torch.long, device=dev)         # (N,16)
tr = torch.tensor(tr_np, dtype=torch.float32, device=dev)      # (N,S)
kf = torch.tensor(list(KF), dtype=torch.long, device=dev)
inv_sbox_t = torch.tensor(INV_SBOX, dtype=torch.long, device=dev)
hw_t = torch.tensor([bin(n).count('1') for n in range(256)], dtype=torch.float32, device=dev)
sr_idx = torch.tensor(SR_IDX, dtype=torch.long, device=dev)
M9, M11, M13, M14 = [torch.tensor([gmul(x, c) for x in range(256)], dtype=torch.long, device=dev)
                     for c in (9, 11, 13, 14)]

def inv_mix_columns(x):                       # x: (N,16), column-major state
    s = x.reshape(-1, 4, 4)
    a0, a1, a2, a3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    r0 = M14[a0] ^ M11[a1] ^ M13[a2] ^ M9[a3]
    r1 = M9[a0] ^ M14[a1] ^ M11[a2] ^ M13[a3]
    r2 = M13[a0] ^ M9[a1] ^ M14[a2] ^ M11[a3]
    r3 = M11[a0] ^ M13[a1] ^ M9[a2] ^ M14[a3]
    return torch.stack([r0, r1, r2, r3], dim=-1).reshape(-1, 16)

# ---------------- Step 1: S9' and the missing value e_i ----------------
# S9'[:, j] = InvSbox(CT[i] ^ Kf[i]) with j = SR^-1(i)
S9p = torch.empty_like(ct)
S9p[:, sr_idx] = inv_sbox_t[ct ^ kf]

e = []                                         # e[i], indexed by ciphertext position i
for i in range(16):
    counts = torch.bincount(S9p[:, SR_IDX[i]], minlength=256)
    missing = (counts == 0).nonzero().flatten().tolist()
    assert len(missing) == 1, f"byte {i}: missing values = {missing}"
    e.append(missing[0])
print("missing values e_i:", [hex(v) for v in e])      # e_0 should be 0xda

# ---------------- Step 2: CPA over the 256 guesses of z = Sbox(X) ----------------
leak = torch.empty((N, 256), dtype=torch.float32, device=dev)
for z in range(256):
    kr = [KF[i] ^ SBOX[e[i]] ^ z for i in range(16)]          # Kr[i] = Kf[i] ^ Sbox(e_i) ^ z
    k9 = prev_round_key(kr, 10)
    kr_t = torch.tensor(kr, dtype=torch.long, device=dev)
    k9_t = torch.tensor(k9, dtype=torch.long, device=dev)

    S9 = torch.empty_like(ct)
    S9[:, sr_idx] = inv_sbox_t[ct ^ kr_t]                     # S9 = Sbox^-1(SR^-1(CT ^ Kr))

    t = inv_mix_columns(S9 ^ k9_t)                            # MC^-1(S9 ^ K9)
    u = torch.empty_like(t)
    u[:, sr_idx] = t                                          # SR^-1
    S8 = inv_sbox_t[u]                                        # Sbox^-1

    leak[:, z] = hw_t[S8 ^ S9p].sum(dim=1)                    # HD of register S8 -> S9'

# Pearson correlation of all 256 hypotheses against all samples
tr_c = tr - tr.mean(dim=0, keepdim=True)
tr_ss = (tr_c ** 2).sum(dim=0, keepdim=True)                  # (1,S)
l_c = leak - leak.mean(dim=0, keepdim=True)
l_ss = (l_c ** 2).sum(dim=0, keepdim=True).T                  # (256,1)
rho = (l_c.T @ tr_c) / torch.sqrt(l_ss @ tr_ss).clamp(min=1e-8)   # (256,S)
rho = rho.cpu().numpy()

peak = np.abs(rho).max(axis=1)
order = np.argsort(peak)[::-1]
z_best = int(order[0])
t_best = int(np.abs(rho[z_best]).argmax())
print(f"best z = {hex(z_best)}  rho = {rho[z_best, t_best]:.3f} at sample {t_best}")
print("runner-up:", hex(int(order[1])), f"{peak[order[1]]:.3f}")

X = INV_SBOX[z_best]
print(f"faulty LUT input X = Sbox^-1({hex(z_best)}) = {hex(X)}")

# ---------------- Step 3: real key and master key ----------------
kr = [KF[i] ^ SBOX[e[i]] ^ z_best for i in range(16)]
print("K10    :", bytes(kr).hex())
print("master :", bytes(master_from_k10(kr)).hex())

# ---------------- plot ----------------
plt.figure(figsize=(10, 4))
for z in range(256):
    if z != z_best:
        plt.plot(rho[z], color='grey', alpha=0.3, lw=0.5)
plt.plot(rho[z_best], color='red', lw=1.5, label=f'best z = {hex(z_best)}')
plt.xlabel('sample'); plt.ylabel('correlation'); plt.legend(); plt.grid(alpha=0.3)
plt.title('CPA of S8 -> S9\'')
plt.savefig('cpa.png', dpi=200)
plt.show()
