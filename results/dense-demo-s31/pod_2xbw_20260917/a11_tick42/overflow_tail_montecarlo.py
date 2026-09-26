#!/usr/bin/env python3
"""Monte Carlo check of the formula used in boot_overflow_estimate.py, in the region where events are frequent enough to
count: for a uniform-ternary secret s at N = 2^17 and uniformly random c0, c1 mod q0, the overflow vector is
I_j = round(c0_j/q0 + sum_i s_i c1_{j-i}/q0) (negacyclic). Predicted: sigma_I = sqrt((h+1)/12) and
P(max_j |I_j| > B) = 1 - (1 - 2Q((B+0.5)/sigma_I))^N. No key material: s and c are synthetic, fresh per trial.
usage: overflow_tail_montecarlo.py [trials=2000] [logN=17]"""
import sys, math, numpy as np
trials = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; logN = int(sys.argv[2]) if len(sys.argv) > 2 else 17
N = 1 << logN; rng = np.random.default_rng(20260918)
k = np.arange(N); tw = np.exp(1j * np.pi * k / N)             # negacyclic twist
def Q(x): return 0.5 * math.erfc(x / math.sqrt(2))
maxI = np.empty(trials); sig = np.empty(trials); hs = np.empty(trials)
for t in range(trials):
    s = rng.integers(-1, 2, size=N).astype(np.float64)          # uniform ternary
    c1 = rng.random(N) - 0.5; c0 = rng.random(N) - 0.5           # residues / q0, uniform on (-1/2, 1/2)
    x = np.real(np.fft.ifft(np.fft.fft(s * tw) * np.fft.fft(c1 * tw)) / tw) + c0
    I = np.rint(x); maxI[t] = np.max(np.abs(I)); sig[t] = I.std(); hs[t] = np.count_nonzero(s)
h = hs.mean(); sp = math.sqrt((h + 1) / 12)
print(f"N = 2^{logN}, trials {trials}: mean h {h:.0f} (2N/3 = {2 * N / 3:.0f}); sigma_I measured {sig.mean():.3f} (predicted sqrt((h+1)/12) = {sp:.3f})")
print(f"max|I| per ciphertext: min {maxI.min():.0f}, median {np.median(maxI):.0f}, max {maxI.max():.0f}")
print("B | observed P(max|I| > B) | predicted 1-(1-2Q((B+.5)/sigma))^N | observed count")
for B in (380, 400, 420, 440, 460, 480, 500):
    obs = int((maxI > B).sum()); pred = -math.expm1(N * math.log1p(-2 * Q((B + 0.5) / sp)))
    print(f"{B} | {obs / trials:.4f} | {pred:.4f} | {obs}")
