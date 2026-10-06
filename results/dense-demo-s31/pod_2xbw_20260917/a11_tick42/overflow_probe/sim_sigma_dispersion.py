#!/usr/bin/env python3
"""Side check for PROBE_C_REPORT.md (pure numpy, no FHE library, no key material): how much should the per-ciphertext estimate of
sigma_I scatter from ciphertext to ciphertext?

The N overflows I_j of one ciphertext share the same c1, so they are weakly correlated: Cov(I_j, I_j') = (1/12) * autocorr_s(j'-j).
For jointly Gaussian I that inflates Var(sample variance) from 2 sigma^4 / N to 2 sigma^4 (1 + R2) / N, R2 = sum_{k != 0} rho_k^2 ~ 1.
This toy model (small ring, 30-bit modulus, exact integer arithmetic) measures the scatter for two kinds of ciphertext under ONE fixed
ternary secret:  A. c1 i.i.d. uniform (the textbook model of an evaluated ciphertext),
                 B. a fresh PUBLIC-KEY encryption c1 = a*u + e1, c0 = b*u + e0 with ONE fixed public key (a, b = -a*s + e), u ternary.
It prints the ratio Var(sigma_hat^2) / (2 sigma^4 / N): 1.0 = independent-sample scatter, 1 + R2 = the correlated-Gaussian prediction.
"""
import numpy as np, math, json, os
rng = np.random.default_rng(20260918)
N, q, M = 1024, (1 << 30) + 3, 6000
def negconv(a, b):
    f = np.convolve(a, b); r = f[:N].copy(); r[: N - 1] -= f[N:]; return r
def centered(x):
    x = np.mod(x, q); return np.where(x > q // 2, x - q, x)
s = rng.integers(-1, 2, size=N).astype(np.int64); h = int(np.count_nonzero(s)); sig2 = (h + 1) / 12.0
ac = negconv(s, s[::-1].copy())            # not the negacyclic autocorrelation in index order, so compute it directly:
ac = np.array([np.sum(s[: N - k] * s[k:]) - np.sum(s[N - k:] * s[: k]) if k else np.sum(s * s) for k in range(N)], dtype=np.float64)
R2 = float(np.sum((ac[1:] / ac[0]) ** 2))
a_pk = rng.integers(0, q, size=N).astype(np.int64); e_pk = np.rint(rng.normal(0, 3.19, size=N)).astype(np.int64)
b_pk = centered(e_pk - negconv(centered(a_pk), s))
out = {"N": N, "q": q, "ciphertexts": M, "hamming": h, "sigmaPred": math.sqrt(sig2), "R2": R2, "prediction_correlatedGaussian": 1 + R2}
for name in ("A_uniform_c1", "B_public_key_encryption"):
    v = np.empty(M); mx = np.empty(M)
    for t in range(M):
        if name.startswith("A"):
            c1 = rng.integers(-(q // 2), q // 2 + 1, size=N).astype(np.int64)
            c0 = centered(-negconv(c1, s))                      # message 0, noise 0: t = c0 + c1*s is an exact multiple of q
        else:
            u = rng.integers(-1, 2, size=N).astype(np.int64)
            e0 = np.rint(rng.normal(0, 3.19, size=N)).astype(np.int64); e1 = np.rint(rng.normal(0, 3.19, size=N)).astype(np.int64)
            c1 = centered(negconv(centered(a_pk), u) + e1); c0 = centered(negconv(b_pk, u) + e0)
        tt = c0 + negconv(c1, s)
        I = np.floor_divide(tt + q // 2, q)
        v[t] = I.var(); mx[t] = np.abs(I).max()
    ratio = float(v.var() / (2 * sig2 ** 2 / N))
    out[name] = {"meanVar_over_pred": float(v.mean() / sig2), "VarOfVar_over_independent": ratio,
                 "stdErrOfThatRatio": float(ratio * math.sqrt(2.0 / M)), "meanMaxAbsI_over_sigma": float(mx.mean() / math.sqrt(sig2))}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_sigma_dispersion.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
