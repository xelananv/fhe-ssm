#!/usr/bin/env python3
"""Probe C: the distribution of the ModRaise overflow I = round((c0 + c1*s)/q0) on SAVED ciphertexts of the recorded
ring-2^17 session, from the tower-0 dump written by harness/overflow_dist_dump.cpp.

INPUT  (scratch, OUTSIDE the repo, mode 700): secret.i8, ct_XXX.c0.i64, ct_XXX.c1.i64, index.jsonl
OUTPUT (repo, AGGREGATES ONLY -- no coefficient of the secret and no per-coefficient value derived from it is written):
       probe_c_stats.json, probe_c_tables.md next to this script.

x_j = c0_j/q0 + sum_k s-hat(j,k) * c1_k/q0   (negacyclic: s-hat(j,k) = s_{j-k} for k <= j, -s_{j-k+N} for k > j)
is computed by a twisted FFT in float64; I_j = round(x_j).  The float result is VALIDATED against exact Python-integer
arithmetic on a handful of coefficients per ciphertext, always including that ciphertext's argmax |I|.

usage: overflow_dist_stats.py --scratch /tmp/.../overflow_dist
"""
import argparse, json, math, os, sys
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--scratch", required=True, action="append", help="dump directory; repeat to pool several dumps made under the SAME secret")
ap.add_argument("--exact-per-ct", type=int, default=6); ap.add_argument("--tag", default="", help="suffix for the output file names")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
for S in args.scratch:
    if not os.path.realpath(S).startswith("/tmp/"):
        sys.exit("scratch must be under /tmp/")

hdr = None; cts = []; s = None
for S in args.scratch:
    idx = [json.loads(l) for l in open(os.path.join(S, "index.jsonl")) if l.strip().startswith("{")]
    h_ = [r for r in idx if r.get("rec") == "header"][0]
    s_ = np.fromfile(os.path.join(S, "secret.i8"), dtype=np.int8).astype(np.int64)
    if hdr is None: hdr, s = h_, s_
    else: assert h_["q0"] == hdr["q0"] and h_["ringDim"] == hdr["ringDim"] and np.array_equal(s_, s), "dumps were not made under the same context and secret"
    for r in idx:
        if r.get("rec") == "ct": r["_dir"] = S; cts.append(r)
N = hdr["ringDim"]; q0 = hdr["q0"]
assert s.size == N and set(np.unique(s).tolist()) <= {-1, 0, 1}
h = int(np.count_nonzero(s)); sumS2 = int(np.sum(s * s))
sigma_pred = math.sqrt((sumS2 + 1) / 12.0)

# twisted FFT for the negacyclic product
k = np.arange(N); tw = np.exp(1j * np.pi * k / N); Fs = np.fft.fft(s.astype(np.float64) * tw)
def negconv_float(c1_over_q):
    return np.real(np.fft.ifft(np.fft.fft(c1_over_q * tw) * Fs) * np.conj(tw))

# exact (c1*s)_j with Python integers: split c1 = hi*2^30 + lo so every numpy partial sum stays inside int64
def exact_t(j, c0, c1):
    sh = np.empty(N, dtype=np.int64)            # s-hat(j, k) for k = 0..N-1
    sh[: j + 1] = s[j::-1]                      # k <= j : s_{j-k}
    if j + 1 < N: sh[j + 1:] = -s[:j:-1]        # k >  j : -s_{j-k+N}  (indices N-1 .. j+1)
    lo = c1 & ((1 << 30) - 1); hi = c1 >> 30    # arithmetic shift: c1 == hi * 2^30 + lo exactly, 0 <= lo < 2^30
    return int(c0[j]) + (int(np.sum(sh * hi)) << 30) + int(np.sum(sh * lo))

def Qf(x): return 0.5 * math.erfc(x / math.sqrt(2))

# The N values I_j of ONE ciphertext are not independent: they are N differently-signed sums over the SAME c1, and
#   Cov(I_j, I_j') = (1/12) * A_k,  A_k = negacyclic autocorrelation of s at lag k = j'-j,  rho_k = A_k / h  (tiny, ~1/sqrt(h)).
# For the sample variance what matters is Cov(I_j^2, I_j'^2) = 2 Cov^2 + (joint 4th cumulant). The summands are UNIFORM on (-1/2, 1/2)
# (4th cumulant -1/120), so the cumulant term is -(1/120) * #{positions where both shifted copies of s are nonzero}; summed over the
# nonzero lags that count is exactly h(h-1). With 2 sigma^4 = 2 h^2/144 this gives
#   Var(sample variance) = (2 sigma^4 / N) * (1 + R2 - 0.6 (h-1)/h),   R2 = sum_{k != 0} rho_k^2   (an aggregate of the secret: one number)
# and NOT the 2 sigma^4 / N of independent samples. sim_sigma_dispersion.py checks this formula on a toy ring (predicted 1.34, simulated 1.27-1.35).
ac = np.real(np.fft.ifft(np.abs(Fs) ** 2) * np.conj(tw))        # negacyclic autocorrelation of s, lag 0..N-1
R2 = float(np.sum((ac[1:] / ac[0]) ** 2))
var_inflate = 1.0 + R2 - 0.6 * (h - 1) / h
se_inflate = math.sqrt(var_inflate)

rng = np.random.default_rng(20260918)
per = []; pooled = []; worst_float_err = 0.0; exact_checked = 0; exact_mismatch = 0
for r in cts:
    c0 = np.fromfile(os.path.join(r["_dir"], "ct_%03d.c0.i64" % r["k"]), dtype=np.int64)
    c1 = np.fromfile(os.path.join(r["_dir"], "ct_%03d.c1.i64" % r["k"]), dtype=np.int64)
    assert c0.size == N and c1.size == N and r["q0"] == q0
    x = c0.astype(np.float64) / q0 + negconv_float(c1.astype(np.float64) / q0)
    I = np.rint(x).astype(np.int64); frac = x - I
    jmax = int(np.argmax(np.abs(I)))
    for j in [jmax] + [int(v) for v in rng.integers(0, N, size=args.exact_per_ct)]:
        t = exact_t(j, c0, c1)
        Iex = (t + q0 // 2) // q0                       # floor((t + q0/2)/q0): Python's // floors for negatives too
        fex = (t - Iex * q0) / q0
        exact_checked += 1
        if Iex != int(I[j]): exact_mismatch += 1
        worst_float_err = max(worst_float_err, abs((Iex + fex) - x[j]))
    c1u = c1.astype(np.float64) / q0
    per.append({"k": len(per), "file": "/".join(r["path"].split("/")[-4:-3] + r["path"].split("/")[-2:]), "towers": r["towers"], "noiseScaleDeg": r["noiseScaleDeg"],
                "log2ScalingFactor": r["log2ScalingFactor"], "meanI": float(I.mean()), "sigmaI": float(I.std()), "maxAbsI": int(np.abs(I).max()),
                "maxAbsI_over_sigmaPred": float(np.abs(I).max() / sigma_pred), "rmsFrac": float(np.sqrt(np.mean(frac ** 2))),
                "maxAbsFrac": float(np.abs(frac).max()), "c1_mean": float(c1u.mean()), "c1_std": float(c1u.std()),
                "nOver3sigma": int(np.count_nonzero(np.abs(I) > int(math.floor(3 * sigma_pred)))), "nOver4sigma": int(np.count_nonzero(np.abs(I) > int(math.floor(4 * sigma_pred)))),
                "c1_std_uniform": 1 / math.sqrt(12)})
    pooled.append(I)
P_ = np.concatenate(pooled); n = int(P_.size)
sig = float(P_.std()); mean = float(P_.mean())
m4 = float(np.mean((P_ - mean) ** 4)); kurt_excess = m4 / sig ** 4 - 3.0
absI = np.abs(P_)
top10 = np.sort(absI)[-10:][::-1].tolist()

tails = []
for z in (1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6):
    b = int(math.floor(z * sigma_pred)); obs = int(np.count_nonzero(absI > b))
    exp = n * 2 * Qf((b + 0.5) / sigma_pred)
    tails.append({"z": z, "bound": b, "observed": obs, "expectedGaussian": exp, "ratio": (obs / exp) if exp > 0 else None,
                  "poissonSigma": math.sqrt(exp)})
probs = (0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999)
try:
    from scipy.stats import norm
    zq = [float(norm.ppf(p)) for p in probs]
except Exception:
    zq = [None] * len(probs)
quant = []
srt = np.sort(P_)
for p, z in zip(probs, zq):
    e = float(srt[min(n - 1, int(math.floor(p * n)))])
    quant.append({"p": p, "empirical": e, "gaussian_sigmaPred": (z * sigma_pred) if z is not None else None,
                  "ratio": (e / (z * sigma_pred)) if z not in (None, 0.0) else None})
# histogram in units of sigma_pred (aggregate over all ciphertexts)
edges = np.arange(-6.0, 6.0001, 0.5) * sigma_pred
hist, _ = np.histogram(P_, bins=edges)
# I is an integer = round(x): the integers in [lo, hi) are the continuous interval [ceil(lo) - 0.5, ceil(hi) - 0.5)
# (numpy's last bin is closed on the right; +-6 sigma_pred is not an integer here, so that changes nothing)
exp_hist = [n * (Qf((math.ceil(lo) - 0.5) / sigma_pred) - Qf((math.ceil(hi) - 0.5) / sigma_pred)) for lo, hi in zip(edges[:-1], edges[1:])]
chi2 = float(sum((o - e) ** 2 / e for o, e in zip(hist.tolist(), exp_hist) if e >= 20)); dof = sum(1 for e in exp_hist if e >= 20) - 1

reqs = [p for p in per if "/req." in p["file"]]; resps = [p for p in per if "/resp." in p["file"]]
def pool_of(sel): return np.concatenate([pooled[i] for i, p in enumerate(per) if p in sel]) if sel else np.array([0])
# probability, under the Gaussian model with sigma_pred, that the largest of n samples is at least the observed maximum
def p_max_at_least(m, n_): return 1.0 - (1.0 - 2 * Qf((m - 0.5) / sigma_pred)) ** n_
se_ct = sigma_pred / math.sqrt(2 * N) * se_inflate            # standard error of ONE ciphertext's sigma estimate
split = {}
for name, sel in (("requests", reqs), ("responses", resps)):
    a_ = np.abs(pool_of(sel)); n_ = int(a_.size)
    zs = [(p["sigmaI"] - sigma_pred) / se_ct for p in sel]
    try:
        from scipy.stats import chi2 as _chi2
        p_hi = float(_chi2.sf(sum(z * z for z in zs), len(zs)))
    except Exception:
        p_hi = None
    split[name] = {"perCtSigmaChi2_pUpper": p_hi, "ciphertexts": len(sel), "samples": n_, "sigma": float(pool_of(sel).std()), "sigmaStdErr": float(sigma_pred / math.sqrt(2 * n_) * se_inflate),
                   "perCtSigmaChi2": float(sum(z * z for z in zs)), "perCtSigmaChi2Dof": len(zs), "perCtSigmaChi2_ifIndependent": float(sum(z * z for z in zs) * se_inflate ** 2),
                   "maxAbsI": int(a_.max()), "pMaxAtLeast": p_max_at_least(int(a_.max()), n_),
                   "tails": [{"z": z, "bound": int(math.floor(z * sigma_pred)), "observed": int(np.count_nonzero(a_ > int(math.floor(z * sigma_pred)))),
                              "expectedGaussian": n_ * 2 * Qf((int(math.floor(z * sigma_pred)) + 0.5) / sigma_pred)} for z in (3, 4, 4.5, 5, 5.5)]}
stats = {"ringDim": N, "q0": q0, "hamming": h, "sumS2": sumS2, "sigmaPred": sigma_pred, "K": 512, "K_over_sigmaPred": 512 / sigma_pred,
         "ciphertexts": len(per), "pooledSamples": n, "pooledMean": mean, "pooledSigma": sig, "pooledSigma_over_pred": sig / sigma_pred,
         "pooledMaxAbsI": int(absI.max()), "pooledTop10AbsI": [int(v) for v in top10], "pooledMaxAbsI_over_sigmaPred": float(absI.max() / sigma_pred), "pMaxAtLeast_pooled": p_max_at_least(int(absI.max()), n),
         "pooledSigmaStdErr": sigma_pred / math.sqrt(2 * n) * se_inflate, "secretAutocorrR2": R2, "sampleVarianceInflation": var_inflate, "perCtSigmaStdErr": se_ct, "kurtosisStdErr": math.sqrt(24.0 / n), "split": split, "excessKurtosis": kurt_excess,
         "requestsSigma": float(pool_of(reqs).std()), "responsesSigma": float(pool_of(resps).std()),
         "exactChecked": exact_checked, "exactMismatch": exact_mismatch, "worstFloatVsExactAbs": worst_float_err,
         "tails": tails, "quantiles": quant, "histEdgesInSigmaPred": [float(e / sigma_pred) for e in edges], "histObserved": hist.tolist(),
         "histExpectedGaussian": exp_hist, "chi2": chi2, "chi2dof": dof, "perCiphertext": per,
         "dumpHeader": {k: hdr[k] for k in hdr if k not in ("cmdline",)}}
json.dump(stats, open(os.path.join(HERE, "probe_c_stats%s.json" % args.tag), "w"), indent=1)

L = []
L.append(f"ring N = {N}, q0 = {q0}; secret: ternary = {hdr['secretIsTernary']}, towers 0/1 agree = {hdr['secretTowers01Agree']}, Hamming weight h = {h} "
         f"(h/N = {h / N:.5f}; uniform ternary expects 2/3), predicted sigma_I = sqrt((h+1)/12) = {sigma_pred:.4f}, K/sigma = {512 / sigma_pred:.4f}\n")
L.append(f"pooled over {len(per)} ciphertexts: n = {n}, mean = {mean:.4f}, sigma = {sig:.4f} (measured/predicted = {sig / sigma_pred:.5f}), "
         f"max|I| = {int(absI.max())} = {absI.max() / sigma_pred:.3f} sigma_pred, excess kurtosis = {kurt_excess:+.5f}; "
         f"requests only sigma = {stats['requestsSigma']:.4f}, responses only sigma = {stats['responsesSigma']:.4f}\n")
L.append(f"under the Gaussian model with sigma_pred, P(max of n samples >= {int(absI.max())}) = {p_max_at_least(int(absI.max()), n):.4f}; "
         f"R2 = sum over nonzero lags of the squared normalised negacyclic autocorrelation of the secret = {R2:.4f}, so a sigma estimate has standard error "
         f"sigma/sqrt(2n) * sqrt(1 + R2 - 0.6(h-1)/h) (factor {se_inflate:.4f}; derivation in the script, checked by sim_sigma_dispersion.py): "
         f"{sigma_pred / math.sqrt(2 * n) * se_inflate:.4f} pooled, {se_ct:.4f} for one ciphertext "
         f"(it would be {sigma_pred / math.sqrt(2 * n):.4f} and {sigma_pred / math.sqrt(2 * N):.4f} for independent samples); "
         f"standard error of the excess kurtosis (independent-sample formula) = {math.sqrt(24.0 / n):.5f}\n")
for name in ("requests", "responses"):
    sp = split[name]
    L.append(f"{name}: {sp['ciphertexts']} ciphertexts, n = {sp['samples']}, sigma = {sp['sigma']:.4f} (+- {sp['sigmaStdErr']:.4f}), max|I| = {sp['maxAbsI']} "
             f"(P(max >= that) = {sp['pMaxAtLeast']:.4f}); chi^2 of the per-ciphertext sigma estimates against sigma_pred = {sp['perCtSigmaChi2']:.1f} on {sp['perCtSigmaChi2Dof']} (upper-tail p = {sp['perCtSigmaChi2_pUpper']:.4f}) "
             f"(it would read {sp['perCtSigmaChi2_ifIndependent']:.1f} with the independent-sample standard error); tail counts observed/expected: "
             + ", ".join(f"|I|>{t['bound']}: {t['observed']}/{t['expectedGaussian']:.2f}" for t in sp["tails"]) + "\n")
L.append(f"ten largest |I| of the pooled sample: {[int(v) for v in top10]} (in units of sigma_pred: {[round(v / sigma_pred, 3) for v in top10]})\n")
L.append(f"float-vs-exact validation: {exact_checked} coefficients recomputed with Python integers (each ciphertext's argmax|I| plus {args.exact_per_ct} random), "
         f"I mismatches = {exact_mismatch}, worst |x_float - x_exact| = {worst_float_err:.3e}\n")
e3 = N * 2 * Qf((int(math.floor(3 * sigma_pred)) + 0.5) / sigma_pred); e4 = N * 2 * Qf((int(math.floor(4 * sigma_pred)) + 0.5) / sigma_pred)
L.append(f"per ciphertext (Gaussian expectation per ciphertext: #(|I| > 3 sigma) = {e3:.1f}, #(|I| > 4 sigma) = {e4:.2f}):\n")
L.append("| k | file | towers | noiseScaleDeg | log2 SF | mean I | sigma_I | max|I| | max|I|/sigma_pred | #>3 sigma | #>4 sigma | rms frac | max|frac| | std(c1/q0) (uniform: 0.28868) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for p in per:
    L.append(f"| {p['k']} | {p['file']} | {p['towers']} | {p['noiseScaleDeg']} | {p['log2ScalingFactor']:.2f} | {p['meanI']:+.3f} | {p['sigmaI']:.3f} | {p['maxAbsI']} | "
             f"{p['maxAbsI_over_sigmaPred']:.3f} | {p['nOver3sigma']} | {p['nOver4sigma']} | {p['rmsFrac']:.3e} | {p['maxAbsFrac']:.3e} | {p['c1_std']:.5f} |")
L.append("\nTail counts, pooled (Gaussian expectation uses sigma_pred and the integer continuity correction P(|I| > b) = 2Q((b+0.5)/sigma)):\n")
L.append("| z | bound b = floor(z*sigma_pred) | observed #(|I| > b) | Gaussian expectation | observed/expected | sqrt(expected) |")
L.append("|---|---|---|---|---|---|")
for t in tails:
    L.append(f"| {t['z']} | {t['bound']} | {t['observed']} | {t['expectedGaussian']:.2f} | {t['ratio']:.4f} | {t['poissonSigma']:.2f} |" if t["ratio"] is not None else f"| {t['z']} | {t['bound']} | {t['observed']} | 0 | - | - |")
L.append("\nNormal-quantile comparison, pooled:\n")
L.append("| p | empirical quantile of I | sigma_pred * z_p | ratio |")
L.append("|---|---|---|---|")
for qn in quant:
    L.append(f"| {qn['p']} | {qn['empirical']:.1f} | {qn['gaussian_sigmaPred']:.2f} | {qn['ratio']:.4f} |" if qn["ratio"] is not None else f"| {qn['p']} | {qn['empirical']:.1f} | {qn['gaussian_sigmaPred']} | - |")
L.append(f"\nhistogram in 0.5-sigma_pred bins over [-6, 6] sigma: chi^2 = {chi2:.1f} on {dof} degrees of freedom (bins with expectation >= 20)\n")
open(os.path.join(HERE, "probe_c_tables%s.md" % args.tag), "w").write("\n".join(L) + "\n")
print("\n".join(L))
