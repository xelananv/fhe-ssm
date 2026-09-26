#!/usr/bin/env python3
"""tail_probe2.jsonl (harness/overflow_tail_probe2.cpp, the diagnostic replication) -> tail_probe2_tables.md. Records-only."""
import json, math, os
from scipy.stats import poisson, chi2, binomtest
HERE = os.path.dirname(os.path.abspath(__file__))
recs = []
for l in open(os.path.join(HERE, "tail_probe2.jsonl")):
    if l.strip().startswith("{"):
        try: recs.append(json.loads(l))
        except json.JSONDecodeError: pass
h = [r for r in recs if r["rec"] == "header"][0]; cls = [r for r in recs if r["rec"] == "class"]; out = [r for r in recs if r["rec"] == "outlier"]
prog = [r for r in recs if r["rec"] == "progress"]
last = cls[-1] if cls else (prog[-1] if prog else None)
sig = h["sigmaPred"]
def Q(x): return 0.5 * math.erfc(x / math.sqrt(2))
L = [f"ring N = {h['ringDim']}, a NEW stock uniform-ternary key (Hamming weight {h['hamming']}, sigma_pred = {sig:.4f}), class `evaluated` only, "
     f"source sha256 {h['harnessSha256']}, argv `{h['cmdline']}`; status: {'complete' if cls else 'PARTIAL (last progress record)'}\n"]
if last:
    L += ["| ciphertexts | samples | sigma / sigma_pred | max|I| (in sigma) | chi^2 of per-ciphertext variances, correlated SE (dof; upper-tail p) | " +
          " | ".join(f"|I| > {t['z']} sigma (b = {t['bound']})" for t in last["tails"]) + " |", "|---|---|---|---|---|" + "---|" * len(last["tails"]),
          f"| {last['ciphertexts']} | {last['samples']} | {last['sigmaOverPred']:.5f} | {last['maxAbsI']} ({last['maxOverSigma']:.3f}) | "
          f"{last['perCtVarChi2_correlatedSE']:.0f} ({last['dof']}; {chi2.sf(last['perCtVarChi2_correlatedSE'], last['dof']):.3g}) | " +
          " | ".join(f"{t['observed']} / {t['expectedGaussian']:.2f} (p_hi {poisson.sf(t['observed'] - 1, t['expectedGaussian']):.3g})" for t in last["tails"]) + " |", ""]
thr = min((r["maxAbsI"] for r in out), default=None)
L.append(f"Ciphertexts whose final max|I| reached the logging threshold (5.3 sigma): {len(out)}. For each: the overflow at the SAME coefficient one step earlier "
         f"(`scalar` stage = the ciphertext before the last rescale), the largest max|I| over the 11 earlier stages of that ciphertext, and the decrypt check.\n")
L += ["| ct | I at argmax | in sigma | argmax j | that ciphertext's sigma_hat | # |I| > 4 sigma in it | frac at argmax | I at the same j before the last rescale | largest stage max|I| | relErrRms of the decrypted result vs the clear computation |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(out, key=lambda r: -r["maxAbsI"]):
    L.append(f"| {r['ct']} | {r['signedI']:+d} | {r['maxOverSigma']:.2f} | {r['argmax']} | {r['sigmaHat']:.2f} | {r['nOver4sigma']} | {r['fracAtArgmax']:+.2e} | "
             f"{r['stages'][-1]['IatFinalArgmax']:+d} | {max(s['maxAbsI'] for s in r['stages'])} | {r['relErrRms']:.1e} |")
npos = sum(1 for r in out if r["signedI"] > 0)
if out:
    L.append(f"\nsigns: {npos} positive, {len(out) - npos} negative (two-sided binomial p = {binomtest(npos, len(out), 0.5).pvalue:.3f}); "
             f"argmax in the upper half of the ring (j >= N/2): {sum(1 for r in out if r['argmax'] >= h['ringDim'] // 2)} of {len(out)}")
    n_ct = last["ciphertexts"] if last else 0
    b = thr - 1 if thr else None
    L.append(f"expected number of ciphertexts with max|I| >= {thr} under the Gaussian model: {n_ct * (1 - (1 - 2 * Q((thr - 0.5) / sig)) ** h['ringDim']):.2f} (observed {len(out)})")
open(os.path.join(HERE, "tail_probe2_tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
