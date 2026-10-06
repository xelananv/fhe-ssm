#!/usr/bin/env python3
"""tail_probe.jsonl (harness/overflow_tail_probe.cpp) -> tail_probe_tables.md. Records-only: every number is a field of a `class` record
or a Poisson / chi-square probability computed from such fields."""
import json, math, os
from scipy.stats import poisson, chi2
HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, "tail_probe.jsonl")) if l.strip().startswith("{")]
h = [r for r in recs if r["rec"] == "header"][0]; cls = [r for r in recs if r["rec"] == "class"]
L = [f"ring N = {h['ringDim']}, stock uniform-ternary key with Hamming weight {h['hamming']}, sigma_pred = {h['sigmaPred']:.4f}, R2 = {h['R2']:.4f}, "
     f"sample-variance inflation 1 + R2 - 0.6(h-1)/h = {h['sampleVarianceInflation']:.4f}; {h['ctsPerClass']} ciphertexts per class "
     f"(source sha256 {h['harnessSha256']}, argv `{h['cmdline']}`)\n",
     "| class | ciphertexts | towers | noiseScaleDeg | samples | sigma / sigma_pred | max|I| (in sigma) | chi^2 of per-ciphertext variances, correlated SE (dof; upper-tail p) | same with the independent-sample SE |",
     "|---|---|---|---|---|---|---|---|---|"]
for c in cls:
    L.append(f"| {c['class']} | {c['ciphertexts']} | {c['towers']} | {c['noiseScaleDeg']} | {c['samples']} | {c['sigmaOverPred']:.5f} | {c['maxAbsI']} ({c['maxOverSigma']:.3f}) | "
             f"{c['perCtVarChi2_correlatedSE']:.0f} ({c['dof']}; {chi2.sf(c['perCtVarChi2_correlatedSE'], c['dof']):.3g}) | {c['perCtVarChi2_independentSE']:.0f} ({chi2.sf(c['perCtVarChi2_independentSE'], c['dof']):.3g}) |")
L += ["", "Tail counts, observed / Gaussian expectation (two-sided Poisson-style reading: `p_hi` = P(count >= observed)):", "",
      "| class | " + " | ".join(f"|I| > {t['z']} sigma (b = {t['bound']})" for t in cls[0]["tails"]) + " |", "|---|" + "---|" * len(cls[0]["tails"])]
for c in cls:
    L.append(f"| {c['class']} | " + " | ".join(f"{t['observed']} / {t['expectedGaussian']:.2f} (ratio {t['observed'] / t['expectedGaussian']:.3f}, p_hi {poisson.sf(t['observed'] - 1, t['expectedGaussian']):.3g})"
                                               if t["expectedGaussian"] > 0 else "-" for t in c["tails"]) + " |")
open(os.path.join(HERE, "tail_probe_tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
