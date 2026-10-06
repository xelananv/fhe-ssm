#!/usr/bin/env python3
"""Pool the `evaluated` class of tail_probe.jsonl (run 1) and tail_probe2.jsonl (run 2, a different key) by sigma threshold; keep `fresh`
(run 1) beside it. Records-only: sums of the `tails` fields of the two `class` records. -> tail_probe_combined.md"""
import json, os
from scipy.stats import poisson
HERE = os.path.dirname(os.path.abspath(__file__))
def classes(fn): return [r for r in (json.loads(l) for l in open(os.path.join(HERE, fn)) if l.strip().startswith("{")) if r.get("rec") == "class"]
r1 = {c["class"]: c for c in classes("tail_probe.jsonl")}; r2 = {c["class"]: c for c in classes("tail_probe2.jsonl")}
rows = [("fresh, run 1", [r1["fresh"]]), ("evaluated, run 1", [r1["evaluated"]])]
if "evaluated" in r2: rows += [("evaluated, run 2 (new key)", [r2["evaluated"]]), ("evaluated, runs 1 + 2", [r1["evaluated"], r2["evaluated"]])]
zs = [t["z"] for t in r1["fresh"]["tails"]]
L = ["| sample | ciphertexts | samples | max|I| in sigma | " + " | ".join(f"> {z} sigma: observed / expected (p_hi)" for z in zs) + " |", "|---|---|---|---|" + "---|" * len(zs)]
for name, cs in rows:
    cells = []
    for k, z in enumerate(zs):
        o = sum(c["tails"][k]["observed"] for c in cs); e = sum(c["tails"][k]["expectedGaussian"] for c in cs)
        cells.append(f"{o} / {e:.2f} ({poisson.sf(o - 1, e):.3g})")
    L.append(f"| {name} | {sum(c['ciphertexts'] for c in cs)} | {sum(c['samples'] for c in cs)} | {max(c['maxOverSigma'] for c in cs):.3f} | " + " | ".join(cells) + " |")
import math
def Q(x): return 0.5 * math.erfc(x / math.sqrt(2))
h1 = [r for r in (json.loads(l) for l in open(os.path.join(HERE, "tail_probe.jsonl")) if l.strip().startswith("{")) if r.get("rec") == "header"][0]
ev = r1["evaluated"]; p1 = 2 * Q((ev["maxAbsI"] - 0.5) / h1["sigmaPred"])
n_all = ev["samples"] + (r2["evaluated"]["samples"] if "evaluated" in r2 else 0)
L += ["", f"The largest value of run 1's `evaluated` class is |I| = {ev['maxAbsI']} = {ev['maxOverSigma']:.3f} sigma. Under the Gaussian model P(|I_j| >= {ev['maxAbsI']}) = {p1:.3e} per "
          f"sample, i.e. an expected {p1 * ev['samples']:.2e} such values among the {ev['samples']} samples of that class and {p1 * n_all:.2e} among the {n_all} "
          f"`evaluated` samples of both runs."]
hist = {int(k): v for k, v in ev["hist"].items()}; top = sorted(((abs(k), k, v) for k, v in hist.items() if abs(k) >= 84), reverse=True)
L.append(f"Run 1, `evaluated`, every value with |I| >= 84 (value x count): " + ", ".join(f"{k:+d} x {v}" for a, k, v in top)
         + f"; the same for `fresh`: " + ", ".join(f"{k:+d} x {v}" for a, k, v in sorted(((abs(k), k, v) for k, v in {int(k): v for k, v in r1['fresh']['hist'].items()}.items() if abs(k) >= 84), reverse=True)))
L.append("")
L.append("Moments of the run-1 histograms (a sum of h uniforms has excess kurtosis -1.2/h = %.5f for this key's h = %d; standard errors for n independent samples: mean %.4f, skewness %.5f, kurtosis %.5f):"
         % (-1.2 / h1["hamming"], h1["hamming"], h1["sigmaPred"] / math.sqrt(ev["samples"]), math.sqrt(6.0 / ev["samples"]), math.sqrt(24.0 / ev["samples"])))
for name in ("fresh", "evaluated"):
    hh = {int(k): v for k, v in r1[name]["hist"].items()}; n = sum(hh.values())
    m1 = sum(k * v for k, v in hh.items()) / n; var = sum((k - m1) ** 2 * v for k, v in hh.items()) / n
    m3 = sum((k - m1) ** 3 * v for k, v in hh.items()) / n; m4 = sum((k - m1) ** 4 * v for k, v in hh.items()) / n
    b5 = int(math.floor(5 * h1["sigmaPred"]))
    L.append(f"- {name}: mean {m1:+.5f}, sigma/sigma_pred {math.sqrt(var) / h1['sigmaPred']:.5f}, skewness {m3 / var ** 1.5:+.6f}, excess kurtosis {m4 / var ** 2 - 3:+.6f}; "
             f"values above +5 sigma: {sum(v for k, v in hh.items() if k > b5)}, below -5 sigma: {sum(v for k, v in hh.items() if k < -b5)}")
open(os.path.join(HERE, "tail_probe_combined.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
