#!/usr/bin/env python3
"""Records + one derivation, for the question of 2026-09-19 ("what's the headroom on the logit error? how much longer until it overflows?").
RECORDS (mirrored fidelity.json): the plaintext model's top-two margin distribution, the raw top-1 flips, and the flip rate that a larger error
level would give (Gaussian error of the top-two logit difference, its std calibrated on the run: rate = 0.399 * rho * sigma_d, rho = margin density
near zero; then integrated over the EMPIRICAL margins). DERIVATION (weights only, no measurement): how a per-tick error in the recurrent state
S' = a S + (1 - a) x accumulates, from the model's own per-channel decays a = a_min + (a_max - a_min) sigmoid(a_raw): var(t)/var(1) =
(1 - a^2t)/(1 - a^2), averaged over all channels. usage: error_headroom.py [--arm LONG] [--weights ~/Documents/fhe-ssm-backup/mac_art/pbd430a_weights.npz]"""
import argparse, bisect, json, math, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="LONG"); ap.add_argument("--root", default=os.path.join(HERE, "pod_pull"))
ap.add_argument("--weights", default=os.path.expanduser("~/Documents/fhe-ssm-backup/mac_art/pbd430a_weights.npz")); ap.add_argument("--config", default=os.path.join(REPO, "ml-eval", "artifacts", "pbd430a_config.json"))
a = ap.parse_args()
rows = json.load(open(os.path.join(a.root, "s37", "serve_" + a.arm, "fidelity.json"))); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
n = len(rows); T = max(r["tick"] for r in rows) + 1; m = sorted(r["refTop1Margin"] for r in rows if r.get("refTop1Margin") is not None)
fl = [r for r in rows if not r["top1"]]; lvl = st.median([r["relErrRms"] for r in rows])
print(f"# logit-error headroom, arm {a.arm}: {T} ticks, {n} lane-ticks\n")
print("## records: the plaintext model's top-two margin (logits) and the flips\n")
print("- margin percentiles: " + ", ".join(f"p{q} {m[int(q / 100 * len(m))]:.4f}" for q in (1, 5, 10, 25, 50, 75)))
print(f"- raw top-1 flips: {len(fl)} of {n} = {100 * len(fl) / n:.3f} %; largest margin among them {max(r['refTop1Margin'] for r in fl):.4f}; lane-ticks with a margin at or below that: {sum(1 for v in m if v <= max(r['refTop1Margin'] for r in fl))}")
rho = sum(1 for v in m if v <= 0.05) / len(m) / 0.05; rate = len(fl) / n; sig = rate / (0.3989 * rho)
print(f"- margin density near zero {rho:.3f} per logit => std of the error of the top-two logit difference {sig:.4f} logits at the run's median relErrRms {lvl:.6f}; the MEDIAN decision margin is {st.median(m) / sig:.0f} of those stds away")
def fr(sd): return sum(0.5 * math.erfc(v / sd / math.sqrt(2)) for v in m) / len(m)
print("\n| error level | relErrRms | expected raw top-1 flips per lane-tick | top-1 agreement |"); print("|---|---|---|---|")
for g in (1, 2, 3, 5, 10, 30, 100): print(f"| x{g} | {lvl * g:.4f} | {100 * fr(sig * g):.2f} % | {100 * (1 - fr(sig * g)):.2f} % |")
by = {}
for r in rows: by.setdefault(r["tick"], []).append(r["relErrRms"])
cs, ls = [], []
for b0 in range(0, T - T % 10, 10): cs.append(b0 + 5); ls.append(sum(st.median(by[t]) for t in range(b0, b0 + 10)) / 10)
X = [math.log(c) for c in cs]; Y = [math.log(v) for v in ls]; mx, my = sum(X) / len(X), sum(Y) / len(Y)
al = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X); c0 = math.exp(my - al * mx)
print(f"\n## the fitted power law (a DESCRIPTION of ticks 5..{cs[-1]}, not a law): level = {c0:.6f} * tick^{al:.3f}; block levels: " + ", ".join(f"{v:.6f}" for v in ls) + "\n")
print("| error level | reached at tick (if the power law held) | serving time at 176 s per tick |"); print("|---|---|---|")
for g in (1.5, 2, 3, 5, 10):
    t = (ls[-1] * g / c0) ** (1 / al); print(f"| x{g} of the last block ({ls[-1] * g:.4f}) | {t:,.0f} | {t * 176 / 86400:,.1f} days |")
try:
    import numpy as np
    cfg = json.load(open(a.config)); z = np.load(a.weights); lo, hi = cfg["a_min"], cfg["a_max"]
    dec = np.concatenate([(lo + (hi - lo) / (1 + np.exp(-z[k].astype(np.float64)))).ravel() for k in z.files if "a_raw" in k]); tau = 1 / (1 - dec)
    print(f"\n## derivation from the weights: the model's memory is bounded ({dec.size} decay channels, a in [{cfg['a_min']}, {cfg['a_max']}] by construction)\n")
    print(f"- trained decays: min {dec.min():.5f}, median {np.median(dec):.5f}, p99 {np.percentile(dec, 99):.5f}, max {dec.max():.5f}; time constants 1/(1-a): median {np.median(tau):.0f} ticks, p99 {np.percentile(tau, 99):.0f}, max {tau.max():.0f}; channels above 100 ticks: {100 * np.mean(tau > 100):.1f} %, above 300: {100 * np.mean(tau > 300):.1f} %")
    print("- accumulation of a per-tick state error, std factor sqrt(mean_c (1 - a^2t)/(1 - a^2)): " + ", ".join(f"t={t}: {math.sqrt(np.mean((1 - dec ** (2 * t)) / (1 - dec ** 2))):.2f}" for t in (1, 5, 15, 25, 50, 100, 150, 300, 1000)) + f"; slowest channel alone saturates at {1 / math.sqrt(1 - dec.max() ** 2):.1f}")
    print("- S' = a S + (1 - a) x is a convex average per channel (bounded by the inputs); the only other cross-tick carry is the previous token's normalised u (one tick of memory); the residual stream restarts from the embedding every tick")
except Exception as e: print(f"\n(weights not readable here: {e})")
