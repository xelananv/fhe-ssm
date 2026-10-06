#!/usr/bin/env python3
"""Records-only: how the logit error of the encrypted session evolves over a long autoregressive chain (rental of 2026-09-19).
From the MIRRORED fidelity.json (one row per lane per tick; relErrRms = rms(FHE logits - plaintext logits) / rms(plaintext logits) on the SAME
fed sequence). Prints, per block of ticks: the mean of the per-tick MEDIAN relErrRms (R3: rms, never max, as the level), the mean of the per-tick
90th percentile, raw top-1 flips with their plaintext top-two margins, decode-rule pick disagreements; then the per-lane late/early ratio and a
log-log slope between block centres (a description of the records, not a law).
usage: fidelity_trend.py [--arm LONG] [--root pod_pull] [--block 10]"""
import argparse, json, math, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="LONG"); ap.add_argument("--root", default=os.path.join(HERE, "pod_pull")); ap.add_argument("--block", type=int, default=10)
a = ap.parse_args()
rows = json.load(open(os.path.join(a.root, "s37", "serve_" + a.arm, "fidelity.json"))); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
by = {}
for r in rows: by.setdefault(r["tick"], []).append(r)
T = sorted(by); nT = len(T)
def pct(v, q): v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
print(f"# fidelity trend, arm {a.arm}: {nT} compared ticks, {len(rows)} lane-ticks")
print(f"\n| ticks | mean of per-tick median relErrRms | mean of per-tick 90th pct | raw top-1 flips (margins) | pick-only disagreements | generating lane-ticks |"); print("|---|---|---|---|---|---|")
centres, levels = [], []
for b0 in range(0, T[-1] + 1, a.block):
    tt = [t for t in T if b0 <= t < b0 + a.block]
    if not tt: continue
    m = [st.median([r["relErrRms"] for r in by[t]]) for t in tt]; q = [pct([r["relErrRms"] for r in by[t]], 0.9) for t in tt]
    fl = [r for t in tt for r in by[t] if not r["top1"]]; po = [r for t in tt for r in by[t] if r["top1"] and r.get("pickAgree") is False]
    g = sum(1 for t in tt for r in by[t] if r.get("generating")); n = sum(len(by[t]) for t in tt)
    print(f"| {b0}-{tt[-1]} | {sum(m) / len(m):.6f} | {sum(q) / len(q):.6f} | {len(fl)}" + (" (" + ", ".join(str(r.get("refTop1Margin")) for r in fl) + ")" if fl else "") + f" | {len(po)} | {g}/{n} |")
    if len(tt) == a.block: centres.append(b0 + a.block / 2); levels.append(sum(m) / len(m))
fl = [r for r in rows if not r["top1"]]; po = [r for r in rows if r["top1"] and r.get("pickAgree") is False]
print(f"\n- raw top-1 {len(rows) - len(fl)}/{len(rows)} = {100 * (1 - len(fl) / len(rows)):.3f} %; largest plaintext top-two margin among the flips: {max((r.get('refTop1Margin') or 0) for r in fl) if fl else None} logits; pick-only disagreements {len(po)}")
if len(centres) >= 3:
    for i, j in ((0, len(centres) - 1), (1, len(centres) - 1)):
        print(f"- log-log slope of the block level between block centres {centres[i]:.0f} and {centres[j]:.0f}: {math.log(levels[j] / levels[i]) / math.log(centres[j] / centres[i]):.3f}")
    # least squares of ln(level) on ln(centre), all full blocks
    X = [math.log(c) for c in centres]; Y = [math.log(v) for v in levels]; mx, my = sum(X) / len(X), sum(Y) / len(Y)
    sl = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X); ic = my - sl * mx
    print(f"- least-squares power law through all {len(centres)} full blocks: level = {math.exp(ic):.6f} * tick^{sl:.3f} (description of these records only)")
L = {}
for r in rows: L.setdefault(r["lane"], []).append(r)
if nT >= 30:
    e0, e1, l0, l1 = 5, 15, T[-1] - 10, T[-1]; ra = []
    for l, v in L.items():
        e = [r["relErrRms"] for r in v if e0 <= r["tick"] <= e1]; t = [r["relErrRms"] for r in v if l0 <= r["tick"] <= l1]
        if e and t: ra.append(st.median(t) / st.median(e))
    ra.sort(); print(f"- per lane, median relErrRms of ticks {l0}-{l1} over ticks {e0}-{e1}: lanes with ratio > 1: {sum(1 for x in ra if x > 1)} of {len(ra)}; median ratio {st.median(ra):.2f}, quartiles {pct(ra, 0.25):.2f} / {pct(ra, 0.75):.2f}")
top = sorted(rows, key=lambda r: -r["relErrRms"])[:5]
print("- five largest lane-ticks: " + "; ".join(f"tick {r['tick']} lane {r['lane']} {r['relErrRms']:.5f}" for r in top))
for r in top[:2]:
    seq = sorted((x for x in L[r["lane"]] if r["tick"] - 2 <= x["tick"] <= r["tick"] + 4), key=lambda x: x["tick"])
    print(f"  - lane {r['lane']} ticks {seq[0]['tick']}-{seq[-1]['tick']}: " + ", ".join(f"{x['relErrRms']:.4f}" for x in seq))
