#!/usr/bin/env python3
"""Records-only paired comparison of two arms of the 2026-09-19 rental that differ in ONE thing (the bootstrap table): arm A (default LONG, K = 768 split)
and arm B (default CTRL512, stock K = 512) — same binary, box, store, prompts, decode rule, recorder. A lane-tick is PAIRED if the lane's fed token
sequence up to and including that tick is identical in both arms (so both circuits saw the same inputs and the same carried history).
Prints per tick: pairs, median relErrRms in each arm, the median of the per-pair ratio B/A, how many pairs have B > A; then the same over all pairs,
the top-1 / pick agreement of each arm on the paired lane-ticks, and arm B's recorder events beyond 512 with how their ticks decoded.
usage: compare_arms.py [--a LONG] [--b CTRL512] [--root pod_pull]"""
import argparse, json, math, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--a", default="LONG"); ap.add_argument("--b", default="CTRL512"); ap.add_argument("--root", default=os.path.join(HERE, "pod_pull"))
a = ap.parse_args()
def load(arm):
    rows = json.load(open(os.path.join(a.root, "s37", "serve_" + arm, "fidelity.json"))); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
    d = {}
    for r in rows: d[(r["lane"], r["tick"])] = r
    return d
A, B = load(a.a), load(a.b); TB = max(t for _, t in B) + 1; lanes = sorted({l for l, _ in B})
pairs = []
for l in lanes:
    same = True
    for t in range(TB):
        ra, rb = A.get((l, t)), B.get((l, t))
        if ra is None or rb is None: break
        same = same and ra.get("fed") == rb.get("fed")
        if not same: break
        pairs.append((t, l, ra, rb))
print(f"# {a.b} (arm B) against {a.a} (arm A): {TB} ticks in B, {len(pairs)} paired lane-ticks of {TB * len(lanes)} (paired = identical fed history in both arms)\n")
print("| tick | pairs | median relErrRms A | median relErrRms B | ratio of medians B/A | median of per-pair B/A | pairs with B > A |"); print("|---|---|---|---|---|---|---|")
def line(name, ps):
    ea = [p[2]["relErrRms"] for p in ps]; eb = [p[3]["relErrRms"] for p in ps]; q = [y / x for x, y in zip(ea, eb) if x > 0]
    print(f"| {name} | {len(ps)} | {st.median(ea):.6f} | {st.median(eb):.6f} | {st.median(eb) / st.median(ea):.3f} | {st.median(q):.3f} | {sum(1 for v in q if v > 1)} of {len(q)} |")
for t in range(TB):
    ps = [p for p in pairs if p[0] == t]
    if ps: line(str(t), ps)
line("all", pairs)
ea = [p[2]["relErrRms"] for p in pairs]; eb = [p[3]["relErrRms"] for p in pairs]; n = len(pairs); k = sum(1 for x, y in zip(ea, eb) if y > x)
z = (k - n / 2) / math.sqrt(n / 4) if n else float("nan")
lq = [math.log(y / x) for x, y in zip(ea, eb) if x > 0 and y > 0]; m = sum(lq) / len(lq); sd = st.pstdev(lq) / math.sqrt(len(lq))
print(f"\n- sign test over all pairs: B > A in {k} of {n} (z = {z:+.2f}); mean of ln(B/A) = {m:+.4f} +- {sd:.4f} (s.e.) => geometric-mean ratio B/A = {math.exp(m):.3f} [{math.exp(m - 2 * sd):.3f}, {math.exp(m + 2 * sd):.3f}] (2 s.e.)")
for nm, idx in ((a.a, 2), (a.b, 3)):
    print(f"- arm {nm} on the paired lane-ticks: raw top-1 {sum(1 for p in pairs if p[idx]['top1'])}/{n}, picks {sum(1 for p in pairs if p[idx].get('pickAgree'))}/{n}, decode-failed rows {sum(1 for p in pairs if p[idx].get('decFailed'))}")
# arm B recorder events
served = 0; ev = []; tbl = None
for l in open(os.path.join(a.root, "s37", "serve_" + a.b, "logs", "server.jsonl"), errors="replace"):
    l = l.strip()
    if not l.startswith("{"): continue
    try: r = json.loads(l)
    except Exception: continue
    if r.get("serve") == "served": served += 1
    elif "fideslibBootTable" in r: tbl = r
    elif r.get("bootRecord") is True and r.get("maxI", 0) > 480: r["tick"] = served; ev.append(r)
print(f"\n## arm {a.b}: library line `{json.dumps(tbl)[:200]}`; bootstraps with max overflow > 480 (its table's bound is {tbl.get('tableK') if tbl else '?'})\n")
print("| tick | bootstrap n | max overflow | stage | the tick in B: top-1 | median relErrRms | max relErrRms | same tick in A: median | max |"); print("|---|---|---|---|---|---|---|---|---|")
for r in ev:
    vb = [B[k2] for k2 in B if k2[1] == r["tick"]]; va = [A[k2] for k2 in A if k2[1] == r["tick"]]
    fb = (f"{sum(1 for x in vb if x['top1'])}/{len(vb)} | {st.median([x['relErrRms'] for x in vb]):.6f} | {max(x['relErrRms'] for x in vb):.6f}" if vb else "not compared | |")
    fa = (f"{st.median([x['relErrRms'] for x in va]):.6f} | {max(x['relErrRms'] for x in va):.6f}" if va else " | ")
    print(f"| {r['tick']} | {r['n']} | {r['maxI']} | `{r.get('stage')}` | {fb} | {fa} |")
