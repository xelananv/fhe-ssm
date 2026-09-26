#!/usr/bin/env python3
"""Records-only companion of long_report.py for the bootstrap flight recorder (rental of 2026-09-19), from the MIRRORED server.jsonl:
  1. every bootstrap whose max overflow exceeds 512 (the stock table's bound): during which tick, at which stage, and how that tick decoded;
  2. is the exceedance rate STATIONARY over the run? counts above 430 / 460 / 480 per block of ticks against the Gaussian model
     (sigma = the recorder's sigmaIPred, integer overflow => B + 1/2, N = 2^17 coefficients; nothing fitted);
  3. the same per STAGE family (rmsnorm / scan / other), since a bootstrap's input distribution could in principle differ by site.
A bootstrap belongs to tick t if its bootRecord line precedes the t-th `served` line (tick 0 = the cold tick).
usage: recorder_events.py [--arm LONG] [--root pod_pull] [--block 10]"""
import argparse, json, math, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="LONG"); ap.add_argument("--root", default=os.path.join(HERE, "pod_pull")); ap.add_argument("--block", type=int, default=10)
a = ap.parse_args()
D = os.path.join(a.root, "s37", "serve_" + a.arm); served = 0; rec = []; sig = 85.4
for l in open(os.path.join(D, "logs", "server.jsonl"), errors="replace"):
    l = l.strip()
    if not l.startswith("{"): continue
    try: r = json.loads(l)
    except Exception: continue
    if r.get("serve") == "served": served += 1
    elif r.get("bootRecordArmed") and r.get("sigmaIPred"): sig = r["sigmaIPred"]
    elif r.get("bootRecord") is True: r["tick"] = served; rec.append(r)
rows = json.load(open(os.path.join(D, "fidelity.json"))); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
by = {}
for r in rows: by.setdefault(r["tick"], []).append(r)
N = 131072
def pmax(B): return 1 - (1 - math.erfc((B + 0.5) / sig / math.sqrt(2))) ** N
live = [r for r in rec if r.get("rmsI", 0) > 1]; done = [r for r in live if r["tick"] < served]      # only completed ticks enter the per-block table
print(f"# recorder events, arm {a.arm}: {served} served ticks, {len(rec)} bootstraps recorded, {len(live)} with non-trivial inputs, sigma {sig:.2f}")
print(f"\n## bootstraps with max overflow > 512 (model expectation over {len(live)} bootstraps: {pmax(512) * len(live):.2f})\n")
print("| tick | bootstrap n | max overflow | coefficients > 512 | stage | rmsI | tick decoded: top-1 | picks | relErrRms median | max |"); print("|---|---|---|---|---|---|---|---|---|---|")
for r in sorted((r for r in live if r["maxI"] > 512), key=lambda r: r["n"]):
    v = by.get(r["tick"])
    f = (f"{sum(1 for x in v if x['top1'])}/{len(v)} | {sum(1 for x in v if x.get('pickAgree'))}/{len(v)} | {st.median([x['relErrRms'] for x in v]):.6f} | {max(x['relErrRms'] for x in v):.6f}" if v else "not compared yet | | |")
    print(f"| {r['tick']} | {r['n']} | {r['maxI']} | {r.get('coeffsOver512')} | `{r.get('stage')}` | {r['rmsI']} | {f} |")
print(f"\n## exceedances per block of {a.block} completed ticks (observed / expected under the model)\n")
print("| ticks | bootstraps | > 430 | > 460 | > 480 | > 512 | rmsI mean |"); print("|---|---|---|---|---|---|---|")
T = sorted({r["tick"] for r in done})
for b0 in range(0, (max(T) + 1) if T else 0, a.block):
    blk = [r for r in done if b0 <= r["tick"] < b0 + a.block]
    if not blk: continue
    cells = " | ".join(f"{sum(1 for r in blk if r['maxI'] > B)} / {pmax(B) * len(blk):.1f}" for B in (430, 460, 480, 512))
    print(f"| {b0}-{min(b0 + a.block - 1, max(T))} | {len(blk)} | {cells} | {sum(r['rmsI'] for r in blk) / len(blk):.3f} |")
cells = " | ".join(f"{sum(1 for r in done if r['maxI'] > B)} / {pmax(B) * len(done):.1f}" for B in (430, 460, 480, 512))
print(f"| all | {len(done)} | {cells} | {sum(r['rmsI'] for r in done) / max(len(done), 1):.3f} |")
print("\n## by stage family (all recorded bootstraps with non-trivial inputs)\n")
print("| stage family | bootstraps | > 430 | > 460 | > 480 | > 512 | rmsI mean |"); print("|---|---|---|---|---|---|---|")
def fam(s):
    s = str(s); return "rmsnorm" if "rmsnorm" in s else "scan" if "scan" in s else "canon/carry" if ("canon" in s or "carry" in s) else s.split(".")[-2] if s.count(".") >= 2 else s
F = {}
for r in live: F.setdefault(fam(r.get("stage")), []).append(r)
for k in sorted(F, key=lambda k: -len(F[k])):
    blk = F[k]; cells = " | ".join(f"{sum(1 for r in blk if r['maxI'] > B)} / {pmax(B) * len(blk):.1f}" for B in (430, 460, 480, 512))
    print(f"| {k} | {len(blk)} | {cells} | {sum(r['rmsI'] for r in blk) / len(blk):.3f} |")
