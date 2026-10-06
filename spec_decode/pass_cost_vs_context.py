#!/usr/bin/env python3
"""pass_cost_vs_context.py -- the per-token cost of a stateful FHE session as a
function of context position (tick), from the records a demo session leaves.

Inputs (either or both):
  --session DIR      client session dir with transcript.json (tick, encSec, serveSec, decSec;
                     serveSec = upload + server tick + download, i.e. it INCLUDES the wire)
  --server-log PATH  the pod's server.log / server.jsonl: the {"serve":"served",...} lines
                     (reqMsPerToken, reqBoots, reqEncPtMs, peakVramGB per request) and, when
                     present, the {"vramTrace":"serve.done.N",...} lines (per device usedGB)

Outputs (in --out, default = the session dir or the server log's dir):
  pass_cost_vs_context.csv   one row per tick
  pass_cost_vs_context.md    the table + the flatness reading
  pass_cost_vs_context.svg   seconds vs tick: server reqMsPerToken/1000 (and client serveSec if present)

Flatness reading (POD_PLAN_S37 A11 PASS criteria, records-only):
  (a) reqEncPtMs == 0 and no store appends at every tick >= 1   (the store has converged)
  (b) reqBoots constant over ticks >= 1                          (period-1 trajectory)
  (c) median(last 8) within 3.0 % of median(first 8) over ticks >= 1, and no tick outside
      +-10 % of the run median                                   (no trend, no spikes)
  (d) peakVramGB flat over ticks >= 1
The verdict is FLAT only with >= 64 ticks; with fewer it prints "partial (N ticks)" and never
upgrades the claim. Tick 0 is excluded from (b)-(d) by construction (cold start / store build).
Nothing here is a projection: every number is transcribed from the two files named above.
"""
import argparse, csv, json, os, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("--session", default=None)
ap.add_argument("--server-log", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--label", default="", help="free text printed in the md header (binary, box, load state)")
ap.add_argument("--min-ticks", type=int, default=64, help="ticks needed for a FLAT verdict (A11: 64)")
a = ap.parse_args()
if not a.session and not a.server_log:
    sys.exit("need --session and/or --server-log")

# ---- client side ---------------------------------------------------------
client = {}
if a.session:
    p = os.path.join(a.session, "transcript.json")
    if os.path.exists(p):
        for t in json.load(open(p)):
            client[int(t["tick"])] = {"serveSec": t.get("serveSec"), "encSec": t.get("encSec"),
                                      "decSec": t.get("decSec"), "ctlSec": t.get("ctlSec"),
                                      "retired": t.get("phases", []).count("retired")}
    else:
        print(f"warn: no transcript.json in {a.session}", file=sys.stderr)

# ---- server side ---------------------------------------------------------
server, vram_done = {}, {}
if a.server_log and os.path.exists(a.server_log):
    for ln in open(a.server_log, errors="replace"):
        ln = ln.strip()
        if ln.startswith('{"serve":"served"'):
            try: d = json.loads(ln)
            except Exception: continue
            server[int(d["req"])] = d
        elif ln.startswith('{"vramTrace":"serve.done.'):
            try: d = json.loads(ln)
            except Exception: continue
            n = int(d["vramTrace"].split(".")[-1])
            vram_done.setdefault(n, {})[int(d["dev"])] = d
elif a.server_log:
    print(f"warn: server log not found: {a.server_log}", file=sys.stderr)

ticks = sorted(set(client) | set(server))
if not ticks:
    sys.exit("no ticks found")

rows = []
for n in ticks:
    c, s = client.get(n, {}), server.get(n, {})
    vd = vram_done.get(n, {})
    rows.append({
        "tick": n,
        "serveSec_client": c.get("serveSec"),
        "encSec_client": c.get("encSec"), "decSec_client": c.get("decSec"), "ctlSec_client": c.get("ctlSec"),
        "reqMsPerToken": s.get("reqMsPerToken"), "reqEvalMs": s.get("reqEvalMs"), "reqBootMs": s.get("reqBootMs"),
        "reqEncPtMs": s.get("reqEncPtMs"), "reqBoots": s.get("reqBoots"), "failed": s.get("failed"),
        "storeAppends": s.get("storeAppends"), "peakVramGB": s.get("peakVramGB"),
        "usedGB_per_dev": ";".join(f"{dv}:{vd[dv].get('usedGB'):.2f}" for dv in sorted(vd)) if vd else "",
        "poolSlackGB_per_dev": ";".join(f"{dv}:{vd[dv].get('poolSlackGB'):.2f}" for dv in sorted(vd) if vd[dv].get("poolSlackGB") is not None) if vd else "",
        "poolTransientGB_per_dev": ";".join(f"{dv}:{vd[dv].get('poolTransientGB'):.2f}" for dv in sorted(vd) if vd[dv].get("poolTransientGB") is not None) if vd else "",
    })

out = a.out or a.session or os.path.dirname(os.path.abspath(a.server_log))
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "pass_cost_vs_context.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- flatness reading ------------------------------------------------------
warm = [r for r in rows if r["tick"] >= 1]
def num(xs): return [x for x in xs if isinstance(x, (int, float))]
findings, verdict_bits = [], []
n_warm = len(warm)
srv_s = num([r["reqMsPerToken"] for r in warm]);  srv_s = [x / 1000.0 for x in srv_s]
cli_s = num([r["serveSec_client"] for r in warm])
series_name, series = ("server reqMsPerToken", srv_s) if srv_s else ("client serveSec (includes the wire)", cli_s)

enc = num([r["reqEncPtMs"] for r in warm]); app = num([r["storeAppends"] for r in warm])
if enc:
    bad = [r["tick"] for r in warm if isinstance(r["reqEncPtMs"], (int, float)) and r["reqEncPtMs"] > 0]
    ok_a = not bad and (not app or max(app) == 0)
    findings.append(f"(a) store converged: {'yes' if ok_a else 'NO'} -- reqEncPtMs > 0 at ticks {bad if bad else 'none'}"
                    + (f"; storeAppends max {max(app)}" if app else "; storeAppends not recorded"))
    verdict_bits.append(ok_a)
boots = num([r["reqBoots"] for r in warm])
if boots:
    ok_b = min(boots) == max(boots)
    findings.append(f"(b) reqBoots constant: {'yes' if ok_b else 'NO'} -- {sorted(set(boots))}")
    verdict_bits.append(ok_b)
if len(series) >= 2:
    med = statistics.median(series)
    k = min(8, len(series) // 2) or 1
    first, last = statistics.median(series[:k]), statistics.median(series[-k:])
    drift = (last - first) / first * 100.0 if first else float("nan")
    outl = [warm[i]["tick"] for i, x in enumerate(series) if abs(x - med) / med > 0.10]
    ok_c = abs(drift) <= 3.0 and not outl
    findings.append(f"(c) no trend in {series_name}: {'yes' if ok_c else 'NO'} -- median {med:.1f} s; "
                    f"median(first {k}) {first:.1f} s vs median(last {k}) {last:.1f} s = {drift:+.1f} % (bar +-3.0 %); "
                    f"ticks outside +-10 % of the median: {outl if outl else 'none'}; min {min(series):.1f}, max {max(series):.1f}")
    verdict_bits.append(ok_c)
vr = num([r["peakVramGB"] for r in warm])
if vr:
    ok_d = (max(vr) - min(vr)) <= 0.5
    findings.append(f"(d) peakVramGB flat: {'yes' if ok_d else 'NO'} -- min {min(vr):.2f}, max {max(vr):.2f} GB (bar: spread <= 0.5 GB)")
    verdict_bits.append(ok_d)
fails = num([1 for r in warm if r["failed"] is True])
if fails:
    findings.append(f"(e) served requests with failed:true: {len(fails)}"); verdict_bits.append(False)

if n_warm >= a.min_ticks and verdict_bits and all(verdict_bits):
    verdict = f"FLAT over {n_warm} warm ticks (>= {a.min_ticks}): the per-token time does not grow with context on this record"
elif verdict_bits and all(verdict_bits):
    verdict = f"partial ({n_warm} warm ticks < {a.min_ticks}): no trend detected, NOT a flatness claim"
elif verdict_bits:
    verdict = f"NOT FLAT on this record ({n_warm} warm ticks): see the failed criteria"
else:
    verdict = "no per-tick fields to judge"

# ---- markdown --------------------------------------------------------------
L = [f"# Pass cost vs context position\n", f"*{a.label}*\n" if a.label else "",
     f"Sources: `{a.session or '-'}` (transcript.json), `{a.server_log or '-'}` (served lines). "
     f"Client `serveSec` includes the wire both ways; the server-side clock is `reqMsPerToken` (R6: one binary, one load state per record).\n",
     "| tick | client serveSec | server reqMsPerToken (s) | reqBoots | reqEncPtMs | storeAppends | peakVramGB | per-device usedGB at serve.done |",
     "|---|---|---|---|---|---|---|---|"]
def fmt(x, nd=1):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
for r in rows:
    L.append(f"| {r['tick']} | {fmt(r['serveSec_client'])} | {fmt((r['reqMsPerToken'] or 0)/1000.0) if r['reqMsPerToken'] is not None else ''} | "
             f"{fmt(r['reqBoots'])} | {fmt(r['reqEncPtMs'])} | {fmt(r['storeAppends'])} | {fmt(r['peakVramGB'], 2)} | {r['usedGB_per_dev']} |")
L.append("\n## Flatness reading (A11 criteria)\n")
L += [f"- {x}" for x in findings]
L.append(f"\n**Verdict: {verdict}.**\n")
open(os.path.join(out, "pass_cost_vs_context.md"), "w").write("\n".join(L) + "\n")

# ---- svg (no matplotlib dependency) ------------------------------------------
W, H, ml, mr, mt, mb = 900, 420, 70, 20, 30, 50
pts_srv = [(r["tick"], r["reqMsPerToken"] / 1000.0) for r in rows if isinstance(r["reqMsPerToken"], (int, float))]
pts_cli = [(r["tick"], r["serveSec_client"]) for r in rows if isinstance(r["serveSec_client"], (int, float))]
allp = pts_srv + pts_cli
xs = [p[0] for p in allp]; ys = [p[1] for p in allp]
x0, x1 = min(xs), max(max(xs), min(xs) + 1); y1 = max(ys) * 1.08 or 1.0
def X(x): return ml + (x - x0) / (x1 - x0) * (W - ml - mr)
def Y(y): return H - mb - (y / y1) * (H - mt - mb)
def poly(pts, col, name):
    d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
    circ = "".join(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{col}"><title>{name} tick {x}: {y:.1f} s</title></circle>' for x, y in pts)
    return f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="1.5"/>{circ}'
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12">',
       f'<rect width="{W}" height="{H}" fill="#fff"/>',
       f'<line x1="{ml}" y1="{Y(0):.1f}" x2="{W-mr}" y2="{Y(0):.1f}" stroke="#888"/>',
       f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{Y(0):.1f}" stroke="#888"/>']
nt = 5
for i in range(nt + 1):
    yv = y1 * i / nt
    svg.append(f'<line x1="{ml-4}" y1="{Y(yv):.1f}" x2="{W-mr}" y2="{Y(yv):.1f}" stroke="#eee"/>'
               f'<text x="{ml-8}" y="{Y(yv)+4:.1f}" text-anchor="end">{yv:.0f}</text>')
step = max(1, (x1 - x0) // 16 or 1)
for xv in range(int(x0), int(x1) + 1, int(step)):
    svg.append(f'<text x="{X(xv):.1f}" y="{H-mb+16}" text-anchor="middle">{xv}</text>')
svg.append(f'<text x="{(ml+W-mr)/2:.0f}" y="{H-8}" text-anchor="middle">context position (tick)</text>')
svg.append(f'<text transform="translate(16,{(mt+H-mb)/2:.0f}) rotate(-90)" text-anchor="middle">seconds per token</text>')
if pts_srv: svg.append(poly(pts_srv, "#1f5fbf", "server reqMsPerToken"))
if pts_cli: svg.append(poly(pts_cli, "#c0392b", "client serveSec"))
leg = []
if pts_srv: leg.append(("#1f5fbf", "server reqMsPerToken / 1000"))
if pts_cli: leg.append(("#c0392b", "client serveSec (includes the wire)"))
for i, (col, name) in enumerate(leg):
    svg.append(f'<rect x="{ml+10}" y="{mt+4+i*18}" width="12" height="12" fill="{col}"/><text x="{ml+28}" y="{mt+14+i*18}">{name}</text>')
svg.append(f'<text x="{W-mr}" y="{mt-10}" text-anchor="end" fill="#555">{verdict}</text>')
svg.append("</svg>")
open(os.path.join(out, "pass_cost_vs_context.svg"), "w").write("\n".join(svg))
print(json.dumps({"passCostVsContext": True, "ticks": len(rows), "warmTicks": n_warm, "out": out, "verdict": verdict}))
for x in findings: print("  " + x)
