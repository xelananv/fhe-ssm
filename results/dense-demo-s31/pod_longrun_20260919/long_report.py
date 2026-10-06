#!/usr/bin/env python3
"""Records-only report of the long AUTOREGRESSIVE pod-side run (rental of 2026-09-19) from its MIRRORED directory.
Every number is transcribed or computed from the files named in the header of the output: pod_pull/s37/serve_<ARM>/logs/server.jsonl
(served lines, vramTrace serve.done.N lines, ptCacheTick lines, bootRecord lines, the library's table lines), .../fidelity.json (one row per
lane per tick), pod_pull/mem_sampler.log (host memory, one line per minute). rms errors only (R3); times are the WALL clock of the layer loop
(reqLayerLoopMs) and the driver's request->reply serveSec -- reqMsPerToken is a timer sum and is printed only as such.
usage: long_report.py [--arm LONG] [--root pod_pull] [--prompts prompt_select/longrun_prompts64_v4] [--out LONG_REPORT.md]"""
import argparse, json, math, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="LONG"); ap.add_argument("--root", default=os.path.join(HERE, "pod_pull"))
ap.add_argument("--prompts", default=os.path.join(HERE, "prompt_select", "longrun_prompts64_v4")); ap.add_argument("--out", default=None)
ap.add_argument("--text", type=int, default=1, help="1 = decode the generated text per lane (needs the tokenizer)")
a = ap.parse_args()
D = os.path.join(a.root, "s37", "serve_" + a.arm); SJ = os.path.join(D, "logs", "server.jsonl"); FJ = os.path.join(D, "fidelity.json")
served, vram, pt, rec, tables, other = [], {}, [], [], [], []
for l in open(SJ, errors="replace"):
    l = l.strip()
    if not l.startswith("{"): continue
    try: r = json.loads(l)
    except Exception: continue
    if r.get("serve") == "served": served.append(r)
    elif str(r.get("vramTrace", "")).startswith("serve.done."): vram[(int(r["vramTrace"].split(".")[-1]), int(r.get("dev", 0)))] = r   # per DEVICE (a dict keyed on the request alone kept only the last card: 2026-09-19 correction)
    elif r.get("ptCacheTick"): pt.append(r)
    elif r.get("bootRecord") is True: rec.append(r)
    elif "fideslibBootTable" in r or "openfheBootScaleEnc" in r or r.get("bootRecordArmed"): tables.append(r)
    elif "fatal" in r or r.get("serve") == "exit": other.append(r)
rows = json.load(open(FJ)); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
by = {}
for r in rows: by.setdefault(r["tick"], []).append(r)
O = []; P = O.append
def med(x): return st.median(x) if x else float("nan")
def slope(y):
    n = len(y); xs = list(range(n)); mx, my = sum(xs) / n, sum(y) / n
    return sum((x - mx) * (v - my) for x, v in zip(xs, y)) / max(sum((x - mx) ** 2 for x in xs), 1e-30)
P(f"# Long autoregressive run, arm {a.arm}: records-only report"); P("")
P(f"sources: `{os.path.relpath(SJ, REPO)}`, `{os.path.relpath(FJ, REPO)}`, `{os.path.relpath(os.path.join(a.root, 'mem_sampler.log'), REPO)}`"); P("")
for t in tables: P("library / instrument line: `" + json.dumps(t)[:300] + "`")
P("")
# ---------------------------------------------------------------- 1. cost per tick
warm = served[1:]; w = [d["reqLayerLoopMs"] / 1000 for d in warm]; n = len(served)
P(f"## 1. Cost per tick ({n} served ticks; tick 0 is the cold tick)"); P("")
if w:
    f8, l8 = w[:8], w[-8:]
    P(f"- WALL clock of the layer loop (`reqLayerLoopMs`), warm ticks 1..{n - 1}: median **{med(w):.1f} s** (min {min(w):.1f}, max {max(w):.1f}); first 8 {med(f8):.1f} s vs last 8 {med(l8):.1f} s = {100 * (med(l8) / med(f8) - 1):+.2f} % (bar +-3 %); ticks outside +-10 % of the median: {sum(1 for v in w if abs(v / med(w) - 1) > 0.10)}; least-squares slope {1000 * slope(w):+.2f} ms per tick")
    ss = [by[t][0].get("serveSec") for t in sorted(by) if t >= 1 and by[t][0].get("serveSec") is not None]
    if ss: P(f"- request -> reply as the driver sees it (`serveSec`), warm: median {med(ss):.1f} s (min {min(ss):.1f}, max {max(ss):.1f})")
    P(f"- `reqBoots`: tick 0 {served[0]['reqBoots']}, warm ticks {sorted({d['reqBoots'] for d in warm})}; `reqEncPtMs` summed over warm ticks: {sum(d.get('reqEncPtMs', 0) for d in warm)}; X2 cache misses after tick 1: {sum(p.get('ptCacheMisses', 0) for p in pt[2:])} (tick 0: {pt[0].get('ptCacheMisses') if pt else None}, tick 1: {pt[1].get('ptCacheMisses') if len(pt) > 1 else None})")
    sm = [d["reqMsPerToken"] / 1000 for d in warm]; P(f"- for the record only, the timer SUM `reqMsPerToken`: median {med(sm):.1f} s (not a wall clock; with the recorder every bootstrap is synchronous, so its boot term is the real time: `reqBootMs` median {med([d['reqBootMs'] / 1000 for d in warm]):.1f} s); recorder's own time `reqBootRecMs` median {med([d.get('reqBootRecMs', 0) / 1000 for d in warm]):.2f} s per tick")
P("")
# ---------------------------------------------------------------- 2. memory
P("## 2. Memory"); P("")
ks = [k for k in sorted(vram) if k[0] >= 1]
if ks:
    for dev in sorted({k[1] for k in ks}):
        kd = [k for k in ks if k[1] == dev]; pu = [vram[k].get("poolUsedGB") for k in kd]; pr = [vram[k].get("poolReservedGB") for k in kd]; ug = [vram[k].get("usedGB") for k in kd]
        P(f"- device {dev}, pool after request 1..{kd[-1][0]} (`vramTrace serve.done.N`): distinct `poolUsedGB` values {sorted(set(pu))}, `poolReservedGB` {sorted(set(pr))}, `usedGB` {sorted(set(ug))}")
    P("  (last session on the unpatched library: pool-used +3,563,424 B per request per card)")
ms = os.path.join(a.root, "mem_sampler.log")
if os.path.exists(ms):
    pid_rows = []; maps_seen = {}
    for l in open(ms, errors="replace"):
        f = [x.strip() for x in l.split("|")]
        if len(f) < 10 or f[0].startswith("#") or ("serve_" + a.arm + "/") not in f[2] + "/": continue
        try: pid_rows.append((f[1], int(f[3]), int(f[4]), int(f[6]), f[9])); maps_seen.setdefault(f[1], set()).add(f[8]) if int(f[3]) >= 1 else None
        except Exception: pass
    if pid_rows:
        pid = pid_rows[-1][0]; pr_ = [x for x in pid_rows if x[0] == pid]; first = {}; last = {}
        for _, nn, rss, anon, g in pr_:
            first.setdefault(nn, (rss, anon, g)); last[nn] = (rss, anon, g)
        ns = sorted(k for k in first if k >= 2)
        if len(ns) >= 2:
            # The host record is a STEP function, so it is reported as its steps and its final plateau -- NOT as an average per request (question,
            # 2026-09-19: an earlier version divided a one-off step by the request count and printed "+13.1 kB per request", which reads as steady growth).
            warm_rows = [x for x in pr_ if x[1] >= 1]; steps = []; prev = None
            for _, nn, rss, anon, g in warm_rows:
                if prev is not None and rss != prev[1]: steps.append((prev[0], nn, rss - prev[1]))
                prev = (nn, rss)
            r0, r1 = warm_rows[0][2], warm_rows[-1][2]; plateau_from = steps[-1][1] if steps else warm_rows[0][1]
            flat_samples = sum(1 for x in warm_rows if x[2] == r1 and x[1] >= plateau_from)
            P(f"- host, serving process pid {pid} (one sample per minute): VmRSS {r0:,} kB after request 1 -> {r1:,} kB at {warm_rows[-1][1]} served = {(r1 - r0):+,} kB in {len(steps)} discrete step(s): "
              + "; ".join(f"{d:+,} kB in the minute in which `served` went {a0} -> {b0}" for a0, b0, d in steps)
              + f"; **unchanged in all {flat_samples} samples since ({warm_rows[-1][1] - plateau_from} requests)**; mappings count values {sorted(maps_seen.get(pid, []))}; total change = {100 * (r1 - r0) / r0:.5f} % of the resident set (last session on the unpatched library: +20.7 MB per request, every request); cards (nvidia-smi used MiB) first `{first[ns[0]][2]}` last `{last[ns[-1]][2]}`")
P("")
# ---------------------------------------------------------------- 3. fidelity
P("## 3. Fidelity against the plaintext model on the SAME fed sequence (per lane per tick)"); P("")
T = sorted(by); tot = len(rows); dec = [t for t in T if any(r.get("decFailed") for r in by[t])]
t1 = sum(1 for r in rows if r["top1"]); pk = sum(1 for r in rows if r.get("pickAgree")); haspick = any("pickAgree" in r for r in rows)
gen = [r for r in rows if r.get("generating")]; rd = [r for r in rows if not r.get("generating")]
P(f"- ticks compared {len(T)}, lane-ticks {tot}; decode-failed ticks: {dec}; fallback feeds: {sum(1 for r in rows if r.get('fallbackFeed'))}")
P(f"- raw argmax agreement (circuit fidelity): {t1}/{tot} = {100 * t1 / tot:.3f} %" + (f"; decode-rule pick agreement (what is fed back): {pk}/{tot} = {100 * pk / tot:.3f} %" if haspick else ""))
P(f"- while READING (teacher-forced prompt ticks): top-1 {sum(1 for r in rd if r['top1'])}/{len(rd)}; while GENERATING: top-1 {sum(1 for r in gen if r['top1'])}/{len(gen)}" + (f", picks {sum(1 for r in gen if r.get('pickAgree'))}/{len(gen)}" if haspick else ""))
h = len(T) // 2; e1 = [r["relErrRms"] for t in T[:h] for r in by[t]]; e2 = [r["relErrRms"] for t in T[h:] for r in by[t]]
P(f"- relErrRms per lane-tick: median {med([r['relErrRms'] for r in rows]):.6f}; first half of the ticks {med(e1):.6f}, second half {med(e2):.6f}; per-tick medians min {min(med([r['relErrRms'] for r in by[t]]) for t in T):.6f} max {max(med([r['relErrRms'] for r in by[t]]) for t in T):.6f}; worst single lane-tick {max(r['relErrRms'] for r in rows):.5f}")
mm = [r for r in rows if not r["top1"] or (haspick and not r.get("pickAgree"))]
if mm:
    P(""); P("| tick | lane | raw FHE argmax | raw plaintext argmax | refTop1Margin | FHE pick | plaintext pick | relErrRms | generating |"); P("|---|---|---|---|---|---|---|---|---|")
    for r in mm[:80]: P(f"| {r['tick']} | {r['lane']} | {r['encArgmax']} | {r['refArgmax']} | {r.get('refTop1Margin')} | {r.get('encPick', '')} | {r.get('refPick', '')} | {r['relErrRms']} | {bool(r.get('generating'))} |")
P("")
# ---------------------------------------------------------------- 4. recorder
P("## 4. Bootstrap flight recorder (pod TEST key; max overflow of every bootstrap's mod-raise input)"); P("")
if rec:
    mi = [r["maxI"] for r in rec if r.get("rmsI", 0) > 1]; zero = len(rec) - len(mi)
    P(f"- bootstraps recorded {len(rec)} ({zero} on all-zero cold-start carries); rmsI mean {sum(r['rmsI'] for r in rec if r.get('rmsI', 0) > 1) / max(len(mi), 1):.3f}; maxFrac max {max(r.get('maxFrac', 0) for r in rec):.4f}")
    # Model: x_j = ([c0] + [c1]*s)_j / q0 is a sum of h + 1 uniform(-1/2, 1/2) terms (h = the key's Hamming weight), Gaussian to < 0.1 % at 6 sigma,
    # sigma = sqrt((h + 1) / 12) = the recorder's own `sigmaIPred`; the recorded overflow is the INTEGER I_j = round(x_j), so |I_j| > B <=> |x_j| >= B + 1/2
    # (continuity correction; without it and with sigma rounded to 85.4 the expectation is ~4 % too high at B = 400 and ~5 % at B = 512).
    sig = next((t["sigmaIPred"] for t in tables if t.get("bootRecordArmed") and t.get("sigmaIPred")), 85.4); N = 131072
    P(""); P(f"| threshold B | bootstraps with max overflow > B (observed) | expected: Gaussian x, sigma {sig:.2f} (the recorder's `sigmaIPred`), integer I = round(x) so P(abs(I) > B) = erfc((B + 0.5) / (sigma sqrt 2)), N = 2^17 independent coefficients | observed / expected |"); P("|---|---|---|---|")
    for B in (400, 430, 460, 480, 500, 512, 519, 530, 537, 600, 768):
        p = 1 - (1 - math.erfc((B + 0.5) / sig / math.sqrt(2))) ** N; ob = sum(1 for v in mi if v > B); ex = p * len(mi)
        P(f"| {B} | {ob} | {ex:.2f} | {(ob / ex):.2f} |" if ex >= 0.5 else f"| {B} | {ob} | {ex:.2f} | |")
    # independent bins, so that one deficit is not counted at every threshold below it
    edges = (400, 430, 460, 480, 500, 512, 10 ** 9); chi = 0.0; cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        plo = 1 - (1 - math.erfc((lo + 0.5) / sig / math.sqrt(2))) ** N; phi = 0.0 if hi > 10 ** 6 else 1 - (1 - math.erfc((hi + 0.5) / sig / math.sqrt(2))) ** N
        ob = sum(1 for v in mi if lo < v <= hi); ex = (plo - phi) * len(mi); chi += (ob - ex) ** 2 / max(ex, 1e-9); cells.append(f"({lo}, {hi if hi < 10 ** 6 else 'inf'}]: {ob} / {ex:.1f}")
    P(""); P(f"- the same in disjoint bins of the per-bootstrap maximum (observed / expected): " + "; ".join(cells) + f"; Pearson chi-square {chi:.1f} on {len(cells)} bins (the bins' expectations come from the per-bootstrap maximum's distribution under the model; no parameter is fitted)")
    big = sorted([r for r in rec if r["maxI"] > 512], key=lambda r: -r["maxI"])
    P(""); P(f"- bootstraps beyond the stock table's bound 512: {len(big)}; the run's maximum {max(mi)}")
    for r in big[:20]: P(f"  - bootstrap n = {r['n']}: maxI {r['maxI']} (coeffsOver512 {r.get('coeffsOver512')}, stage `{r.get('stage')}`, rmsI {r['rmsI']})")
P("")
# ---------------------------------------------------------------- 5. text
if a.text:
    sys.path.insert(0, os.path.join(REPO, "ml-eval")); import fhe_client as FC; tok = FC.get_tokenizer()
    prompts = [l.rstrip("\n") for l in open(a.prompts + ".txt") if l.strip()]
    try: cls = json.load(open(a.prompts + ".txt.classes.json"))
    except Exception: cls = {}
    bl = {}
    for r in rows: bl.setdefault(r["lane"], []).append(r)
    P("## 5. What the encrypted session generated (decrypted with the pod test key; **bold** = the fed token differs from the plaintext model's pick)"); P("")
    P("| lane | class / source | prompt tokens | generated | pick = plaintext | prompt | generated text |"); P("|---|---|---|---|---|---|---|")
    for lane in sorted(bl):
        seq = sorted(bl[lane], key=lambda r: r["tick"]); g = [r for r in seq if r.get("generating")]
        txt = "".join(("**" + tok.decode([r.get("encPick", r["encArgmax"])]) + "**") if not r.get("pickAgree", r["top1"]) else tok.decode([r.get("encPick", r["encArgmax"])]) for r in g if r.get("encPick", r["encArgmax"]) >= 0)
        pr = prompts[lane] if lane < len(prompts) else ""
        P(f"| {lane} | {cls.get(str(lane), '')} | {len(tok.encode(pr))} | {len(g)} | {sum(1 for r in g if r.get('pickAgree', r['top1']))}/{len(g)} | {pr[:90].replace('|', '/')}{'…' if len(pr) > 90 else ''} | {txt.replace('|', '/').replace(chr(10), ' ⏎ ')} |")
out = a.out or os.path.join(HERE, f"{a.arm}_REPORT.md"); open(out, "w").write("\n".join(O) + "\n"); print("\n".join(O[:40])); print("... ->", out)
