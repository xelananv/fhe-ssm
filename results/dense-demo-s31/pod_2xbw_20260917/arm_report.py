#!/usr/bin/env python3
"""Records-only report of one pod-side served arm (A11, X2, ...) from its mirrored directory
(pod_pull/s37/serve_<ARM>/{logs/server.jsonl,fidelity.json}); optional --compare <ARM2> puts the warm ticks of two arms
side by side (only meaningful for arms served by the SAME binary on the SAME box in the SAME load state: the R6 rule (never compare a timing across binaries or machine load states)).
Every number printed is transcribed or computed from the two files named in the header; rms errors only (R3).
usage: arm_report.py A11 [--compare X2] [--root pod_pull/s37] [--prompts <file>]"""
import argparse, json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ap = argparse.ArgumentParser(); ap.add_argument("arm"); ap.add_argument("--compare", default=None)
ap.add_argument("--root", default=os.path.join(HERE, "pod_pull", "s37")); ap.add_argument("--tokens", action="store_true", help="decode mismatch tokens (needs the tokenizer)")
ap.add_argument("--prompts", default=os.path.join(REPO, "results/dense-demo-s31/sessions/a11_prompts64_long.txt"))
a = ap.parse_args()
def load(arm):
    d = os.path.join(a.root, "serve_" + arm); served, pt, other = [], [], []
    for l in open(os.path.join(d, "logs", "server.jsonl"), errors="replace"):
        l = l.strip()
        if not l.startswith("{"): continue
        try: r = json.loads(l)
        except Exception: continue
        if r.get("serve") == "served": served.append(r)
        elif r.get("ptCacheTick"): pt.append(r)
        elif "fatal" in r or r.get("serve") == "exit" or r.get("summary") is True: other.append(r)
    rows = []
    fj = os.path.join(d, "fidelity.json")
    if os.path.exists(fj):
        dd = json.load(open(fj)); rows = dd.get("rows", dd) if isinstance(dd, dict) else dd
    return d, served, pt, other, rows
def med(x): return statistics.median(x) if x else float("nan")
def report(arm):
    d, served, pt, other, rows = load(arm)
    by = {}
    for r in rows: by.setdefault(r["tick"], []).append(r)
    print(f"## arm {arm}: {len(served)} served ticks, {len(by)} decoded+compared ticks\n")
    print(f"sources: `{os.path.relpath(d, REPO)}/logs/server.jsonl` (served lines), `{os.path.relpath(d, REPO)}/fidelity.json` (per-lane rows)\n")
    print("| tick | reqMsPerToken s | reqBoots | reqEncPtMs | hostPtEncode n | hostPtEncode s | reqBootMs s | peakVramGB | top-1 | decFailed | relErrRms median | relErrRms worst | dup lanes agree |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(served):
        b = by.get(i, []); e = sorted(r["relErrRms"] for r in b)
        print(f"| {i} | {s['reqMsPerToken']/1000:.1f} | {s.get('reqBoots')} | {s.get('reqEncPtMs')} | {s.get('reqHostPtEncodeCount','')} | "
              f"{(s.get('reqHostPtEncodeMs') or 0)/1000:.1f} | {(s.get('reqBootMs') or 0)/1000:.1f} | {s.get('peakVramGB')} | "
              + (f"{sum(1 for r in b if r['top1'])}/{len(b)} | {sum(1 for r in b if r.get('decFailed'))} | {e[len(e)//2]:.5f} | {e[-1]:.5f} | {b[0]['encArgmax'] == b[-1]['encArgmax']} |" if b else " | | | | |"))
    w = [s["reqMsPerToken"] / 1000 for s in served[1:]]
    if w:
        n = len(w); half = n // 2; x = list(range(1, n + 1)); mx = sum(x) / n; my = sum(w) / n
        slope = sum((xi - mx) * (yi - my) for xi, yi in zip(x, w)) / max(sum((xi - mx) ** 2 for xi in x), 1e-30)
        print(f"\nwarm ticks 1..{n}: median {med(w):.1f} s, min {min(w):.1f}, max {max(w):.1f}, mean {my:.2f}; first half median {med(w[:half]):.1f} vs second half {med(w[half:]):.1f} "
              f"({(med(w[half:]) / med(w[:half]) - 1) * 100:+.2f} %); least-squares slope {slope * 1000:+.1f} ms per tick of context; "
              f"per lane-token at 64 lanes: {med(w) / 64:.2f} s")
        print(f"reqBoots warm: {sorted(set(s.get('reqBoots') for s in served[1:]))}; reqEncPtMs warm: {sorted(set(s.get('reqEncPtMs') for s in served[1:]))}; failed requests: {sum(1 for s in served if s.get('failed'))}; "
              f"peakVramGB: {min(s.get('peakVramGB') or 0 for s in served)}–{max(s.get('peakVramGB') or 0 for s in served)}")
    if rows:
        tot = len(rows); ok = sum(1 for r in rows if r["top1"]); df = sorted({r["tick"] for r in rows if r.get("decFailed")})
        rel = [r["relErrRms"] for r in rows if not r.get("decFailed")]
        dec = [r for r in rows if not r.get("decFailed")]; okd = sum(1 for r in dec if r["top1"])
        print(f"fidelity: decode-failed ticks {df or 'none'} ({tot - len(dec)} lane-ticks, every lane of those ticks); on the {len(dec)} decoded lane-ticks top-1 {okd}/{len(dec)} ({okd / max(len(dec), 1) * 100:.2f} %), "
              f"relErrRms median {med(rel):.5f}, worst {max(rel):.5f}; over ALL {tot} lane-ticks (failed ticks counted as misses) top-1 {ok}/{tot} ({ok / tot * 100:.2f} %)")
        mm = [r for r in rows if not r["top1"] and not r.get("decFailed")]
        if mm:
            tok = None
            if a.tokens:
                sys.path.insert(0, os.path.join(REPO, "ml-eval")); import fhe_client as FC; tok = FC.get_tokenizer()
            print("\ntop-1 disagreements (the plaintext model's own top-1 margin in logits beside each: a small margin = a near-tie):\n")
            print("| tick | lane | FHE argmax | plaintext argmax | refTop1Margin | relErrRms |"); print("|---|---|---|---|---|---|")
            for r in mm:
                f = lambda i: (repr(tok.decode([i])) + f" ({i})") if tok and i >= 0 else str(i)
                print(f"| {r['tick']} | {r['lane']} | {f(r['encArgmax'])} | {f(r['refArgmax'])} | {r.get('refTop1Margin')} | {r['relErrRms']} |")
        # error trend over context
        ticks = sorted(by); h = len(ticks) // 2
        if h >= 2:
            m1 = med([med([r["relErrRms"] for r in by[t]]) for t in ticks[:h]]); m2 = med([med([r["relErrRms"] for r in by[t]]) for t in ticks[h:]])
            print(f"\nper-tick median relErrRms: first half of the ticks {m1:.5f}, second half {m2:.5f}")
    for p in pt: print("ptCacheTick:", json.dumps({k: p.get(k) for k in ("pass", "ptCalls", "ptCacheHits", "ptCacheMisses", "ptCacheDenseFallback", "ptCacheSkipped", "ptCacheEntries", "ptCacheLimbs", "ptCacheResidentGB")}))
    for o in other:
        if "fatal" in o: print("FATAL line:", json.dumps(o)[:400])
        elif o.get("summary") is True: print("summary:", json.dumps({k: o.get(k) for k in ("ptCache", "ptCacheVerify", "ptCacheMaxGb", "ptCalls", "ptCacheEntries", "ptCacheHits", "ptCacheMisses", "ptCacheMissMs", "ptCacheDenseFallback", "ptCacheDenseMiss", "ptCacheSkipped", "ptCacheLimbs", "ptCacheResidentGB", "ptCacheVerified", "ptCacheVerifyFail", "hostPtEncodeMs", "hostPtEncodeCount", "boots", "peakVramGB") if k in o}))
        else: print("server exit:", json.dumps(o)[:200])
    return served, by
s1, b1 = report(a.arm)
if a.compare:
    print(); s2, b2 = report(a.compare)
    w1 = [s["reqMsPerToken"] / 1000 for s in s1[1:]]; w2 = [s["reqMsPerToken"] / 1000 for s in s2[1:]]
    if w1 and w2:
        k = ("reqMsPerToken", "reqEvalMs", "reqBootMs", "reqHostPtEncodeMs", "reqHostPtEncodeCount", "reqLayerLoopMs", "reqLayerLoopUntimedMs", "reqBoots", "peakVramGB")
        print(f"\n## warm-tick comparison {a.arm} vs {a.compare} (same binary / box / load state required: R6)\n")
        print(f"| field (median over warm ticks) | {a.arm} (n={len(w1)}) | {a.compare} (n={len(w2)}) | delta |"); print("|---|---|---|---|")
        for f in k:
            v1 = med([s.get(f) or 0 for s in s1[1:]]); v2 = med([s.get(f) or 0 for s in s2[1:]])
            print(f"| {f} | {v1:g} | {v2:g} | {v2 - v1:+g} ({((v2 / v1 - 1) * 100) if v1 else float('nan'):+.1f} %) |")
        print(f"\n{a.arm} warm range {min(w1):.1f}–{max(w1):.1f} s; {a.compare} warm range {min(w2):.1f}–{max(w2):.1f} s")
        # same inputs, same keys: identical argmax expected tick by tick where both decoded
        same = diff = 0
        for t in sorted(set(b1) & set(b2)):
            m1 = {r["lane"]: r["encArgmax"] for r in b1[t]}; m2 = {r["lane"]: r["encArgmax"] for r in b2[t]}
            for l in m1:
                if l in m2: same += (m1[l] == m2[l]); diff += (m1[l] != m2[l])
        print(f"FHE argmax {a.arm} vs {a.compare} on the ticks both decoded: {same} equal, {diff} different")
