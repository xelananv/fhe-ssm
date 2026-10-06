#!/usr/bin/env python3
"""Tidy CSV tables for plotting, transcribed from the MIRRORED records of the long autoregressive run (rental of 2026-09-19). Nothing is computed beyond
per-tick aggregates of the per-lane rows (median / 90th percentile / max of relErrRms, agreement counts) and the assignment of a bootstrap to the tick in
flight (a bootRecord line belongs to tick t if it precedes the t-th `served` line; tick 0 is the cold tick).
usage: make_plot_data.py [--arm LONG] [--root pod_pull] [--out plots_data]"""
import argparse, csv, gzip, json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="LONG"); ap.add_argument("--root", default=os.path.join(HERE, "pod_pull")); ap.add_argument("--out", default=os.path.join(HERE, "plots_data"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
D = os.path.join(a.root, "s37", "serve_" + a.arm); served, vram, pt, rec = [], {}, [], []
for l in open(os.path.join(D, "logs", "server.jsonl"), errors="replace"):
    l = l.strip()
    if not l.startswith("{"): continue
    try: r = json.loads(l)
    except Exception: continue
    if r.get("serve") == "served": served.append(r)
    elif str(r.get("vramTrace", "")).startswith("serve.done."): vram[(int(r["vramTrace"].split(".")[-1]), int(r.get("dev", 0)))] = r
    elif r.get("ptCacheTick"): pt.append(r)
    elif r.get("bootRecord") is True: r["tick"] = len(served); rec.append(r)
rows = json.load(open(os.path.join(D, "fidelity.json"))); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
by = {}
for r in rows: by.setdefault(r["tick"], []).append(r)
def pct(v, q): v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
# ---- per tick
cols = ["tick", "wall_layer_loop_s", "driver_request_to_reply_s", "boots", "eval_ms", "boot_ms", "norm_boot_ms", "timer_sum_ms_NOT_A_WALL_CLOCK", "recorder_ms", "enc_pt_ms",
        "pt_cache_calls", "pt_cache_misses", "gpu0_pool_used_gb", "gpu0_pool_reserved_gb", "gpu0_device_used_gb", "gpu1_pool_used_gb", "gpu1_pool_reserved_gb", "gpu1_device_used_gb", "lanes", "generating_lanes", "raw_top1_agree", "pick_agree",
        "relerr_rms_median", "relerr_rms_p90", "relerr_rms_max", "bootstraps_recorded", "max_overflow_in_tick", "bootstraps_over_512_in_tick"]
with open(os.path.join(a.out, f"{a.arm}_per_tick.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for t, d in enumerate(served):
        v = by.get(t, []); e = [r["relErrRms"] for r in v]; rb = [r for r in rec if r["tick"] == t and r.get("rmsI", 0) > 1]; v0 = vram.get((t, 0), {}); v1 = vram.get((t, 1), {}); p = pt[t] if t < len(pt) else {}
        w.writerow([t, d["reqLayerLoopMs"] / 1000, v[0].get("serveSec") if v else "", d["reqBoots"], d["reqEvalMs"], d["reqBootMs"], d.get("reqNormBootMs"), d["reqMsPerToken"], d.get("reqBootRecMs"), d.get("reqEncPtMs"),
                    p.get("ptCalls", ""), p.get("ptCacheMisses", ""), v0.get("poolUsedGB", ""), v0.get("poolReservedGB", ""), v0.get("usedGB", ""), v1.get("poolUsedGB", ""), v1.get("poolReservedGB", ""), v1.get("usedGB", ""), len(v), sum(1 for r in v if r.get("generating")),
                    sum(1 for r in v if r["top1"]), sum(1 for r in v if r.get("pickAgree")), (st.median(e) if e else ""), (pct(e, 0.9) if e else ""), (max(e) if e else ""),
                    len(rb), (max(r["maxI"] for r in rb) if rb else ""), sum(1 for r in rb if r["maxI"] > 512)])
# ---- per lane per tick
keys = ["tick", "lane", "generating", "fed", "fedSrc", "encArgmax", "refArgmax", "top1", "encPick", "refPick", "pickAgree", "refTop1Margin", "relErrRms", "serveSec"]
with open(os.path.join(a.out, f"{a.arm}_per_lane_tick.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(keys)
    for r in sorted(rows, key=lambda r: (r["tick"], r["lane"])): w.writerow([r.get(k, "") for k in keys])
# ---- per bootstrap (gzip)
with gzip.open(os.path.join(a.out, f"{a.arm}_per_bootstrap.csv.gz"), "wt", newline="") as f:
    w = csv.writer(f); w.writerow(["tick", "bootstrap_n", "stage", "max_overflow", "rms_overflow", "coeffs_over_512", "max_frac_message_over_q0", "transparent_or_zero_input", "recorder_ms"])
    for r in rec: w.writerow([r["tick"], r["n"], r.get("stage"), r["maxI"], r["rmsI"], r.get("coeffsOver512"), r.get("maxFrac"), int(not r.get("rmsI", 0) > 1), r.get("ms")])
# ---- host memory / cards, one line per minute
ms = os.path.join(a.root, "mem_sampler.log")
with open(os.path.join(a.out, "host_memory_per_minute.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["utc", "pid", "serve_dir", "served", "vmrss_kb", "vmhwm_kb", "rssanon_kb", "mappings", "gpu0_used_mib", "gpu1_used_mib"])
    for l in open(ms, errors="replace"):
        x = [c.strip() for c in l.split("|")]
        if len(x) < 10 or x[0].startswith("#"): continue
        g = (x[9].split() + ["", ""])[:2]; w.writerow([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[8], g[0], g[1]])
print(f"{a.arm}: {len(served)} ticks, {len(rows)} lane-ticks, {len(rec)} bootstraps -> {a.out}")
