#!/usr/bin/env python3
"""Reproducible report tables for TICK_SCALING_BOUNDS_20260903.md.

Sub-commands (all derived from tick_level_trace.py; none read results/):
  probe   -- remaining levels at branch births, demo geometry, both models, and at lam = 68
  grid    -- boots per tick over (lam, F, margin, I, h, model)
  sweep   -- boots per tick vs lam at the demo geometry, against the ideal and the depth floor
  sites   -- where the boots fire (warm mean per tick), demo configuration
  sanity  -- the cell matching the record used in the report's sanity section (depth 41, h = 1)
  variants -- architecture levers priced on the replay (Tier B inputs)
  schedule -- S3.3 (2026-09-03): every table of SCHEDULE_DESIGN_20260903.md -- the lever
              matrix (boots, Lambda, rotations, pool builds) at lam 22/24 x eager/lazy,
              the boot-site histograms and operand-depth probes, the 1000-tick
              periodicity/store-key study (I1, I6), the lam sweep, the uniform-I grid and
              the Tier B variants under parent-first
"""
import json
import math
import sys
from collections import Counter

import tick_level_trace as T
import tick_bounds_tables as B

SEEDS = "../../ml-eval/artifacts/pbd430a_seeds.json"


class A:
    pass


def args(depth=41, lam=22, F=3, margin=4, iters=3, seeds=SEEDS, lanes_block=True, lazy=False, ticks=30,
         d=1024, dff=4096, L=24, shift=True, parallel=False,
         schedule="stock", pf_entry="loop", no_pf_tail=False, share=False, late=False, nd2=False, pool=2):
    a = A()
    a.depth = depth; a.extra_depth = 0; a.dboot = 0; a.lam = lam; a.boot_floor = F; a.margin = margin
    a.dpad = d; a.d = d; a.dff = dff; a.layers = L; a.iters = iters; a.seeds = seeds
    a.lanes_block = lanes_block; a.shift_mix = shift; a.lazy = lazy; a.final_boot = False
    a.ticks = ticks; a.json = None; a.sites = False; a.parallel_block = parallel
    # S3.3 schedule levers (all default OFF = the stock replay)
    a.schedule = schedule; a.pf_entry = pf_entry; a.no_pf_tail = no_pf_tail; a.share_babies = share
    a.late_margin = late; a.newton_depth2 = nd2; a.pool_levels = pool
    a.boot_deg2 = False; a.trace_levels = False
    return a


def variants():
    """Architecture levers priced on the replay (demo configuration, eager):
    boots per tick, weight multiplies, rotations, limb-weighted multiplies."""
    rows = [
        ("baseline: bundle seeds, k=4, sequential block, shift-mix", dict()),
        ("Newton I=3 everywhere (uniform)", dict(seeds=None, iters=3)),
        ("Newton I=2 everywhere", dict(seeds=None, iters=2)),
        ("Newton I=1 everywhere (affine gain + 1 step)", dict(seeds=None, iters=1)),
        ("dff = d (k=1, K=1)", dict(dff=1024)),
        ("dff = 2d (k=2)", dict(dff=2048)),
        ("parallel block: one norm feeds both branches", dict(parallel=True)),
        ("no shift-mix", dict(shift=False)),
        ("parallel block + I=2 + k=2", dict(parallel=True, seeds=None, iters=2, dff=2048)),
        ("parallel block + I=2 + k=1", dict(parallel=True, seeds=None, iters=2, dff=1024)),
        ("no lane masks (h=1, single lane or broadcast)", dict(lanes_block=False)),
    ]
    print("variant | boots/tick warm mean [min-max] | ctpt_diag | rot | diag_limbs | refresh calls")
    for label, kw in rows:
        o = T.run(args(ticks=30, **kw))
        w = warm(o); b = [t["boots"] for t in w]; t = w[-1]
        print(f"{label} | {sum(b)/len(b):6.1f} [{min(b)}-{max(b)}] | {t['ctpt_diag']} | {t['rot']} | "
              f"{t['diag_limbs']} | {t['refresh_calls']}")


def warm(o, skip=8):
    return o["per_tick"][skip:]


def probe():
    for label, kw in (("demo eager (depth 41, lam 22)", dict()),
                      ("demo lazy  (depth 41, lam 22)", dict(lazy=True)),
                      ("lam 68 eager (depth 87)", dict(depth=87, lam=68)),
                      ("lam 10 eager (depth 29)", dict(depth=29, lam=10))):
        o = T.run(args(**kw))
        print(f"== {label}")
        print("  site                  n     mean-remaining   min  max")
        for k, v in o["probe"].items():
            print(f"  {k:20s} {v['n']:5d} {v['mean_remaining']:9.2f} {v['min']:9d} {v['max']:4d}")


def grid():
    print("lam F margin I h model : boots/tick warm mean [min-max]   refreshCalls/tick")
    for lam in (7, 8, 10, 13, 16, 19, 22, 24, 30, 40, 68):
        for F in (3, 4):
            for margin in (4, 6):
                for I in (3, 4):
                    for h in (1, 2):
                        for lazy in (False, True):
                            try:
                                o = T.run(args(depth=max(41, lam + 19), lam=lam, F=F, margin=margin, iters=I,
                                               seeds=None, lanes_block=(h == 2), lazy=lazy, ticks=16))
                            except RuntimeError:
                                print(f"{lam:3d} {F} {margin} {I} {h} {'lazy ' if lazy else 'eager'} : level_overflow")
                                continue
                            w = warm(o, 4); b = [t["boots"] for t in w]
                            print(f"{lam:3d} {F} {margin} {I} {h} {'lazy ' if lazy else 'eager'} : "
                                  f"{sum(b)/len(b):7.1f} [{min(b)}-{max(b)}]   {w[-1]['refresh_calls']}")


def sweep():
    s = json.load(open(SEEDS))
    Itm = sum(s[f"L{i}.tm"]["iters"] for i in range(24)); Icm = sum(s[f"L{i}.cm"]["iters"] for i in range(24))
    Cpath = 4 * (Itm + Icm) + 26 * 24 + (4 * s["ln_out"]["iters"] + 6)
    Dmin = 2 * (Itm + Icm) + 16 * 24 + (2 * s["ln_out"]["iters"] + 3)
    print(f"sum I_tm {Itm}, sum I_cm {Icm}, I_out {s['ln_out']['iters']}; "
          f"C_path/tick (Lemma 8) {Cpath}; D_min/tick (Lemma 11) {Dmin}")
    print("lam depth | boots/tick eager | lazy | ideal ceil((C_path-depth)/lam) | floor ceil((D_min-depth)/lam)+L/lam")
    for lam in (7, 8, 10, 13, 16, 19, 22, 24, 30, 40, 50, 68):
        depth = lam + 19
        row = []
        for lazy in (False, True):
            try:
                o = T.run(args(depth=depth, lam=lam, lazy=lazy, ticks=24))
                b = [t["boots"] for t in warm(o)]
                row.append(f"{sum(b)/len(b):6.1f}")
            except RuntimeError:
                row.append("overflow")
        ideal = math.ceil((Cpath - depth) / lam)
        floor = max(0, math.ceil((Dmin - depth) / lam)) + 24 / lam
        print(f"{lam:3d} {depth:5d} | {row[0]} | {row[1]} | {ideal:3d} | {floor:5.1f}")


def sites():
    for lazy in (False, True):
        o = T.run(args(lazy=lazy, ticks=40))
        w = warm(o); b = [t["boots"] for t in w]
        print(f"== demo configuration, {'lazy' if lazy else 'eager'}: warm mean {sum(b)/len(b):.2f} "
              f"min {min(b)} max {max(b)} per layer {sum(b)/len(b)/24:.2f}; "
              f"refresh calls/tick {w[-1]['refresh_calls']}")
        c = {k: v for k, v in w[-1].items() if k not in ("boot_sites", "store_keys")}
        print("   counts (last tick):", c)
        cnt = Counter()
        for t in w:
            for s in t["boot_sites"]:
                cnt[s.split(".", 1)[1] if s.startswith("L") else s] += 1
        n = len(w)
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"   {v/n:6.2f}  {k}")


def sanity():
    # The record cell: ring 2^17, depth 41 (extra-depth 12), dnum 3, level budget {3,3}, boot-floor 3,
    # matvec-margin 4, compressed store, BLOCK layout, ONE lane, NO --lanes-block (h = 1), bundle seeds.
    for lazy in (False, True):
        r = B.cell(1024, 24, 17, dboot=19, F=3, margin=4, lanes_block=False, lazy=lazy, seeds=SEEDS, depth=41)
        print(f"== sanity cell ({'lazy' if lazy else 'eager'}): depth {r['D']} lam {r['lam']} REP {r['REP']} h=1")
        for k in ("boots", "boots_min", "boots_max", "B_min", "ctpt_diag", "M_min", "rot", "R_min", "ell_pt",
                  "bytes_h2d_GB", "t_boot_s", "t_pt_s", "t_rot_s", "t_elem_s", "t_rs_s", "t_add_s", "t_h2d_s",
                  "t_expand_s", "T_lb_s", "T_ub_s"):
            v = r[k]
            print(f"   {k:16s} {v:.3f}" if isinstance(v, float) else f"   {k:16s} {v}")
        # unit-cost implied by the report's ASSUMED constants at this cell's geometry
        N = 2 ** 17
        print(f"   assumed c_boot(2^17, D=41) ms = {B.c_boot(N, 41):.1f}; "
              f"c_pt(2^17, l={r['ell_pt']:.1f}) ms = {B.c_pt(N, r['ell_pt']):.3f}; "
              f"c_rot(2^17, l={r['ell_pt']:.1f}) ms = {B.c_rot(N, r['ell_pt']):.3f}")


# ---------------------------------------------------------------------------
# S3.3 schedule study. Every number in SCHEDULE_DESIGN_20260903.md comes from
# the output of this sub-command (trace_schedule_20260903.txt).
def _row(label, **kw):
    o = T.run(args(**kw)); w = warm(o); n = len(w); b = [t["boots"] for t in w]; t = w[-1]
    lam_ = sum(x["diag_limbs"] for x in w) / n; wv = sum(x["diag_limbs_wv"] for x in w) / n
    wkr = sum(x["diag_limbs_wkr"] for x in w) / n; pb = sum(x["pool_builds"] for x in w) / n
    print(f"{label:52s} | {sum(b)/n:6.1f} [{min(b):3d}-{max(b):3d}] | rot {t['rot']:6d} | "
          f"Lambda {lam_/1e6:5.2f}M (wv {wv/1e6:4.2f} wk+wr {wkr/1e6:4.2f}) | pool {pb:5.1f} | period {o['period']}")
    return o


def _hist(label, **kw):
    o = T.run(args(**kw)); w = warm(o); n = len(w); cnt = Counter()
    for t in w:
        for s_ in t["boot_sites"]:
            cnt[s_.split(".", 1)[1] if s_.startswith("L") else s_] += 1
    b = [t["boots"] for t in w]
    print(f"== {label}: warm mean {sum(b)/n:.2f} [{min(b)}-{max(b)}]; parent-rule boots/tick "
          f"{sum(t['boots_parent'] for t in w)/n:.1f}")
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:30]:
        print(f"   {v/n:6.2f}  {k}")
    print("   operand probe (remaining levels = depth - GetLevel(), warm part):")
    for k in ("Hs@tm.entry", "Hs@cm.entry", "y0@tm.rmsnorm", "y0@cm.rmsnorm", "u@tm.norm.out", "u2@cm.norm.out",
              "kk@birth", "hd@wv.in", "mv.win@in", "mv.wout@in", "mv.wk@in", "mv.wr@in", "mv.wv@in"):
        v = o["probe"].get(k)
        if v:
            print(f"     {k:16s} mean {v['mean_remaining']:6.2f} min {v['min']:3d} max {v['max']:3d}")


def _long(label, **kw):
    o = T.run(args(ticks=1000, **kw)); pt = o["per_tick"]; w = pt[100:]; b = [t["boots"] for t in w]
    seen = set(); newk = []
    for t in pt:
        ks = set(t["store_keys"]); newk.append(len(ks - seen)); seen |= ks
    last_new = max(i for i, n in enumerate(newk) if n > 0)
    depth = o["params"]["depth"]; Dpad = o["params"]["Dpad"]; h = o["params"]["h"]
    n_diag = h * Dpad - (h - 1)
    by = sum(n_diag * (depth + 1 - k[3]) * 16 * Dpad for k in seen)     # Lemma 13: 16*Dpad bytes per limb per (diag, half)
    print(f"{label:30s} | period {str(o['period']):12s} | boots ticks 100-999 mean {sum(b)/len(b):6.2f} [{min(b)}-{max(b)}] "
          f"| store keys/tick {len(set(pt[-1]['store_keys']))} distinct {len(seen)} = {by/1e9:.1f} GB (Lemma 13, h={h}) "
          f"| last new key tick {last_new} | new keys ticks 0..24: {newk[:25]}")


def schedule():
    print("### A. lever matrix (demo geometry: depth 41, lam 22, F 3, m 4, h 2, bundle seeds; ticks 8-29 warm)")
    print("variant | boots/tick warm mean [min-max] | rot/tick | Lambda = sum of operand limbs over weight multiplies (wv part, wk+wr part) | pool (re)builds/tick | steady-state period (first, recur) within 30 ticks")
    for lazy in (False, True):
        tag = "lazy " if lazy else "eager"
        print(f"== {tag}, lam 22 (depth 41)")
        _row(f"{tag} stock", lazy=lazy)
        for e in ("loop", "branch", "both", "always"):
            _row(f"{tag} parent-first/{e}", lazy=lazy, schedule="parent-first", pf_entry=e)
        _row(f"{tag} parent-first/loop, P2 tail-lift OFF", lazy=lazy, schedule="parent-first", no_pf_tail=True)
        for e in ("loop", "branch", "always"):
            _row(f"{tag} parent-first/{e} + newton-depth2", lazy=lazy, schedule="parent-first", pf_entry=e, nd2=True)
        _row(f"{tag} stock + newton-depth2", lazy=lazy, nd2=True)
        _row(f"{tag} stock + late-margin", lazy=lazy, late=True)
        _row(f"{tag} parent-first/loop + late-margin", lazy=lazy, schedule="parent-first", late=True)
        _row(f"{tag} stock + share-babies", lazy=lazy, share=True)
        _row(f"{tag} parent-first/loop + share-babies", lazy=lazy, schedule="parent-first", share=True)
        _row(f"{tag} ALL: pf/loop + share + late + newton-depth2", lazy=lazy, schedule="parent-first", share=True, late=True, nd2=True)
        print(f"== {tag}, lam 24 (depth 43 = D_max at dnum 3, Lemma 11)")
        _row(f"{tag} stock lam24", lazy=lazy, depth=43, lam=24)
        for e in ("loop", "branch", "always"):
            _row(f"{tag} parent-first/{e} lam24", lazy=lazy, depth=43, lam=24, schedule="parent-first", pf_entry=e)
        for e in ("loop", "branch"):
            _row(f"{tag} parent-first/{e} + newton-depth2 lam24", lazy=lazy, depth=43, lam=24, schedule="parent-first", pf_entry=e, nd2=True)
        _row(f"{tag} ALL lam24", lazy=lazy, depth=43, lam=24, schedule="parent-first", share=True, late=True, nd2=True)
    print("== pool LRU capacity (eager, lam 22): (re)builds per warm tick")
    for pool in (1, 2, 3, 4, 6, 8):
        _row(f"stock pool-levels={pool}", pool=pool)
        _row(f"parent-first/loop pool-levels={pool}", schedule="parent-first", pool=pool)
    print()
    print("### B. boot-site histograms (warm mean per tick) and operand depths")
    _hist("eager stock", )
    _hist("eager parent-first/loop", schedule="parent-first")
    _hist("eager parent-first/branch", schedule="parent-first", pf_entry="branch")
    _hist("eager parent-first/loop + newton-depth2", schedule="parent-first", nd2=True)
    _hist("eager stock + late-margin", late=True)
    _hist("lazy parent-first/loop", lazy=True, schedule="parent-first")
    _hist("lazy parent-first/loop + newton-depth2", lazy=True, schedule="parent-first", nd2=True)
    _hist("eager parent-first/loop lam24", depth=43, lam=24, schedule="parent-first")
    _hist("lazy parent-first/loop + newton-depth2 lam24", depth=43, lam=24, lazy=True, schedule="parent-first", nd2=True)
    print()
    print("### C. 1000-tick study: periodicity of the carried state (I1), store keys = distinct (layer, op, chunk, level) (I6)")
    for lazy in (False, True):
        tag = "lazy" if lazy else "eager"
        _long(f"{tag} stock", lazy=lazy)
        _long(f"{tag} parent-first/loop", lazy=lazy, schedule="parent-first")
        _long(f"{tag} parent-first/loop + newton-depth2", lazy=lazy, schedule="parent-first", nd2=True)
        _long(f"{tag} parent-first/loop lam24", lazy=lazy, depth=43, lam=24, schedule="parent-first")
        _long(f"{tag} parent-first/loop + newton-depth2 lam24", lazy=lazy, depth=43, lam=24, schedule="parent-first", nd2=True)
    print()
    print("### D. lam sweep (D = lam + 19), bundle seeds")
    s = json.load(open(SEEDS))
    Itm = sum(s[f"L{i}.tm"]["iters"] for i in range(24)); Icm = sum(s[f"L{i}.cm"]["iters"] for i in range(24)); Io = s["ln_out"]["iters"]
    Cpath = 4 * (Itm + Icm) + 26 * 24 + (4 * Io + 6)             # Lemma 8
    Cpath2 = 2 * (Itm + Icm) + 26 * 24 + (2 * Io + 6)            # depth-2 Newton: norm 2I + 6 per site
    print(f"C_path (Lemma 8) {Cpath}; with depth-2 Newton {Cpath2}")
    print("lam depth | stock eager | pf eager | pf+nd2 eager | stock lazy | pf lazy | pf+nd2 lazy | ideal ceil((C_path-D)/lam) | ideal nd2 | Lambda(M) eager stock/pf/pf+nd2")
    for lam in (13, 16, 19, 22, 24, 30, 40, 68):
        depth = lam + 19; cells = []; lams = []
        for lazy in (False, True):
            for kw in (dict(), dict(schedule="parent-first"), dict(schedule="parent-first", nd2=True)):
                try:
                    o = T.run(args(depth=depth, lam=lam, lazy=lazy, ticks=30, **kw)); w = warm(o)
                    b = [t["boots"] for t in w]; cells.append(f"{sum(b)/len(b):6.1f}")
                    if not lazy:
                        lams.append(f"{sum(t['diag_limbs'] for t in w)/len(w)/1e6:.2f}")
                except RuntimeError:
                    cells.append("ovfl"); lams.append("-")
        print(f"{lam:3d} {depth:5d} | " + " | ".join(cells) + f" | {math.ceil((Cpath-depth)/lam):3d} | {math.ceil((Cpath2-depth)/lam):3d} | " + "/".join(lams))
    print()
    print("### E. uniform-I grid (no seeds), lam 22 depth 41: I F m h model : stock | parent-first/loop | parent-first/loop + newton-depth2")
    for I in (3, 4):
        for F in (3, 4):
            for m in (4, 6):
                for h in (1, 2):
                    for lazy in (False, True):
                        cells = []
                        for kw in (dict(), dict(schedule="parent-first"), dict(schedule="parent-first", nd2=True)):
                            try:
                                o = T.run(args(F=F, margin=m, iters=I, seeds=None, lazy=lazy, ticks=24, lanes_block=(h == 2), **kw))
                                w = warm(o); b = [t["boots"] for t in w]; cells.append(f"{sum(b)/len(b):6.1f} [{min(b)}-{max(b)}]")
                            except RuntimeError:
                                cells.append("overflow")
                        print(f"{I} {F} {m} {h} {'lazy ' if lazy else 'eager'} : " + " | ".join(cells))
    print()
    print("### G. lazy model with the CPU-observed bootstrap semantics (EvalBootstrap output carries a pending rescale, --boot-deg2;")
    print("###    harness/cpu_real_model.cpp --cell level-probe / --cell pf-trace, 2026-09-03): demo geometry and lam 24")
    for kw, label in ((dict(), "stock"), (dict(schedule="parent-first"), "parent-first/loop"),
                      (dict(schedule="parent-first", nd2=True), "parent-first/loop + newton-depth2"),
                      (dict(schedule="parent-first", share=True, nd2=True), "ALL (share + pf + nd2)")):
        for dep, lam in ((41, 22), (43, 24)):
            a = args(depth=dep, lam=lam, lazy=True, ticks=30, **kw); a.boot_deg2 = True
            o = T.run(a); w = warm(o); b = [t["boots"] for t in w]
            print(f"lazy+boot-deg2 {label:36s} lam {lam} | {sum(b)/len(b):6.1f} [{min(b):3d}-{max(b):3d}] | rot {w[-1]['rot']:6d} | "
                  f"Lambda {sum(t['diag_limbs'] for t in w)/len(w)/1e6:5.2f}M | pool {sum(t['pool_builds'] for t in w)/len(w):5.1f}")
    print()
    print("### F. Tier B architecture variants (eager, lam 22): boots under stock | parent-first/loop | parent-first/loop + newton-depth2 ; ctpt_diag ; rot ; Lambda(M) under parent-first")
    rows = [
        ("baseline: bundle seeds, k=4, sequential block, shift-mix", dict()),
        ("Newton I=3 everywhere (uniform)", dict(seeds=None, iters=3)),
        ("Newton I=2 everywhere", dict(seeds=None, iters=2)),
        ("Newton I=1 everywhere", dict(seeds=None, iters=1)),
        ("dff = d (k=1, K=1)", dict(dff=1024)),
        ("dff = 2d (k=2)", dict(dff=2048)),
        ("parallel block: one norm feeds both branches", dict(parallel=True)),
        ("parallel block + I=3 (uniform)", dict(parallel=True, seeds=None, iters=3)),
        ("parallel block + I=2 + k=2", dict(parallel=True, seeds=None, iters=2, dff=2048)),
        ("parallel block + I=2 + k=1", dict(parallel=True, seeds=None, iters=2, dff=1024)),
    ]
    for label, kw in rows:
        cells = []; lam_pf = rot = cd = None
        for sk in (dict(), dict(schedule="parent-first"), dict(schedule="parent-first", nd2=True)):
            o = T.run(args(ticks=30, **kw, **sk)); w = warm(o); b = [t["boots"] for t in w]; t = w[-1]
            cells.append(f"{sum(b)/len(b):6.1f} [{min(b)}-{max(b)}]")
            if sk.get("schedule") == "parent-first" and not sk.get("nd2"):
                lam_pf = sum(x["diag_limbs"] for x in w) / len(w) / 1e6; rot = t["rot"]; cd = t["ctpt_diag"]
        print(f"{label:52s} | " + " | ".join(cells) + f" | {cd} | {rot} | {lam_pf:.2f}")


if __name__ == "__main__":
    {"probe": probe, "grid": grid, "sweep": sweep, "sites": sites, "sanity": sanity,
     "variants": variants, "schedule": schedule}[sys.argv[1]]()
