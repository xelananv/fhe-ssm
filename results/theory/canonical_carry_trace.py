#!/usr/bin/env python3
"""canonical_carry_trace.py -- the S3.3 replay (tick_level_trace.py, untouched) driven
with the 2026-09-04 canonical-carry hand-off, for the S3.5 complexity analysis.

WHAT IT ADDS (nothing in tick_level_trace.py is modified):
  --handoff canon   : gpu_real_model.cu `canon` (3ca2ae9): at every tick end (and on the
                      cold zeros) each carry -- scanState[l] and uPrev[l] -- is aligned DOWN
                      to depth-BOOT_FLOOR with alignTo (one mult-by-one per level, counted as
                      `align_steps`) and bootstrapped (counted as `canon_boots`), so it enters
                      the next tick at the post-bootstrap level depth-lam.
  --handoff drop    : the exact-drop variant of section 2b: the carry is dropped (no
                      arithmetic, no boot) to depth-BOOT_FLOOR; the next tick's own
                      refresh() boots it where the stock rule fires.
  --handoff dropmax : exact drop to a FIXED canonical level c chosen as the deepest
                      tick-end level any carry reached on the previous tick (reported); a
                      carry already deeper than c cannot be raised and is booted+dropped.
  --handoff stock   : the pre-3ca2ae9 behaviour (the drifting carry), for the control.

REPORTS per tick: stock-rule boots, canon boots, align steps (sum, max gap), the state
signature period (I1), the number of DISTINCT store keys seen so far (the store's
convergence: it must stop growing after tick 0 under a period-1 trajectory), pool
(re)builds, the per-layer matvec operand levels (the pool's level cycle), diag_limbs
(Lambda), and the tick-end level of every carry BEFORE the hand-off.
With --trace-levels it also prints the first refresh/boot lines of layer 0 (the level
of the fresh input Hs when its first boot fires -- the wire-size question of section 3).

Every rule is the replay's own (Lemmas 4-10 of TICK_SCALING_BOUNDS_20260903.md); this
file only adds the hand-off and the bookkeeping.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tick_level_trace as T  # noqa: E402

SEEDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ml-eval", "artifacts",
                     "pbd430a_seeds.json")


def handoff(sim, carries, mode, F, fixed_level, stats):
    """Apply the hand-off to every live carry. carries: list of (name, Ct)."""
    depth, lam = sim.depth, sim.lam
    for name, c in carries:
        if c is None:
            continue
        pre = sim.rep(c)
        stats["end_levels"].append((name, pre))
        if mode == "stock":
            continue
        if mode == "canon":                       # gpu_real_model.cu canon(): alignTo(depth-F) then boot()
            target = depth - F
            gap = max(0, target - pre)
            stats["align_steps"] += gap
            stats["max_gap"] = max(stats["max_gap"], gap)
            sim.align_to(c, target)               # mult-by-one chain, one level each (:2418-2423)
            sim.boot(c, "canon." + name.split(".")[0])
            stats["canon_boots"] += 1
        elif mode == "drop":                      # exact limb drop to depth-F, no boot
            target = depth - F
            if pre < target:
                c.e = target; c.deg = 1
            stats["drops"] += 1
        elif mode == "mixed":                     # scan: canon (boot); u: exact drop to --u-level (no boot)
            if name.startswith("scan"):
                target = depth - F
                gap = max(0, target - pre)
                stats["align_steps"] += gap
                stats["max_gap"] = max(stats["max_gap"], gap)
                sim.align_to(c, target)
                sim.boot(c, "canon.scan")
                stats["canon_boots"] += 1
            else:
                target = fixed_level
                if pre <= target:
                    c.e = target; c.deg = 1
                    stats["drops"] += 1
                else:                             # deeper than c_u: a drop cannot raise it -> boot+drop (counted)
                    sim.boot(c, "canon.u")
                    stats["canon_boots"] += 1
                    c.e = target; c.deg = 1
        elif mode == "bootdrop":                  # boot every carry (canon), then exact-drop to a chosen entry level
            target = depth - F
            gap = max(0, target - pre)
            stats["align_steps"] += gap
            stats["max_gap"] = max(stats["max_gap"], gap)
            sim.align_to(c, target)
            sim.boot(c, "canon." + name.split(".")[0])
            stats["canon_boots"] += 1
            lvl = stats["scan_level"] if name.startswith("scan") else stats["u_level"]
            if lvl is not None and lvl > sim.rep(c):
                c.e = lvl; c.deg = 1
                stats["drops"] += 1
        elif mode == "dropmax":                   # exact drop to the fixed level c*
            target = fixed_level
            if pre <= target:
                c.e = target; c.deg = 1
                stats["drops"] += 1
            else:                                 # deeper than c*: cannot be raised -> boot then drop
                sim.boot(c, "canon." + name.split(".")[0])
                stats["canon_boots"] += 1
                c.e = target; c.deg = 1
        else:
            raise ValueError(mode)


class SimDeferredScan(T.Sim):
    """The scan carry is READ, never multiplied in place: the readout at tick j of an
    m-tick period is (c . decay^(j+1)) . s_base + sum_i (c . decay^(j-i) b) . x_i, so
    s_base keeps its level for m ticks; every m-th tick s_base <- decay^m . s_base + acc
    (one multiply) and the carry is bootstrapped (canon). Every other rule is T.Sim's.
    Bookkeeping: the terms are ct x pt on copies; the readout's level is max(x_i) + 1."""
    def __init__(self, *a, scan_period=1, **kw):
        super().__init__(*a, **kw)
        self.scan_period = scan_period
        self.phase = 0                              # tick index within the period
        self.xhist = None                           # per layer: levels of the x_i kept this period

    def layer(self, l, Hs, scan, uPrev, I_tm, I_cm):
        if self.xhist is None:
            self.xhist = [[] for _ in range(64)]
        self.note("Hs@tm.entry", Hs)
        if self.s.pf:
            self.refresh_pf(Hs, self.need_entry("tm", I_tm), f"L{l}.tm.rmsnorm.Hs")
        else:
            self.refresh(Hs, 3, f"L{l}.tm.rmsnorm.Hs")
        u = self.rmsnorm(Hs, I_tm, f"L{l}.tm.rmsnorm", kind="tm")
        u_norm = u.copy()
        if self.shift_mix:
            a = u.copy(); self.mul_pt(a)
            if uPrev[l] is not None:
                b = uPrev[l].copy()
                self.refresh(b, 2, f"L{l}.tm.shiftMix.uPrev")
                self.mul_pt(b)
                self.add_aligned(a, b, f"L{l}.tm.shiftMix")
            uPrev[l] = u.copy()
            u = a
        x = self.matvec(u, self.d, self.d, op="win", layer=l)
        # ---- deferred scan: the carry is only read ----
        self.refresh(x, 3, f"L{l}.tm.scan.x")
        bx = x.copy(); self.mul_pt(bx)                              # (c . decay^(j-i) b) . x_i, this tick's term
        self.xhist[l].append(bx.copy())
        S = scan[l].copy(); self.mul_pt(S)                          # (c . decay^(j+1)) . s_base on a COPY
        for term in self.xhist[l]:                                  # + the period's terms (j+1 of them)
            self.add(S, term.copy())
            if len(self.xhist[l]) > 1:
                self.c.ctpt_elem += 1                               # one extra ct x pt per kept term (level-neutral copy)
        if self.phase == self.scan_period - 1:                      # end of the period: fold into the base
            self.refresh(scan[l], 3, f"L{l}.tm.scan.state")
            self.mul_pt(scan[l])                                    # decay^m . s_base
            for term in self.xhist[l]:
                self.add(scan[l], term.copy())
            self.xhist[l] = []
        zc = S.copy(); self.mul_pt(zc)
        zd = x.copy(); self.mul_pt(zd)
        self.add_aligned(zc, zd, f"L{l}.tm.gate.readout")
        self.refresh(zc, 5, f"L{l}.tm.gate.zc")
        g = self.poly_gate3(zc)
        attn = self.matvec(g, self.d, self.d, op="wout", layer=l)
        if not self.parallel_block:
            self.add_aligned(Hs, attn, f"L{l}.tm.residual")
        if self.parallel_block:
            u2 = u_norm
        else:
            self.note("Hs@cm.entry", Hs)
            if self.s.pf:
                self.refresh_pf(Hs, self.need_entry("cm", I_cm), f"L{l}.cm.rmsnorm.Hs")
            else:
                self.refresh(Hs, 3, f"L{l}.cm.rmsnorm.Hs")
            u2 = self.rmsnorm(Hs, I_cm, f"L{l}.cm.rmsnorm", kind="cm")
        if self.s.pf:
            self.refresh_pf(u2, self.need_u2(), f"L{l}.cm.u2")
        key = ("u2", l)
        kks = [self.matvec(u2, min(self.Dpad, self.dff - ch * self.Dpad), self.d, op="wk", share_key=key, layer=l, chunk=ch)
               for ch in range(self.K)]
        rrs = [self.matvec(u2, min(self.Dpad, self.dff - ch * self.Dpad), self.d, op="wr", share_key=key, layer=l, chunk=ch)
               for ch in range(self.K)]
        self._baby_key = None
        ffn = None
        for ch in range(self.K):
            kk, rr = kks[ch].copy(), rrs[ch].copy()
            self.refresh(kk, 5, f"L{l}.cm.hidden.ch{ch}.kk")
            self.refresh(rr, 4, f"L{l}.cm.hidden.ch{ch}.rr")
            k2 = self.mul_cc(kk, kk)
            actT = kk.copy(); self.mul_pt(actT)
            t2 = k2.copy(); self.mul_pt(t2)
            self.add_aligned(actT, t2, f"L{l}.cm.hidden.ch{ch}.act12")
            kkA = kk.copy(); self.align_to(kkA, self.rep(k2))
            k3 = self.mul_cc(k2, kkA); self.mul_pt(k3)
            self.add_aligned(actT, k3, f"L{l}.cm.hidden.ch{ch}.act3")
            r2 = self.mul_cc(rr, rr)
            gt = rr.copy(); self.mul_pt(gt)
            g2t = r2.copy(); self.mul_pt(g2t)
            self.add_aligned(gt, g2t, f"L{l}.cm.hidden.ch{ch}.gate")
            self.refresh(actT, 2, f"L{l}.cm.hidden.ch{ch}.actT")
            self.refresh(gt, 2, f"L{l}.cm.hidden.ch{ch}.gt")
            uu = max(self.rep(actT), self.rep(gt))
            self.align_to(actT, uu); self.align_to(gt, uu)
            hd = self.mul_cc(actT, gt)
            self.refresh(hd, 2, f"L{l}.cm.hidden.ch{ch}.hd")
            if not self.s.late_margin:
                self.refresh(hd, self.margin, f"L{l}.cm.wv.ch{ch}.margin")
            part = self.matvec(hd, self.d, self.Dpad, op="wv", layer=l, chunk=ch)
            if ffn is None:
                ffn = part
            else:
                self.add_aligned(ffn, part, f"L{l}.cm.ffnAccum.ch{ch}")
        if self.parallel_block:
            self.add_aligned(Hs, attn, f"L{l}.tm.residual")
        self.add_aligned(Hs, ffn, f"L{l}.cm.residual")


def run(args):
    depth = 10 + 19 + args.extra_depth if args.depth is None else args.depth
    lam = depth - args.dboot if args.lam is None else args.lam
    L = args.layers
    s = json.load(open(args.seeds))
    I_tm = [s[f"L{i}.tm"]["iters"] for i in range(L)]
    I_cm = [s[f"L{i}.cm"]["iters"] for i in range(L)]
    I_out = s["ln_out"]["iters"]
    sched = T.Sched(schedule=args.schedule, pf_entry=args.pf_entry, pf_tail=not args.no_pf_tail,
                    share_babies=args.share_babies, late_margin=args.late_margin,
                    newton_depth2=args.newton_depth2, pool_levels=args.pool_levels, boot_deg2=args.boot_deg2)
    if args.scan_period > 1:
        sim = SimDeferredScan(depth, lam, args.boot_floor, args.margin, args.dpad, args.d, args.dff,
                              args.lanes_block, True, args.lazy, alpha_res_is_one=True,
                              parallel_block=args.parallel_block, sched=sched, scan_period=args.scan_period)
    else:
        sim = T.Sim(depth, lam, args.boot_floor, args.margin, args.dpad, args.d, args.dff,
                    args.lanes_block, True, args.lazy, alpha_res_is_one=True,
                    parallel_block=args.parallel_block, sched=sched)
    sim.trace_levels = args.trace_levels
    scan = [T.Ct(0, 1) for _ in range(L)]
    # canonical cold start (:4376-4387): explicit zero u_{-1} so the shift-mix takes the
    # resumed path at tick 0 as well; stock cold start leaves uPrev None (u_{-1}=0 folded).
    uPrev = [T.Ct(0, 1) for _ in range(L)] if args.handoff != "stock" else [None] * L
    fixed_level = depth - args.boot_floor if args.u_level is None else args.u_level
    per_tick, sigs, period, seen_keys = [], {}, None, set()
    if args.handoff != "stock":
        st = {"canon_boots": 0, "align_steps": 0, "max_gap": 0, "drops": 0, "end_levels": [],
              "scan_level": args.scan_level, "u_level": args.u_level}
        handoff(sim, [(f"scan.{l}", scan[l]) for l in range(L)] + [(f"u.{l}", uPrev[l]) for l in range(L)],
                args.handoff, args.boot_floor, fixed_level, st)
        cold = st
    else:
        cold = None
    for t in range(args.ticks):
        c0 = T.Counts(**vars(sim.c))
        sim.trace, sim.store_keys, sim.matvec_levels, sim.level_lines = [], [], [], []
        if args.input_level > 0:
            _orig_ct = T.Ct
            class _CtIn(T.Ct):
                _first = [True]
                def __init__(self, e=0, deg=1):
                    if _CtIn._first[0] and e == 0 and deg == 1:
                        _CtIn._first[0] = False
                        e = args.input_level
                    super().__init__(e, deg)
            T.Ct = _CtIn
            hf = sim.tick(L, I_tm, I_cm, I_out, scan, uPrev, args.final_boot)
            T.Ct = _orig_ct
        else:
            hf = sim.tick(L, I_tm, I_cm, I_out, scan, uPrev, args.final_boot)
        out_level = sim.rep(hf)
        dc = {k: getattr(sim.c, k) - getattr(c0, k) for k in vars(sim.c)}
        stock_boots = dc["boots"]
        st = {"canon_boots": 0, "align_steps": 0, "max_gap": 0, "drops": 0, "end_levels": [],
              "scan_level": args.scan_level, "u_level": args.u_level}
        if args.scan_period > 1:
            end_of_period = (sim.phase == args.scan_period - 1)
            carries = ([(f"scan.{l}", scan[l]) for l in range(L)] if end_of_period else []) \
                      + [(f"u.{l}", uPrev[l]) for l in range(L)]
            handoff(sim, carries, args.handoff, args.boot_floor, fixed_level, st)
            if not end_of_period:
                st["end_levels"] += [(f"scan.{l}", sim.rep(scan[l])) for l in range(L)]
            sim.phase = (sim.phase + 1) % args.scan_period
        else:
            handoff(sim, [(f"scan.{l}", scan[l]) for l in range(L)] + [(f"u.{l}", uPrev[l]) for l in range(L)],
                    args.handoff, args.boot_floor, fixed_level, st)
        if args.handoff == "dropmax":
            fixed_level = max(lv for _, lv in st["end_levels"])
        keys_this = set(sim.store_keys)
        new_keys = len(keys_this - seen_keys)
        seen_keys |= keys_this
        per_layer_levels = [sim.matvec_levels[i * 14:(i + 1) * 14] for i in range(L)]   # 14 matvec calls per layer (K=4)
        same_every_layer = all(per_layer_levels[i] == per_layer_levels[0] for i in range(L))
        rec = {"tick": t, "output_level": out_level, "input_level": args.input_level,
               "request_bytes": 2 * (depth + 1 - args.input_level) * (2 * args.dpad * (sim.Dpad and 64)) * 8 if False else None,
               "boots_stock_rule": stock_boots, "canon_boots": st["canon_boots"],
               "boots_total": stock_boots + st["canon_boots"], "align_steps": st["align_steps"],
               "max_gap": st["max_gap"], "rot": dc["rot"], "ctpt_diag": dc["ctpt_diag"],
               "diag_limbs": dc["diag_limbs"], "pool_builds": dc["pool_builds"],
               "store_keys_this_tick": len(keys_this), "new_store_keys": new_keys,
               "store_keys_cumulative": len(seen_keys),
               "matvec_levels_L0": per_layer_levels[0], "matvec_levels_L1": per_layer_levels[1],
               "matvec_levels_Llast": per_layer_levels[L - 1],
               "levels_same_from_L1": all(per_layer_levels[i] == per_layer_levels[1] for i in range(1, L)),
               "levels_same_every_layer": same_every_layer,
               "distinct_levels_per_layer": sorted(set(per_layer_levels[0])),
               "end_levels_scan": [lv for n, lv in st["end_levels"] if n.startswith("scan")],
               "end_levels_u": [lv for n, lv in st["end_levels"] if n.startswith("u.")],
               "fixed_level": fixed_level if args.handoff == "dropmax" else None,
               "boot_sites": sim.trace, "level_lines": sim.level_lines if args.trace_levels else []}
        per_tick.append(rec)
        sig = T.state_signature(scan, uPrev)
        if sig in sigs and period is None:
            period = (sigs[sig], t)
        sigs.setdefault(sig, t)
    return {"params": {"depth": depth, "lam": lam, "F": args.boot_floor, "margin": args.margin, "L": L,
                       "lanes_block": args.lanes_block, "h": sim.h, "lazy": args.lazy,
                       "handoff": args.handoff, "sched_active": sched.active() or ["stock"],
                       "I_tm": I_tm, "I_cm": I_cm, "I_out": I_out},
            "cold": cold, "period": period, "per_tick": per_tick}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--extra-depth", type=int, default=12)
    ap.add_argument("--dboot", type=int, default=19)
    ap.add_argument("--lam", type=int, default=None)
    ap.add_argument("--boot-floor", type=int, default=3)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--dpad", type=int, default=1024)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--dff", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--seeds", type=str, default=SEEDS)
    ap.add_argument("--lanes-block", action="store_true")
    ap.add_argument("--lazy", action="store_true")
    ap.add_argument("--final-boot", action="store_true")
    ap.add_argument("--parallel-block", action="store_true")
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--handoff", choices=("stock", "canon", "drop", "dropmax", "mixed", "bootdrop"), default="canon")
    ap.add_argument("--scan-level", type=int, default=None, help="bootdrop: entry level of the scan carry after its boot")
    ap.add_argument("--input-level", type=int, default=0, help="level the client ciphertext enters at (0 = fresh, 42 limbs)")
    ap.add_argument("--u-level", type=int, default=None, help="mixed: fixed level the u carry is dropped to")
    ap.add_argument("--scan-period", type=int, default=1, help="deferred scan: boot the scan carry every m ticks")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--sites", action="store_true")
    T.add_sched_args(ap)
    args = ap.parse_args()
    out = run(args)
    p = out["params"]
    print(f"handoff={p['handoff']} depth={p['depth']} lam={p['lam']} F={p['F']} m={p['margin']} L={p['L']} "
          f"h={p['h']} lazy={p['lazy']} sched={p['sched_active']} period={out['period']}")
    if out["cold"]:
        c = out["cold"]
        print(f"cold: canon_boots={c['canon_boots']} align_steps={c['align_steps']} max_gap={c['max_gap']}")
    for r in out["per_tick"]:
        print(f"tick {r['tick']}: out-level {r['output_level']} | boots stock-rule {r['boots_stock_rule']} + canon {r['canon_boots']} = "
              f"{r['boots_total']} | align steps {r['align_steps']} (max gap {r['max_gap']}) | rot {r['rot']} | "
              f"Lambda {r['diag_limbs']} | pool builds {r['pool_builds']} | store keys this tick "
              f"{r['store_keys_this_tick']} new {r['new_store_keys']} cum {r['store_keys_cumulative']} | "
              f"levels/layer {r['distinct_levels_per_layer']} same-every-layer={r['levels_same_every_layer']}"
              + (f" | c*={r['fixed_level']}" if r['fixed_level'] is not None else ""))
        print(f"        L0 matvec levels {r['matvec_levels_L0']}")
        print(f"        L1 matvec levels {r['matvec_levels_L1']}  (same for L1..L{p['L']-1}: {r['levels_same_from_L1']}; L{p['L']-1}: {sorted(set(r['matvec_levels_Llast']))})")
        print(f"        tick-end levels scan {r['end_levels_scan']}")
        print(f"        tick-end levels u    {r['end_levels_u']}")
    if args.sites:
        from collections import Counter
        cnt = Counter(s.split(".", 1)[1] if s.startswith("L") else s for s in out["per_tick"][-1]["boot_sites"])
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"  {v:4d}  {k}")
    if args.trace_levels:
        for ln in out["per_tick"][-1]["level_lines"][:40]:
            print(json.dumps(ln))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
