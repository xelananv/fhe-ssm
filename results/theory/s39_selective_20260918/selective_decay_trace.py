#!/usr/bin/env python3
"""selective_decay_trace.py -- what would a NON-FIXED-COEFFICIENT (selective) time-mix cost per tick?

THIS IS NOT A MEASUREMENT and no such model exists in this repository: it is the S3.3/S3.5 level-trace replay
(results/theory/tick_level_trace.py, untouched, imported) with the 2026-09-04 canonical-carry hand-off
(results/theory/canonical_carry_trace.py, untouched, imported) and ONE replaced section of the layer: the scan.
Every other statement of the layer is a verbatim copy of tick_level_trace.Sim.layer (checked below: with every lever
off this file reproduces the imported replay's per-tick counts exactly -- run with --selfcheck).

Cost is SHAPE, not weights (memory note "size-scaling law"): the counts below need no trained model.

The levers (all OFF by default):
  --sel none|diag|proj|lowrank   how the decay a_t depends on the token
        none    : a is a plaintext vector (the demo's model):      S <- a (.) S + b (.) x          (two ct x pt)
        diag    : a_t = q(w (.) u), q a degree-2 polynomial, no matvec (per-channel, "diagonal" selectivity)
        proj    : a_t = q(W_a u), one new d->d matvec per layer (RG-LRU / Griffin-shaped recurrence gate)
        lowrank : a_t = q(A (B u)), d->r->d (Mamba's dt_rank shape); modelled as TWO ct x pt levels, 2r diagonals and
                  lg(d/r) fold rotations per half (a rectangular Halevi-Shoup matvec the harness does NOT have: a projection)
      selective update, all three:  S <- x' + a_t (.) (S - x')   (ONE ct x ct; x' = b (.) x as today)
  --sel-poly P       depth of q on top of the projection (1 = one square, the degree-2 gate; 2 = degree 4 (softplus/exp-like))
  --decay-exp P      Mamba's A-bar = exp(Delta (.) A_n): a further depth-P polynomial in Delta PER STATE COMPONENT n, with
                     per-channel plaintext coefficients (ct x pt folded into the ladder's operands, one level each as the
                     harness prices every ct x pt); 0 = a_t is used directly
  --input-gate       the S3.8 hybrid input gate x <- i(u) (.) x (as an earlier replay study, not included): with --sel proj this
                     is the RG-LRU pair (recurrence gate + input gate)
  --state-width N    N carried state ciphertexts per layer (Mamba: N = 16). Under this packing (one ciphertext = lanes x
                     channels) a d x N state is N ciphertexts per 64-lane block -- or, equivalently, 64/N lanes per block.
  --bc-selective     Mamba's B_t, C_t in R^N: 2N per-lane inner products per layer (ct x pt row, up-tree of lg Dpad
                     rotations, under --lanes-block the e0 mask + down-tree as in the norm's block sum) and a ct x ct each
                     at the state update (B) and at the readout (C). Off: B, C are plaintext vectors (ct x pt).
  --conv K           K-tap causal convolution on x (Mamba: 4): K-1 carried ciphertexts per layer (the demo's shift-mix
                     is the K = 2 case and stays as it is; K > 2 adds K-2 carries and K-2 ct x pt + adds)
  --robust-iters K   every norm site runs max(iters, K) Newton steps (the demo's --newton-robust 0.3 8: K = 8; the
                     long-run plan's 0.1 12: K = 12); 0 = the bundle's iteration counts
Hand-off: canonical (every carry aligned down and bootstrapped at the tick end), as in the demo.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tick_level_trace as T          # noqa: E402
import canonical_carry_trace as CC    # noqa: E402


class SimSel(T.Sim):
    def __init__(self, *a, sel="none", sel_poly=1, decay_exp=0, input_gate=False, state_width=1, bc_selective=False,
                 conv=2, rank=64, drops=None, **kw):
        super().__init__(*a, **kw)
        self.drops = drops or {}                            # S3.7 X8: op class -> R (remaining levels the operand is dropped to)
        self.sel, self.sel_poly, self.decay_exp = sel, sel_poly, decay_exp
        self.input_gate, self.N, self.bc_sel, self.conv, self.rank = input_gate, state_width, bc_selective, conv, rank
        self.extra = {"sel_ctct": 0, "sel_ctpt": 0, "sel_rot": 0, "sel_matvec": 0, "state_boots_in_tick": 0}

    # ---- S3.7 X8 exact drops (gpu_real_model_x.cu x8DropFor / x8DropTo): before a matvec of class op the operand is dropped IN
    # PLACE to depth - R (no arithmetic, no boot); an operand already deeper is left alone. win/wdecay/wgate share R(win).
    def matvec(self, u, rows, in_cols, op="win", share_key=None, layer=-1, chunk=0):
        cls = {"win": "win", "wdecay": "win", "wgate": "win", "wout": "wout", "wk": "wkr", "wr": "wkr", "wv": "wv"}.get(op)
        R = self.drops.get(cls, 0)
        if R > 0:
            target = self.depth - R
            if self.rep(u) < target:
                u.e, u.deg = target, 1
        return super().matvec(u, rows, in_cols, op=op, share_key=share_key, layer=layer, chunk=chunk)

    # ---- helpers -----------------------------------------------------------------------------------------------
    def lane_scalar(self, u, site):
        """One per-lane inner product <row, u> broadcast to every channel slot of its lane: ct x pt with the row (+1), the
        norm's block sum (up-tree lg Dpad rotations; under --lanes-block the e0 mask (+1) and the down-tree)."""
        v = u.copy(); self.mul_pt(v); self.extra["sel_ctpt"] += 1
        lg = int(math.log2(self.Dpad))
        self.c.rot += lg; self.c.rot_limbs += lg * self.limbs(v); self.extra["sel_rot"] += lg
        if self.lanes_block:
            self.mul_pt(v); self.extra["sel_ctpt"] += 1
            self.c.rot += lg; self.c.rot_limbs += lg * self.limbs(v); self.extra["sel_rot"] += lg
        return v

    def lowrank_matvec(self, u, l):
        """d -> r -> d as two rectangular diagonal matvecs: r diagonals each, fold / unfold by lg(d/r) rotations per half,
        one level each. NOT a harness primitive (projection)."""
        h, r = self.h, self.rank
        bs = math.ceil(math.sqrt(r)); g = math.ceil(r / bs)             # BSGS over the r extended diagonals
        v = u.copy()
        for _ in range(2):
            n_rot = h * (bs - 1) + (h - 1) + (g - 1) + h * int(math.log2(max(self.Dpad // r, 1)))
            self.c.rot += n_rot; self.c.rot_limbs += n_rot * self.limbs(v); self.extra["sel_rot"] += n_rot
            n_diag = h * r
            self.c.ctpt_diag += n_diag; self.c.diag_limbs += n_diag * self.limbs(v); self.c.h2d_words_per_limb += n_diag
            self.c.rescale += 1
            v = T.Ct(v.e + 1, 2 if self.lazy else 1)
        self.extra["sel_matvec"] += 2
        return v

    def poly_on(self, v, depth, per_level_ctpt=1):
        """A depth-`depth` power ladder on v (one ct x ct per level) with its plaintext coefficients."""
        for _ in range(depth):
            v = self.mul_cc(v, v); self.extra["sel_ctct"] += 1
        self.c.ctpt_elem += per_level_ctpt * max(depth, 0); self.extra["sel_ctpt"] += per_level_ctpt * max(depth, 0)
        return v

    def decay_gate(self, u, l):
        if self.sel == "diag":
            v = u.copy(); self.refresh(v, 2, f"L{l}.tm.decay.u"); self.mul_pt(v); self.extra["sel_ctpt"] += 1
        elif self.sel == "proj":
            v = self.matvec(u, self.d, self.d, op="wdecay", share_key=("u", l), layer=l); self.extra["sel_matvec"] += 1
        elif self.sel == "lowrank":
            v = self.lowrank_matvec(u, l)
        else:
            raise ValueError(self.sel)
        self.refresh(v, self.sel_poly + self.decay_exp + 2, f"L{l}.tm.decay.v")   # the ladder(s) + the state product, harness discipline
        return self.poly_on(v, self.sel_poly)

    # ---- the layer: tick_level_trace.Sim.layer verbatim except the block between the two markers ------------------
    def layer(self, l, Hs, scan, uPrev, I_tm, I_cm):
        self.note("Hs@tm.entry", Hs)
        if self.s.pf:
            self.refresh_pf(Hs, self.need_entry("tm", I_tm), f"L{l}.tm.rmsnorm.Hs")
        else:
            self.refresh(Hs, 3, f"L{l}.tm.rmsnorm.Hs")
        u = self.rmsnorm(Hs, I_tm, f"L{l}.tm.rmsnorm", kind="tm")
        self.note("u@tm.norm.out", u)
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
        u_full = u.copy()                                   # the operand BEFORE win's X8 drop (elementwise gate paths read this one)
        x = self.matvec(u, self.d, self.d, op="win", layer=l)
        # ======================= replaced section: conv taps, gates, the (selective) state update =======================
        states = scan[l] if isinstance(scan[l], list) else [scan[l]]
        if self.conv > 2:                                   # K-tap conv on x: K-2 extra carried x's (ct x pt each, aligned adds)
            hist = self.xcarry[l]
            acc = x.copy(); self.mul_pt(acc)
            for j, xp in enumerate(hist):
                t = xp.copy(); self.refresh(t, 2, f"L{l}.tm.conv.tap{j}"); self.mul_pt(t)
                self.add_aligned(acc, t, f"L{l}.tm.conv")
            self.xcarry[l] = ([x.copy()] + hist)[: self.conv - 2]
            x = acc
        if self.input_gate:                                 # S3.8 Vector A stub, verbatim
            gi = self.matvec(u, self.d, self.d, op="wgate", share_key=("u", l), layer=l)
            gi = self.mul_cc(gi, gi)
            self.refresh(x, 2, f"L{l}.tm.igate.x")
            self.refresh(gi, 2, f"L{l}.tm.igate.g")
            x = self.mul_cc(x, gi)
        a_t = self.decay_gate(u if self.sel in ("proj", "lowrank") else u_full, l) if self.sel != "none" else None
        if self.bc_sel:
            self.refresh(u_full, 3, f"L{l}.tm.bc.u")        # the 2N inner products need 2 levels (+1 for their ct x ct)
        Bn = [self.lane_scalar(u_full, f"L{l}.B{n}") for n in range(self.N)] if self.bc_sel else None
        Cn = [self.lane_scalar(u_full, f"L{l}.C{n}") for n in range(self.N)] if self.bc_sel else None
        readout = None
        for n, st in enumerate(states):
            b0 = self.c.boots
            self.refresh(st, 3, f"L{l}.tm.scan.state")
            if n == 0:
                self.refresh(x, 3, f"L{l}.tm.scan.x")       # x itself, once (as today: later uses see the refreshed x)
            if self.bc_sel:                                 # (Delta (.) x) B_n : ct x ct
                bq = Bn[n].copy(); self.refresh(bq, 2, f"L{l}.tm.scan.B")
                bx = self.mul_cc(x, bq); self.extra["sel_ctct"] += 1
            else:
                bx = x.copy(); self.mul_pt(bx)              # b (.) x  (as today)
            if self.sel == "none":
                self.mul_pt(st)                             # decay (.) s  (as today)
                self.add_aligned(st, bx, f"L{l}.tm.scan")
            else:
                an = a_t.copy()
                if self.decay_exp:                          # A-bar_n = exp(Delta (.) A_n): its own ladder per component
                    an = self.poly_on(an, self.decay_exp)
                # S <- x' + a (.) (S - x')
                d = st.copy(); self.add_aligned(d, bx.copy(), f"L{l}.tm.scan.diff")
                self.refresh(an, 2, f"L{l}.tm.scan.a")
                self.refresh(d, 2, f"L{l}.tm.scan.d")
                prod = self.mul_cc(an, d); self.extra["sel_ctct"] += 1
                self.add_aligned(prod, bx, f"L{l}.tm.scan")
                st.e, st.deg = prod.e, prod.deg
            self.extra["state_boots_in_tick"] += self.c.boots - b0
            term = st.copy()
            if self.bc_sel:                                 # C_n (.) h_n : ct x ct
                cq = Cn[n].copy(); self.refresh(cq, 2, f"L{l}.tm.readout.C"); self.refresh(term, 2, f"L{l}.tm.readout.h")
                term = self.mul_cc(term, cq); self.extra["sel_ctct"] += 1
            else:
                self.mul_pt(term)                           # c (.) S  (as today: zc)
            if readout is None:
                readout = term
            else:
                self.add_aligned(readout, term, f"L{l}.tm.readout.sum")
        zc = readout
        # ======================= end of the replaced section (the rest is the imported layer, verbatim) ==================
        zd = x.copy(); self.mul_pt(zd)
        self.add_aligned(zc, zd, f"L{l}.tm.gate.readout")
        self.refresh(zc, 5, f"L{l}.tm.gate.zc")
        g = self.poly_gate3(zc)
        attn = self.matvec(g, self.d, self.d, op="wout", layer=l)
        if not self.alpha_one:
            self.mul_pt(attn)
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
            self.note("u2@cm.norm.out", u2)
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
            self.note("kk@birth", kk)
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
            self.note("hd@wv.in", hd)
            part = self.matvec(hd, self.d, self.Dpad, op="wv", layer=l, chunk=ch)
            if ffn is None:
                ffn = part
            else:
                self.add_aligned(ffn, part, f"L{l}.cm.ffnAccum.ch{ch}")
        if self.parallel_block:
            self.add_aligned(Hs, attn, f"L{l}.tm.residual")
        self.add_aligned(Hs, ffn, f"L{l}.cm.residual")


def run(args):
    depth = 10 + 19 + args.extra_depth
    lam = depth - args.dboot
    L = args.layers
    s = json.load(open(args.seeds))
    rk = args.robust_iters
    I_tm = [max(s[f"L{i}.tm"]["iters"], rk) for i in range(L)]
    I_cm = [max(s[f"L{i}.cm"]["iters"], rk) for i in range(L)]
    I_out = max(s["ln_out"]["iters"], rk)
    sched = T.Sched(schedule=args.schedule, pf_entry=args.pf_entry, pf_tail=not args.no_pf_tail, share_babies=args.share_babies,
                    late_margin=args.late_margin, newton_depth2=args.newton_depth2, pool_levels=args.pool_levels,
                    boot_deg2=args.boot_deg2)
    sim = SimSel(depth, lam, args.boot_floor, args.margin, args.dpad, args.d, args.dff, args.lanes_block, True, args.lazy,
                 alpha_res_is_one=True, parallel_block=False, sched=sched, sel=args.sel, sel_poly=args.sel_poly,
                 decay_exp=args.decay_exp, input_gate=args.input_gate, state_width=args.state_width,
                 bc_selective=args.bc_selective, conv=args.conv, rank=args.rank,
                 drops={"wv": args.drop_wv, "wkr": args.drop_wkr, "wout": args.drop_wout, "win": args.drop_win})
    N = args.state_width
    scan = [[T.Ct(0, 1) for _ in range(N)] for _ in range(L)] if (N > 1 or args.force_list) else [T.Ct(0, 1) for _ in range(L)]
    uPrev = [T.Ct(0, 1) for _ in range(L)]
    sim.xcarry = [[T.Ct(0, 1) for _ in range(max(args.conv - 2, 0))] for _ in range(L)]

    def carries():
        out = []
        for l in range(L):
            sts = scan[l] if isinstance(scan[l], list) else [scan[l]]
            out += [(f"scan.{l}.{n}", c) for n, c in enumerate(sts)]
        out += [(f"u.{l}", uPrev[l]) for l in range(L)]
        for l in range(L):
            out += [(f"xc.{l}.{j}", c) for j, c in enumerate(sim.xcarry[l])]
        return out

    def st0():
        return {"canon_boots": 0, "align_steps": 0, "max_gap": 0, "drops": 0, "end_levels": [], "scan_level": None, "u_level": None}
    cold = st0(); CC.handoff(sim, carries(), "canon", args.boot_floor, depth - args.boot_floor, cold)
    rows = []
    for t in range(args.ticks):
        c0 = T.Counts(**vars(sim.c)); e0 = dict(sim.extra); sim.trace = []; sim.store_keys = []; sim.matvec_levels = []
        hf = sim.tick(L, I_tm, I_cm, I_out, scan, uPrev, False)
        dc = {k: getattr(sim.c, k) - getattr(c0, k) for k in vars(sim.c)}
        st = st0(); CC.handoff(sim, carries(), "canon", args.boot_floor, depth - args.boot_floor, st)
        ex = {k: sim.extra[k] - e0[k] for k in sim.extra}
        rows.append({"tick": t, "out_level": sim.rep(hf), "boots_stock_rule": dc["boots"], "canon_boots": st["canon_boots"],
                     "boots_total": dc["boots"] + st["canon_boots"], "align_steps": st["align_steps"], "rot": dc["rot"],
                     "ctct": dc["ctct"], "ctpt_elem": dc["ctpt_elem"] + st["align_steps"], "ctpt_diag": dc["ctpt_diag"],
                     "diag_limbs": dc["diag_limbs"], "matvec_chunks": dc["matvec_chunks"], "carried_cts": len(carries()), **ex,
                     "end_levels_scan": sorted({lv for n, lv in st["end_levels"] if n.startswith("scan")}),
                     "boot_sites": sim.trace})
    return {"params": {"depth": depth, "lam": lam, "F": args.boot_floor, "L": L, "h": sim.h, "sel": args.sel, "sel_poly": args.sel_poly,
                       "decay_exp": args.decay_exp, "input_gate": args.input_gate, "state_width": N, "bc_selective": args.bc_selective,
                       "conv": args.conv, "robust_iters": rk, "drops": {"wv": args.drop_wv, "wkr": args.drop_wkr, "wout": args.drop_wout, "win": args.drop_win}, "sched": sched.active() or ["stock"], "I_sum": sum(I_tm) + sum(I_cm) + I_out},
            "per_tick": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-depth", type=int, default=12); ap.add_argument("--dboot", type=int, default=19)
    ap.add_argument("--boot-floor", type=int, default=3); ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--dpad", type=int, default=1024); ap.add_argument("--d", type=int, default=1024); ap.add_argument("--dff", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=24); ap.add_argument("--seeds", type=str, default=CC.SEEDS)
    ap.add_argument("--lanes-block", action="store_true"); ap.add_argument("--lazy", action="store_true")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--sel", choices=("none", "diag", "proj", "lowrank"), default="none")
    ap.add_argument("--sel-poly", type=int, default=1); ap.add_argument("--decay-exp", type=int, default=0)
    ap.add_argument("--input-gate", action="store_true"); ap.add_argument("--state-width", type=int, default=1)
    ap.add_argument("--bc-selective", action="store_true"); ap.add_argument("--conv", type=int, default=2); ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--robust-iters", type=int, default=0); ap.add_argument("--force-list", action="store_true")
    ap.add_argument("--drop-wv", type=int, default=0); ap.add_argument("--drop-wkr", type=int, default=0)
    ap.add_argument("--drop-wout", type=int, default=0); ap.add_argument("--drop-win", type=int, default=0)
    ap.add_argument("--json", type=str, default=None); ap.add_argument("--sites", action="store_true"); ap.add_argument("--selfcheck", action="store_true")
    T.add_sched_args(ap)
    args = ap.parse_args()
    if args.selfcheck:
        import subprocess, tempfile
        tmp = os.path.join(tempfile.mkdtemp(), "ref.json")
        base = [sys.executable, os.path.join(os.path.dirname(HERE), "canonical_carry_trace.py"), "--handoff", "canon", "--ticks", "3", "--json", tmp]
        ok = True
        for extra in ([], ["--lanes-block"], ["--lanes-block", "--schedule", "parent-first", "--share-babies", "--late-margin"]):
            subprocess.run(base + extra, capture_output=True, text=True, check=True)
            ref = json.load(open(tmp))
            a2 = ap.parse_args(extra + ["--ticks", "3"]); mine = run(a2)
            for r, m in zip(ref["per_tick"], mine["per_tick"]):
                same = (r["boots_total"], r["rot"], r["ctpt_diag"], r["diag_limbs"]) == (m["boots_total"], m["rot"], m["ctpt_diag"], m["diag_limbs"])
                ok &= same
                print(f"selfcheck {extra or ['plain']}: tick {r['tick']} imported replay boots {r['boots_total']} rot {r['rot']} Lambda {r['diag_limbs']} | this file {m['boots_total']} {m['rot']} {m['diag_limbs']} -> {'SAME' if same else 'DIFFERENT'}")
        sys.exit(0 if ok else 1)
    out = run(args)
    p = out["params"]
    print(f"sel={p['sel']}/poly{p['sel_poly']}/exp{p['decay_exp']} igate={p['input_gate']} N={p['state_width']} BCsel={p['bc_selective']} conv={p['conv']} "
          f"robustIters={p['robust_iters']} (sum of Newton iters {p['I_sum']}) h={p['h']} sched={p['sched']} depth={p['depth']} lam={p['lam']}")
    for r in out["per_tick"]:
        print(f"tick {r['tick']}: boots {r['boots_stock_rule']} + canon {r['canon_boots']} = {r['boots_total']} | carried cts {r['carried_cts']} | rot {r['rot']} | "
              f"ct x ct {r['ctct']} | ct x pt elem {r['ctpt_elem']} | matvecs {r['matvec_chunks']} | Lambda {r['diag_limbs']} | sel: ctct {r['sel_ctct']} ctpt {r['sel_ctpt']} "
              f"rot {r['sel_rot']} matvec {r['sel_matvec']} boots at the state update {r['state_boots_in_tick']} | state end levels {r['end_levels_scan']}")
    if args.sites:
        from collections import Counter
        cnt = Counter(".".join(x.split(".")[1:]) if x.startswith("L") else x for x in out["per_tick"][-1]["boot_sites"])
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"  {v:4d}  {k}")
    if args.json:
        for r in out["per_tick"]:
            r.pop("boot_sites", None)
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
