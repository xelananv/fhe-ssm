#!/usr/bin/env python3
"""Deterministic level-trace and operation-count model of ONE TICK of the
encrypted forward pass in hpc_gpu_port/gpu_real_model.cu (T = 1 ciphertext,
block layout, stateful carry). Companion to
results/theory/TICK_SCALING_BOUNDS_20260903.md and, for the schedule levers,
results/theory/SCHEDULE_DESIGN_20260903.md.

THIS IS NOT A MEASUREMENT. Every rule below transcribes one statement of the
harness (file:line cited beside it); the script replays those statements and
counts bootstraps, rotations and multiplies. It reads no record under
results/. Its output is a *derived* quantity in the sense of the repository conventions R5.

Level semantics (gpu_real_model.cu:1271-1275, TrackedCt::level() ==
ct->GetLevel()):
  * ct x ct followed by RescaleInPlace       -> +1 level          (:2919, :2973, :2985, :2992)
  * ct x pt followed by RescaleInPlace       -> +1 level          (:2184-2186 mulPt)
  * EvalAdd of two cts                       -> max of the two    (:2246)
  * EvalAdd of ct and pt, EvalRotate         -> unchanged
  * alignTo(c, t): mult-by-ones until level == t (+1 each)        (:2191-2195)
  * refresh(c, need): boot iff depth - level < need + F           (:2156-2182)
  * boot(): level := depth - LAM (LAM = levels usable after boot) (:2119-2135,
    :954-955: depth = levelsAfter + 19 + extraDepth, levelsAfter = 10)

Two accounting models are provided (see the .md, assumption A3):
  eager: every multiply consumes its level immediately (the harness's own
         bookkeeping; what a GPU library with an explicit rescale would do);
  lazy : OpenFHE FLEXIBLEAUTO semantics -- RescaleInPlace is a no-op
         (rns-leveledshe.cpp:354-358), the level is consumed at the NEXT
         multiply, and GetLevel() therefore reports one level LESS than the
         effective level for a ciphertext with a pending rescale
         (noiseScaleDeg == 2, ckksrns-leveledshe.cpp:603-745).
In both models the *effective* level e obeys the eager rules; only the
value that refresh()/alignTo() read differs (e vs e-(deg-1)).

SCHEDULE LEVERS (S3.3, 2026-09-03). Every lever defaults OFF; with all of
them off this file replays exactly the stock harness (verified against the
pre-edit copy: identical per-tick counts and boot-site histogram, see
SCHEDULE_DESIGN_20260903.md section 2). A lever changes ONLY the placement
of refresh()/boot() calls, never an arithmetic statement -- except
--newton-depth2, which re-associates the SAME Newton polynomial (Tier A-2)
and is therefore subject to the fidelity gate.
  --schedule parent-first   refresh the residual Hs before each norm so that
                            the norm's y chain fits without a mid-loop boot
                            (P1), lift the norm output at the tail so that the
                            branch below it needs no child boot (P2), and
                            guard the shared FFN input u2 once instead of the
                            2K children (P4). Rules: SCHEDULE_DESIGN section 3.
  --pf-entry MODE           P1 entry-need formula: loop (default) | branch |
                            both | always.
  --no-pf-tail              disable P2 (diagnostic).
  --share-babies            baby rotations computed once per input and reused
                            across the 2K wk/wr chunk calls (bit-identical).
  --late-margin             drop refresh(hd, margin) before wv (:4447) and run
                            wv at hd's current level with --bsgs-chunk
                            (bsgsRescaleInner, :578 -- no level effect); the
                            accumulated ffn is guarded once by the residual's
                            addAligned (:4511).
  --newton-depth2           y' = 1.5y + ((-0.5 ms) y)(y y): depth 2 per
                            Newton iteration instead of 4 (same polynomial).
  --pool-levels N           compPool LRU capacity (:1352-1364, stock 2);
                            counted as pool (re)builds per tick.
"""
import argparse
import json
import math
from dataclasses import dataclass, field


@dataclass
class Counts:
    boots: int = 0
    rot: int = 0          # EvalRotate calls (each a key switch)
    ctpt_diag: int = 0    # ct x pt multiplies against weight diagonals (matvec)
    ctpt_elem: int = 0    # ct x pt multiplies against per-channel vectors / constants
    ctct: int = 0         # ct x ct multiplies
    rescale: int = 0      # rescale-consuming operations (eager accounting)
    add: int = 0          # ct + ct additions
    matvec_chunks: int = 0
    h2d_words_per_limb: int = 0   # compressed store words uploaded, per limb (sum over diag-halves of N/REP)
    diag_limbs: int = 0           # sum over diagonal-halves of the operand's live limb count (depth+1-lvl)
    rot_limbs: int = 0            # sum over rotations of the operand's live limb count
    boot_lvlpre_sum: int = 0      # sum of levels at which boots fired (diagnostic)
    refresh_calls: int = 0        # refresh() call sites executed (each boots at most once)
    # ---- S3.3 additions (all zero on the stock path except pool_builds, which
    #      counts the stock LRU-2 pool as well) ----
    rot_shared: int = 0           # baby rotations SKIPPED by --share-babies
    pool_builds: int = 0          # compPool (re)builds under the LRU of --pool-levels
    diag_limbs_wv: int = 0        # Lambda restricted to the wv matvecs
    diag_limbs_wkr: int = 0       # Lambda restricted to wk + wr
    boots_parent: int = 0         # boots fired by the parent-first rules (P1/P2/P4)


@dataclass
class Sched:
    schedule: str = "stock"       # stock | parent-first
    pf_entry: str = "loop"        # loop | branch | both | always
    pf_tail: bool = True
    share_babies: bool = False
    late_margin: bool = False
    newton_depth2: bool = False
    pool_levels: int = 2
    boot_deg2: bool = False       # lazy model only: EvalBootstrap output has a pending rescale (CPU-observed)

    @property
    def pf(self):
        return self.schedule == "parent-first"

    def active(self):
        """Names of the levers that are ON (for the store-file stamp analogue)."""
        out = []
        if self.pf:
            out.append(f"schedule=parent-first/{self.pf_entry}{'' if self.pf_tail else '/notail'}")
        if self.share_babies:
            out.append("share-babies")
        if self.late_margin:
            out.append("late-margin")
        if self.newton_depth2:
            out.append("newton-depth2")
        if self.pool_levels != 2:
            out.append(f"pool-levels={self.pool_levels}")
        if self.boot_deg2:
            out.append("boot-deg2")
        return out


class Ct:
    __slots__ = ("e", "deg")

    def __init__(self, e=0, deg=1):
        self.e = e      # effective (eager) level
        self.deg = deg  # noiseScaleDeg in the lazy model (1 or 2)

    def copy(self):
        return Ct(self.e, self.deg)


class Sim:
    def __init__(self, depth, lam, F, margin, Dpad, d, dff, lanes_block, shift_mix,
                 lazy, alpha_res_is_one=True, count_matvec_dense=True, parallel_block=False,
                 sched=None):
        # parallel_block: ARCHITECTURE VARIANT (not the harness): one norm per
        # layer feeds both branches, h' = h + tm(norm h) + cm(norm h). Used
        # only to price the lever in the report's follow-up; the harness's
        # sequential block is the default.
        self.parallel_block = parallel_block
        self.s = sched if sched is not None else Sched()
        self.depth = depth
        self.lam = lam
        self.F = F
        self.margin = margin
        self.Dpad = Dpad
        self.d = d
        self.dff = dff
        self.K = math.ceil(dff / Dpad)                       # cm chunks (:2835-2840)
        self.BS = math.ceil(math.sqrt(Dpad))                 # :1067
        self.G = math.ceil(Dpad / self.BS)                   # giant steps (:2719 loop)
        self.lanes_block = lanes_block
        self.h = 2 if lanes_block else 1                     # diagonal halves (:2617)
        self.shift_mix = shift_mix
        self.lazy = lazy
        self.alpha_one = alpha_res_is_one
        self.c = Counts()
        self.trace = []
        self.probe = {}                                      # site -> list of "remaining" levels seen
        self.matvec_levels = []                              # reported operand level of every matvec call (pool LRU input)
        self.store_keys = []                                 # (layer, op, chunk, lvl) of every matvec call this tick
        self._pool = []                                      # LRU list of resident pool levels (most recent last)
        self._baby_key = None                                # (--share-babies) identity of the cached baby set
        self._baby_lvl = None
        self.trace_levels = False                            # --trace-levels: emit every refresh/boot as JSON
        self.level_lines = []

    def rem(self, a):
        return self.depth - self.rep(a)

    def note(self, site, a):
        self.probe.setdefault(site, []).append(self.rem(a))

    # ---- level primitives -------------------------------------------------
    def rep(self, a):                      # what GetLevel() reports
        return a.e - (a.deg - 1) if self.lazy else a.e

    def mul_cc(self, a, b):                # EvalMult(ct,ct)+RescaleInPlace
        r = Ct(max(a.e, b.e) + 1, 2 if self.lazy else 1)
        self.c.ctct += 1
        self.c.rescale += 1
        return r

    def mul_pt(self, a, elem=True):        # mulPt / EvalMult(ct,pt)+RescaleInPlace
        a.e += 1
        a.deg = 2 if self.lazy else 1
        if elem:
            self.c.ctpt_elem += 1
        self.c.rescale += 1

    def add(self, a, b):                   # a += b (EvalAdd), FLEXIBLEAUTO reconciliation
        la, lb = self.rep(a), self.rep(b)
        if la > lb:
            deg = a.deg
        elif lb > la:
            deg = b.deg
        else:
            deg = max(a.deg, b.deg)
        a.e = max(a.e, b.e)
        a.deg = deg if self.lazy else 1
        self.c.add += 1

    def boot(self, a, site):
        self.c.boot_lvlpre_sum += self.rep(a)
        pre = self.rep(a)
        if self.lazy and self.s.boot_deg2:
            a.e = self.depth - self.lam + 1     # reported depth-lam, one pending rescale (OpenFHE EvalBootstrap output)
            a.deg = 2
        else:
            a.e = self.depth - self.lam
            a.deg = 1
        self.c.boots += 1
        self.trace.append(site)
        if self.trace_levels:
            self.level_lines.append({"pfTrace": "boot", "site": site, "lvlPre": pre, "lvlPost": self.rep(a),
                                     "remainingAfter": self.depth - self.rep(a)})

    def limbs(self, a):                    # live RNS limbs of a at its reported level (:1371)
        return self.depth + 1 - self.rep(a)

    def refresh(self, a, need, site):      # :2156-2182
        self.c.refresh_calls += 1
        remaining = self.depth - self.rep(a)
        if self.trace_levels:
            self.level_lines.append({"pfTrace": "refresh", "site": site, "level": self.rep(a), "remaining": remaining})
        if remaining < need + self.F:
            self.boot(a, site)
            return True
        return False

    def refresh_pf(self, a, need, site):   # a parent-first placement of the SAME refresh rule
        if self.refresh(a, need, site):
            self.c.boots_parent += 1

    def align_to(self, a, target):         # :2191-2195
        while self.rep(a) < target:
            self.mul_pt(a)

    def add_aligned(self, a, b, site):     # :2197-2246
        self.refresh(a, 2, site + ".a")
        self.refresh(b, 2, site + ".b")
        self.add(a, b)

    def check(self, a, where):
        if self.rep(a) > self.depth:
            raise RuntimeError(f"level_overflow at {where}: {self.rep(a)} > {self.depth}")

    # ---- parent-first need formulas (SCHEDULE_DESIGN section 3) -----------
    def nb(self):
        """Levels from the norm input to the birth of y: sq, [e0], 1/d, b0 (Lemma 4)."""
        return 4 if self.lanes_block else 3

    def dN(self):
        """Levels one Newton iteration consumes on y."""
        return 2 if self.s.newton_depth2 else 4

    def need_entry_loop(self, iters):
        """P1 (loop): rem(Hs) such that refresh(y,5) at iteration I-1 does not fire.
        y at iteration k sits nb + dN*k below Hs; the check needs rem >= 5 + F."""
        return self.nb() + self.dN() * max(iters - 1, 0) + 5

    def need_tail(self, kind):
        """P2: rem(y) at the norm tail such that no guard below the norm output fires.
        out = x*y then gain: 2 levels below y when y is the deeper operand.
        tm : shift 1 + win 1 + b(.)x 1 + readout 1 + gate 3 + wout 1 = 8 below out,
             then the residual's addAligned guard (need 2)      -> 2 + 8 + 2 = 12
        cm : wk 1 + hidden 4 -> hd, then refresh(hd, margin) (:4447) or, under
             --late-margin, the ffnAccum guard on part = hd + 1 (need 2)
                                                                -> 2 + 1 + 4 + m  |  2 + 1 + 4 + 1 + 2
        out: nothing below but the client decode                -> 4 (stock value)"""
        if kind == "tm":
            return 12
        if kind == "cm":
            return 2 + 1 + 4 + (self.margin if not self.s.late_margin else 3)
        return 4

    def need_entry(self, kind, iters):
        mode = self.s.pf_entry
        loop = self.need_entry_loop(iters)
        # 'branch': the tail lift alone rescues the branch iff rem(Hs) - 2 clears
        # the branch need, i.e. rem(Hs) >= need_tail + 2 (+F via refresh).
        branch = self.need_tail(kind) + 2
        if mode == "loop":
            return loop
        if mode == "branch":
            return branch
        if mode == "both":
            return max(loop, branch)
        if mode == "always":
            return self.depth + 1          # rem is never >= depth + 1 + F
        raise ValueError(mode)

    def need_u2(self):
        """P4: rem(u2) such that the wk/wr children need no boot before hd/wv."""
        return self.need_tail("cm") - 2 + 1

    # ---- circuit blocks --------------------------------------------------
    def matvec(self, u, rows, in_cols, op="win", share_key=None, layer=-1, chunk=0):
        """matvecBatchImpl (:2393-2835), n = 1, dense weights, store path.
        Level: one rescale at the end (bsgsRescaleInner = false, :578) -> +1.
        Rotations: (BS-1) babies per half + 1 wrap under --lanes-block
        (:2709-2715, :2464-2468), G-1 giants (:2750, :2817).
        --share-babies: the baby set (and the wrap set) of an input is computed
        once per (identity, level) and reused by later calls with the same
        share_key (the 2K wk/wr chunk calls of one layer, :4383-4387)."""
        self.check(u, "matvec input")
        BS, G, h, Dpad = self.BS, self.G, self.h, self.Dpad
        lvl = self.rep(u)
        n_baby = h * (BS - 1) + (h - 1)
        n_giant = G - 1
        shared = (self.s.share_babies and share_key is not None
                  and self._baby_key == share_key and self._baby_lvl == lvl)
        if self.s.share_babies and share_key is not None and not shared:
            self._baby_key, self._baby_lvl = share_key, lvl
        n_rot = (0 if shared else n_baby) + n_giant
        if shared:
            self.c.rot_shared += n_baby
        self.c.rot += n_rot
        self.c.rot_limbs += n_rot * self.limbs(u)
        n_diag = h * Dpad - (h - 1)           # k = 0 has an empty hi half
        self.c.ctpt_diag += n_diag
        self.c.h2d_words_per_limb += n_diag   # x N/REP words each (:1350, :2676)
        self.c.diag_limbs += n_diag * self.limbs(u)
        if op == "wv":
            self.c.diag_limbs_wv += n_diag * self.limbs(u)
        elif op in ("wk", "wr"):
            self.c.diag_limbs_wkr += n_diag * self.limbs(u)
        self.c.add += n_diag - 1 + (G - 1)
        self.c.matvec_chunks += 1
        self.c.rescale += 1
        self.note(f"mv.{op}@in", u)
        # compressed-store pool: one resident level set per operand level (:1352-1364)
        self.matvec_levels.append(lvl)
        self.store_keys.append((layer, op, chunk, lvl))     # CompKey prefix (:1317-1324): value identity incl. level
        if lvl in self._pool:
            self._pool.remove(lvl)
        else:
            self.c.pool_builds += 1
            while len(self._pool) >= self.s.pool_levels:
                self._pool.pop(0)
        self._pool.append(lvl)
        out = Ct(u.e + 1, 2 if self.lazy else 1)
        return out

    def rmsnorm(self, x, iters, site, kind="out"):
        """:2857-3013. x is passed BY VALUE (the caller's ct is not moved).
        kind in {tm, cm, out} selects the parent-first tail need (P2)."""
        self.check(x, site + " input")
        sq = self.mul_cc(x, x)                                   # :2919
        lg = int(math.log2(self.Dpad))
        self.c.rot += lg                                         # up-tree :2923
        self.c.rot_limbs += lg * self.limbs(sq)
        red = sq
        if self.lanes_block:                                     # :2926-2940
            self.mul_pt(red)                                     # e0 mask (+1)
            self.c.rot += lg                                     # down-tree
            self.c.rot_limbs += lg * self.limbs(red)
        ms = red.copy(); self.mul_pt(ms)                         # :2946 (1/d)
        y = ms.copy(); self.mul_pt(y)                            # :2951 (b0)
        # + a0 plaintext: no level
        self.note("y0@" + site.split(".", 1)[-1], y)
        if self.s.newton_depth2:
            msh = ms.copy(); self.mul_pt(msh)                    # (-0.5) ms, once per norm
        for it in range(iters):
            self.refresh(y, 5, f"{site}.newton{it}.y")           # :2971
            if self.s.newton_depth2:
                self.refresh(msh, 5, f"{site}.newton{it}.ms")
                a = self.mul_cc(msh, y)                          # (-0.5 ms) y
                b = self.mul_cc(y, y)                            # y y
                t = self.mul_cc(a, b)                            # -0.5 ms y^3
                y15 = y.copy(); self.mul_pt(y15)                 # 1.5 y
                self.add(y15, t)                                 # y' = 1.5y - 0.5 ms y^3
                y = y15
            else:
                self.refresh(ms, 5, f"{site}.newton{it}.ms")
                y2 = self.mul_cc(y, y)                           # :2973
                tms = self.mul_cc(ms, y2)                        # :2985
                self.mul_pt(tms)                                 # :2988 (-0.5)
                # + 1.5 plaintext
                y = self.mul_cc(y, tms)                          # :2992
        if self.s.pf and self.s.pf_tail:
            self.refresh_pf(y, self.need_tail(kind), f"{site}.tail.y")   # P2 (need >= stock 4)
        else:
            self.refresh(y, 4, f"{site}.tail.y")                 # :3002
        out = self.mul_cc(x, y)                                  # :3004
        self.mul_pt(out)                                         # :3008/:3010 (gain)
        return out

    def poly_gate3(self, z):
        """:3921-3960: g = z*(p0 + p1 z + p2 z^2)."""
        z2 = self.mul_cc(z, z)
        term2 = z2.copy(); self.mul_pt(term2)
        term1 = z.copy(); self.mul_pt(term1)
        self.align_to(term1, self.rep(term2))
        self.add(term1, term2)
        zA = z.copy()
        self.align_to(zA, self.rep(term1))
        g = self.mul_cc(zA, term1)
        return g

    def layer(self, l, Hs, scan, uPrev, I_tm, I_cm):
        # ---- time mix (:4037-4370) ----
        self.note("Hs@tm.entry", Hs)
        if self.s.pf:
            self.refresh_pf(Hs, self.need_entry("tm", I_tm), f"L{l}.tm.rmsnorm.Hs")   # P1
        else:
            self.refresh(Hs, 3, f"L{l}.tm.rmsnorm.Hs")                  # :4044
        u = self.rmsnorm(Hs, I_tm, f"L{l}.tm.rmsnorm", kind="tm")
        self.note("u@tm.norm.out", u)
        u_norm = u.copy()                                               # parallel-block variant only
        if self.shift_mix:                                              # :4073-4127
            a = u.copy(); self.mul_pt(a)                                # u*mix
            if uPrev[l] is not None:
                b = uPrev[l].copy()
                self.refresh(b, 2, f"L{l}.tm.shiftMix.uPrev")           # :4117
                self.mul_pt(b)
                self.add_aligned(a, b, f"L{l}.tm.shiftMix")
            uPrev[l] = u.copy()                                         # :4126 (pre-shift u)
            u = a
        x = self.matvec(u, self.d, self.d, op="win", layer=l)           # win :4133
        # scan, single-token branch (:4292-4300)
        self.refresh(scan[l], 3, f"L{l}.tm.scan.state")
        self.refresh(x, 3, f"L{l}.tm.scan.x")
        self.mul_pt(scan[l])                                            # decay (.) s
        bx = x.copy(); self.mul_pt(bx)                                  # b (.) x
        self.add_aligned(scan[l], bx, f"L{l}.tm.scan")
        S = scan[l].copy()
        # gate (:4316-4323)
        zc = S.copy(); self.mul_pt(zc)
        zd = x.copy(); self.mul_pt(zd)
        self.add_aligned(zc, zd, f"L{l}.tm.gate.readout")
        self.refresh(zc, 5, f"L{l}.tm.gate.zc")
        g = self.poly_gate3(zc)
        attn = self.matvec(g, self.d, self.d, op="wout", layer=l)       # wout :4354
        if not self.alpha_one:
            self.mul_pt(attn)                                           # :4361-4366
        if not self.parallel_block:
            self.add_aligned(Hs, attn, f"L{l}.tm.residual")             # :4367
        # ---- channel mix (:4372-4519) ----
        if self.parallel_block:
            u2 = u_norm                                                 # variant: shared norm output
        else:
            self.note("Hs@cm.entry", Hs)
            if self.s.pf:
                self.refresh_pf(Hs, self.need_entry("cm", I_cm), f"L{l}.cm.rmsnorm.Hs")   # P1
            else:
                self.refresh(Hs, 3, f"L{l}.cm.rmsnorm.Hs")             # :4375
            u2 = self.rmsnorm(Hs, I_cm, f"L{l}.cm.rmsnorm", kind="cm")
            self.note("u2@cm.norm.out", u2)
        if self.s.pf:
            self.refresh_pf(u2, self.need_u2(), f"L{l}.cm.u2")          # P4 (shared input guard)
        key = ("u2", l)
        kks = [self.matvec(u2, min(self.Dpad, self.dff - ch * self.Dpad), self.d, op="wk", share_key=key, layer=l, chunk=ch)
               for ch in range(self.K)]                                 # wk :4383
        rrs = [self.matvec(u2, min(self.Dpad, self.dff - ch * self.Dpad), self.d, op="wr", share_key=key, layer=l, chunk=ch)
               for ch in range(self.K)]                                 # wr :4387
        self._baby_key = None                                           # share window closes (:4389)
        ffn = None
        for ch in range(self.K):
            kk, rr = kks[ch].copy(), rrs[ch].copy()
            self.note("kk@birth", kk)
            self.refresh(kk, 5, f"L{l}.cm.hidden.ch{ch}.kk")             # :4399
            self.refresh(rr, 4, f"L{l}.cm.hidden.ch{ch}.rr")
            k2 = self.mul_cc(kk, kk)
            actT = kk.copy(); self.mul_pt(actT)                          # q1
            t2 = k2.copy(); self.mul_pt(t2)                              # q2
            self.add_aligned(actT, t2, f"L{l}.cm.hidden.ch{ch}.act12")
            kkA = kk.copy(); self.align_to(kkA, self.rep(k2))
            k3 = self.mul_cc(k2, kkA); self.mul_pt(k3)                   # q3
            self.add_aligned(actT, k3, f"L{l}.cm.hidden.ch{ch}.act3")
            # + q0 plaintext
            r2 = self.mul_cc(rr, rr)
            gt = rr.copy(); self.mul_pt(gt)                              # r1
            g2t = r2.copy(); self.mul_pt(g2t)                            # r2
            self.add_aligned(gt, g2t, f"L{l}.cm.hidden.ch{ch}.gate")
            # + r0 plaintext
            self.refresh(actT, 2, f"L{l}.cm.hidden.ch{ch}.actT")
            self.refresh(gt, 2, f"L{l}.cm.hidden.ch{ch}.gt")
            uu = max(self.rep(actT), self.rep(gt))
            self.align_to(actT, uu); self.align_to(gt, uu)
            hd = self.mul_cc(actT, gt)
            self.refresh(hd, 2, f"L{l}.cm.hidden.ch{ch}.hd")             # :4427
            if not self.s.late_margin:
                self.refresh(hd, self.margin, f"L{l}.cm.wv.ch{ch}.margin")   # :4447
            self.note("hd@wv.in", hd)
            part = self.matvec(hd, self.d, self.Dpad, op="wv", layer=l, chunk=ch)   # wv :4451
            if ffn is None:
                ffn = part
            else:
                self.add_aligned(ffn, part, f"L{l}.cm.ffnAccum.ch{ch}")   # :4504
        if self.parallel_block:
            self.add_aligned(Hs, attn, f"L{l}.tm.residual")             # variant: both adds at the end
        self.add_aligned(Hs, ffn, f"L{l}.cm.residual")                  # :4511

    def tick(self, L, I_tm, I_cm, I_out, scan, uPrev, final_boot):
        Hs = Ct(0, 1)                                                   # fresh client ct (:1786-1790, :3916)
        for l in range(L):
            self.layer(l, Hs, scan, uPrev, I_tm[l], I_cm[l])
        if self.s.pf:
            self.refresh_pf(Hs, self.need_entry("out", I_out), "output.finalNorm.Hs")   # P1 at the output norm
        hf = self.rmsnorm(Hs, I_out, "output.finalNorm", kind="out")    # :4627/:4671 (no refresh before)
        if final_boot:
            self.boot(hf, "output.finalRefresh")                        # :4628/:4672
        return hf


def state_signature(scan, uPrev):
    return tuple((c.e, c.deg) for c in scan) + tuple((-1, -1) if u is None else (u.e, u.deg) for u in uPrev)


def run(args):
    depth = 10 + 19 + args.extra_depth if args.depth is None else args.depth
    lam = depth - args.dboot if args.lam is None else args.lam
    L = args.layers
    if args.seeds:
        s = json.load(open(args.seeds))
        I_tm = [s[f"L{i}.tm"]["iters"] for i in range(L)]
        I_cm = [s[f"L{i}.cm"]["iters"] for i in range(L)]
        I_out = s["ln_out"]["iters"]
    else:
        I_tm = [args.iters] * L
        I_cm = [args.iters] * L
        I_out = args.iters
    sched = Sched(schedule=getattr(args, "schedule", "stock"),
                  pf_entry=getattr(args, "pf_entry", "loop"),
                  pf_tail=not getattr(args, "no_pf_tail", False),
                  share_babies=getattr(args, "share_babies", False),
                  late_margin=getattr(args, "late_margin", False),
                  newton_depth2=getattr(args, "newton_depth2", False),
                  pool_levels=getattr(args, "pool_levels", 2),
                  boot_deg2=getattr(args, "boot_deg2", False))
    sim = Sim(depth, lam, args.boot_floor, args.margin, args.dpad, args.d, args.dff,
              args.lanes_block, args.shift_mix, args.lazy, alpha_res_is_one=True,
              parallel_block=getattr(args, "parallel_block", False), sched=sched)
    sim.trace_levels = getattr(args, "trace_levels", False)
    scan = [Ct(0, 1) for _ in range(L)]        # cold start: encCh(zeros) at level 0 (:4035)
    uPrev = [None] * L
    per_tick = []
    sigs = {}
    period = None
    for t in range(args.ticks):
        c0 = Counts(**vars(sim.c))
        sim.trace = []
        sim.store_keys = []
        sim.tick(L, I_tm, I_cm, I_out, scan, uPrev, args.final_boot)
        dc = {k: getattr(sim.c, k) - getattr(c0, k) for k in vars(sim.c)}
        dc["tick"] = t
        dc["boot_sites"] = sim.trace
        dc["store_keys"] = sim.store_keys
        per_tick.append(dc)
        sig = state_signature(scan, uPrev)          # the carried state AFTER tick t (I1: periodic?)
        if sig in sigs and period is None:
            period = (sigs[sig], t)                 # ticks sigs[sig]+1 .. t repeat forever
        sigs.setdefault(sig, t)
    probe = {}
    for k, v in sim.probe.items():
        vv = v[len(v) // 3:]                       # warm part of the run
        probe[k] = {"n": len(vv), "mean_remaining": sum(vv) / len(vv), "min": min(vv), "max": max(vv)}
    out = {
        "probe": probe,
        "params": {"depth": depth, "lam": lam, "F": args.boot_floor, "margin": args.margin,
                   "Dpad": args.dpad, "d": args.d, "dff": args.dff, "K": sim.K, "BS": sim.BS,
                   "G": sim.G, "lanes_block": args.lanes_block, "h": sim.h,
                   "shift_mix": args.shift_mix, "lazy": args.lazy, "L": L,
                   "I_tm": I_tm, "I_cm": I_cm, "I_out": I_out, "final_boot": args.final_boot,
                   "sched": vars(sched), "sched_active": sched.active()},
        "period": period,                          # (first tick of the cycle, tick where it recurred) or None
        "per_tick": per_tick,
        "level_lines": sim.level_lines,           # --trace-levels
    }
    return out


def add_sched_args(ap):
    ap.add_argument("--schedule", choices=("stock", "parent-first"), default="stock")
    ap.add_argument("--pf-entry", choices=("loop", "branch", "both", "always"), default="loop")
    ap.add_argument("--no-pf-tail", action="store_true")
    ap.add_argument("--share-babies", action="store_true")
    ap.add_argument("--late-margin", action="store_true")
    ap.add_argument("--newton-depth2", action="store_true")
    ap.add_argument("--pool-levels", type=int, default=2)
    ap.add_argument("--boot-deg2", action="store_true",
                    help="lazy model: the bootstrap output carries a pending rescale (OpenFHE 1.5.1 CPU, observed by harness/cpu_real_model.cpp --cell pf-trace)")
    ap.add_argument("--trace-levels", action="store_true",
                    help="print one JSON line per refresh()/boot() in program order (compare with harness/cpu_real_model.cpp --cell pf-trace)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--extra-depth", type=int, default=12)
    ap.add_argument("--dboot", type=int, default=19, help="levels the bootstrap consumes; lam = depth - dboot")
    ap.add_argument("--lam", type=int, default=None)
    ap.add_argument("--boot-floor", type=int, default=3)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--dpad", type=int, default=1024)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--dff", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--lanes-block", action="store_true")
    ap.add_argument("--shift-mix", action="store_true")
    ap.add_argument("--lazy", action="store_true")
    ap.add_argument("--final-boot", action="store_true")
    ap.add_argument("--parallel-block", action="store_true", help="ARCHITECTURE VARIANT: one norm per layer feeds both branches")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--sites", action="store_true", help="print boot sites of the last tick")
    add_sched_args(ap)
    args = ap.parse_args()
    out = run(args)
    p = out["params"]
    print(f"depth={p['depth']} lam={p['lam']} F={p['F']} margin={p['margin']} Dpad={p['Dpad']} "
          f"d={p['d']} dff={p['dff']} K={p['K']} BS={p['BS']} G={p['G']} h={p['h']} "
          f"lanesBlock={p['lanes_block']} shiftMix={p['shift_mix']} lazy={p['lazy']} L={p['L']} "
          f"I_tm={p['I_tm'][:3]}.. I_cm={p['I_cm'][:3]}.. finalBoot={p['final_boot']} "
          f"sched={p['sched_active'] or ['stock']} period={out['period']}")
    for t in out["per_tick"]:
        print(f"tick {t['tick']}: boots={t['boots']} rot={t['rot']} ctpt_diag={t['ctpt_diag']} "
              f"ctpt_elem={t['ctpt_elem']} ctct={t['ctct']} rescale={t['rescale']} add={t['add']} "
              f"chunks={t['matvec_chunks']} diag_limbs={t['diag_limbs']} pool_builds={t['pool_builds']}")
    if args.sites:
        print("level probe (remaining levels = depth - GetLevel(), warm part of the run):")
        for k, v in out["probe"].items():
            print(f"  {k:20s} n={v['n']:5d} mean {v['mean_remaining']:6.2f} min {v['min']:3d} max {v['max']:3d}")
        from collections import Counter
        cnt = Counter(s.split(".", 1)[1] if s.startswith("L") else s for s in out["per_tick"][-1]["boot_sites"])
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"  {v:4d}  {k}")
    if getattr(args, "trace_levels", False):
        for ln in out["level_lines"]:
            print(json.dumps(ln))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
