#!/usr/bin/env python3
"""EMULATION (not a measurement) of FIDESlib's level / pending-rescale bookkeeping in
evalChebyshevSeries + applyDoubleAngleIterations (vendor/FIDESlib-K = upstream fa972864,
src/CKKS/ApproxModEval.cu:138-366, :371-722, :724-739) under FLEXIBLEAUTO, for OpenFHE 1.5.1's two
uniform-ternary tables: g_coefficientsUniform (degree 88) and g_coefficientsUniformExt (degree 118).

What is modelled: every Ciphertext is the pair (level, NoiseLevel) and each member function changes the pair
exactly as src/CKKS/Ciphertext.cpp does (file:line in the comments): rescale :426, mult :458, square :679,
multScalar :764, multScalarNoPrecheck :747, add :164, sub :206, dropToLevel :1141, evalLinearWSumMutable :1169,
growToLevel :1278, copy :1288, adjustScaleAndLevel :1374, adjustForAddOrSub :1465, adjustForMult :1500,
3-argument mult :1114, square(src) :1132. The in-place side effects on T[] and T2[] are kept.
The polynomial divisions are done for real on the library's tables (numpy chebdiv), so the branch structure
(dc, which recursion arms exist) is the tables' own, not an assumption.

Output: (k, m) from ComputeDegreesPS, ciphertext-ciphertext multiplications, rescales, scalar multiplications used
for level adjustment, and the output (level, NoiseLevel) relative to the input level E0, for both tables and both
possible input NoiseLevels. Levels are FIDESlib levels (limb count - 1): a lower number is a deeper ciphertext.
"""
import os, re, sys
import numpy as np
from numpy.polynomial import chebyshev as C

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
HDR = os.path.join(REPO, "vendor/openfhe-development/src/pke/include/scheme/ckksrns/ckksrns-fhe.h")
src = open(HDR).read()


def table(name):
    body = re.search(name + r"\s*\{(.*?)\};", src, re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", body)]


def degree(c):
    d = len(c) - 1
    while d > 0 and c[d] == 0:
        d -= 1
    return d


def compute_degrees_ps(n):  # ckksrns-utils.cpp:299-330 (n <= 2204 branch)
    rangemap = [(2, 1), (11, 2), (13, 3), (17, 2), (55, 3), (59, 4), (76, 3), (239, 4), (247, 5), (284, 4),
                (991, 5), (1007, 6), (1083, 5), (2015, 6), (2031, 7), (2204, 6)]
    m = next(mm for hi, mm in rangemap if n <= hi)
    k = n // ((1 << m) - 1) + 1
    return k, m


def depth_by_degree(d):  # ckksrns-utils.cpp:82-110
    for hi, dep in ((0, 0), (1, 1), (2, 2), (4, 3), (5, 4), (13, 5), (27, 6), (59, 7), (119, 8), (247, 9)):
        if d <= hi:
            return dep


class Stats:
    def __init__(self):
        self.mult = 0; self.rescale = 0; self.mscalar = 0; self.wsum = 0; self.wsum_inputs = 0


ST = Stats()


class Ct:
    def __init__(self, lvl=-1, nl=1):
        self.lvl, self.nl = lvl, nl

    def clone(self):
        return Ct(self.lvl, self.nl)

    def copy(self, o):  # :1288
        if o is self:
            return
        self.lvl, self.nl = o.lvl, o.nl

    def rescale(self):  # :426
        ST.rescale += 1
        self.lvl -= 1; self.nl -= 1

    def mult_scalar_noprecheck(self):  # :747
        ST.mscalar += 1
        self.nl += 1

    def mult_scalar(self):  # :764
        if self.nl == 2:
            self.rescale()
        assert self.nl == 1
        self.mult_scalar_noprecheck()

    def drop_raw(self, level):  # skip_adjust = true
        if self.lvl > level:
            self.lvl = level

    def adjust_scale_and_level(self, deg2, lvl2):  # :1374
        c1lvl, c2lvl, c1d, c2d = self.lvl, lvl2, self.nl, deg2
        if c1lvl > c2lvl:
            if c1d == 2:
                if c2d == 2:
                    self.mult_scalar_noprecheck()
                    if self.lvl > c2lvl + 1:
                        self.drop_raw(c2lvl + 1)
                    self.rescale()
                else:
                    if c1lvl - 1 == c2lvl:
                        self.rescale()
                    else:
                        self.mult_scalar_noprecheck()
                        self.rescale()
                        if self.lvl - 1 > c2lvl:
                            self.drop_raw(c2lvl + 1)
                        self.rescale()
            else:
                if c2d == 2:
                    self.mult_scalar_noprecheck()
                    self.drop_raw(c2lvl)
                else:
                    self.mult_scalar_noprecheck()
                    if c1lvl - 1 > c2lvl:
                        self.drop_raw(c2lvl + 1)
                    self.rescale()
            assert (self.lvl, self.nl) == (c2lvl, c2d), ((self.lvl, self.nl), (c2lvl, c2d))
            return True
        if c1lvl < c2lvl:
            return False
        if c1d < c2d:
            self.mult_scalar()
        elif c2d < c1d:
            return False
        return True

    def adjust_for_add_or_sub(self, b):  # :1465 (FLEXIBLEAUTO arm)
        return self.adjust_scale_and_level(b.nl, b.lvl)

    def adjust_for_mult(self, b):  # :1500
        if self.adjust_for_add_or_sub(b):
            if self.nl == 2:
                self.rescale()
            return b.nl != 2
        if self.nl == 2:
            self.rescale()
        return False

    def mult(self, b):  # :458 (in place)
        if not self.adjust_for_mult(b):
            b_ = b.clone()
            assert b_.adjust_for_mult(self)
            self.mult(b_)
            return
        assert self.nl == 1 and b.nl == 1 and self.lvl == b.lvl
        ST.mult += 1
        self.nl = 2

    def square(self):  # :679
        if self.nl == 2:
            self.rescale()
        self.mult(self)

    def square_from(self, s):  # :1132
        if s is not self:
            self.copy(s)
        self.square()

    def mult3(self, b, c):  # :1114
        if b.lvl <= c.lvl:
            self.copy(b); self.mult(c)
        else:
            self.copy(c); self.mult(b)

    def add(self, b):  # :164 (sub :206 is the same bookkeeping)
        if not self.adjust_for_add_or_sub(b):
            b_ = b.clone()
            assert b_.adjust_for_add_or_sub(self)
            self.add(b_)
            return
        assert self.lvl == b.lvl and self.nl == b.nl, ((self.lvl, self.nl), (b.lvl, b.nl))

    sub = add

    def drop_to_level(self, level):  # :1141 skip_adjust = false
        if self.lvl > level:
            assert self.adjust_scale_and_level(self.nl, level)

    def grow_to_level(self, level):  # :1278
        if self.lvl < level:
            self.lvl = level

    def wsum(self, ctxs):  # :1169
        ST.wsum += 1; ST.wsum_inputs += len(ctxs)
        if self.lvl == -1:
            self.lvl = ctxs[0].lvl; self.nl = 1
        for c in ctxs:
            assert c.nl == 1, "evalLinearWSumMutable: input with a pending rescale"
            assert c.lvl >= self.lvl, "evalLinearWSumMutable: input shallower in limbs than the output"
        self.nl = 2


def cheb_divmod(f, g):
    q, r = C.chebdiv(np.array(f, dtype=float), np.array(g, dtype=float))
    return list(q), list(r)


def inner_ps(out, coeffs, k, m, T, T2, level_offset, max_m, trace):  # ApproxModEval.cu:138
    k2m2k = k * (1 << (m - 1)) - k
    f2 = list(coeffs) + [0.0] * max(0, 2 * k2m2k + k + 1 - len(coeffs))
    f2 = f2[: 2 * k2m2k + k + 1]
    if len(f2) > len(coeffs):
        f2[-1] = 1.0
    Tkm = [0.0] * (k2m2k + k) + [1.0]
    q, r = cheb_divmod(f2, Tkm)
    r2 = list(r)
    if k2m2k - degree(r2) <= 0:
        r2 = r2 + [0.0] * max(0, k2m2k + 1 - len(r2))
        r2[k2m2k] -= 1
        r2 = r2[: degree(r2) + 1]
    else:
        r2 = (r2 + [0.0] * (k2m2k + 1))[: k2m2k + 1]
        r2[-1] = -1.0
    cq, s = cheb_divmod(r2, q)
    s2 = (list(s) + [0.0] * (k2m2k + 1))[: k2m2k + 1]
    s2[-1] = 1.0
    cu = out
    dc = degree(cq)
    flag_c = False
    tgt = T2[m - 1].lvl + (1 if T2[m - 1].nl == 1 else 0) - level_offset
    if dc >= 1:
        if dc == 1:
            cu.copy(T[0])
            if cq[1] != 1:
                cu.mult_scalar()
        else:
            cu.drop_to_level(tgt); cu.grow_to_level(tgt)
            cu.wsum(T[:dc])
        flag_c = True
    qu = Ct()
    if degree(q) > k:
        assert m > 2
        inner_ps(qu, q, k, m - 1, T, T2, level_offset, max_m, trace)
        if qu.nl == 2:
            qu.rescale()
    else:
        qc = q[:k]
        if degree(qc) > 0:
            ctxs = [T[i] for i in range(len(q) - 1) if q[i + 1] != 0]
            qu.grow_to_level(tgt); qu.drop_to_level(tgt)
            qu.wsum(ctxs)
            if T[k - 1].nl == 1:
                qu.rescale()
            if T[k - 1].nl == 2:
                qu.rescale()
        else:
            qu.copy(T[k - 1])
            if qu.nl == 2:
                qu.rescale()
    su = Ct()
    if degree(s2) > k:
        assert m > 2
        inner_ps(su, s2, k, m - 1, T, T2, level_offset + 1, max_m, trace)
    else:
        sc = s2[:k]
        if degree(sc) > 0:
            ctxs = [T[i] for i in range(len(s2) - 1) if s2[i + 1] != 0]
            su.grow_to_level(tgt - 1); su.drop_to_level(tgt - 1)
            su.wsum(ctxs)
        else:
            su.copy(T[k - 1])
    if flag_c:
        if max_m - m <= 1:
            T2[m - 1].adjust_for_add_or_sub(cu)
        if T2[m - 1].nl == 1 and cu.nl == 2:
            cu.rescale()
        cu.add(T2[m - 1])
    else:
        cu.copy(T2[m - 1])
    cu.mult(qu)
    cu.add(su)
    trace.append((m, level_offset, dc, degree(q), degree(s2), (cu.lvl, cu.nl)))


def eval_chebyshev_series(ctxt, coeffs, trace):  # ApproxModEval.cu:371
    n = degree(coeffs)
    k, m = compute_degrees_ps(n)
    T = [Ct() for _ in range(k)]
    T2 = [Ct() for _ in range(m)]
    T[0].copy(ctxt)
    if T[0].nl == 2:
        T[0].rescale()
    for i in range(2, k + 1):
        if i % 2 == 1:
            T[i // 2].adjust_for_mult(T[i // 2 - 1])
            T[i // 2 - 1].adjust_for_mult(T[i // 2])
            T[i - 1].mult3(T[i // 2 - 1], T[i // 2])
            ctxt.adjust_for_add_or_sub(T[i - 1])
            if ctxt.nl == 1:
                T[i - 1].rescale()
            T[i - 1].sub(ctxt)
        else:
            T[i // 2 - 1].adjust_for_mult(T[i // 2 - 1])
            T[i - 1].square_from(T[i // 2 - 1])
    for i in range(1, k + 1):
        if T[i - 1].nl == 2:
            T[i - 1].rescale()
    powers = [(t.lvl, t.nl) for t in T]
    T2[0].copy(T[-1])
    for i in range(1, m):
        T2[i].square_from(T2[i - 1])
    T2km1 = Ct(); T2km1.copy(T2[0])
    for i in range(1, m):
        T2km1.mult(T2[i])
        T2[0].adjust_for_add_or_sub(T2km1)
        if T2[0].nl == 1:
            T2km1.rescale()
        T2km1.sub(T2[0])
        if T2[0].nl == 2 and i < m - 1:
            T2km1.rescale()
    inner_ps(ctxt, coeffs[: n + 1], k, m, T, T2, 0, m, trace)
    ctxt.sub(T2km1)
    return k, m, powers, [(t.lvl, t.nl) for t in T2]


def double_angle(ctxt, its):  # ApproxModEval.cu:724
    for _ in range(its):
        ctxt.square()


def main():
    R = int(re.search(r"R_UNIFORM\s*=\s*(\d+)", src).group(1))
    E0 = 100  # arbitrary input level; only differences matter
    for name, kname in (("g_coefficientsUniform", "K_UNIFORM"), ("g_coefficientsUniformExt", "K_UNIFORMEXT")):
        coeffs = table(name)
        K = int(re.search(r"\b" + kname + r"\s*=\s*(\d+)", src).group(1))
        for nl0 in (1, 2):
            global ST
            ST = Stats()
            trace = []
            ct = Ct(E0, nl0)
            k, m, powers, t2 = eval_chebyshev_series(ct, coeffs, trace)
            after_series = (ct.lvl - E0, ct.nl)
            mult_series = ST.mult
            double_angle(ct, R)
            eff_in = E0 - (nl0 - 1)
            eff_out = ct.lvl - (ct.nl - 1)
            print(f"{name}: entries {len(coeffs)}, degree {degree(coeffs)}, K = {K}, GetDepthByDegree = {depth_by_degree(degree(coeffs))}, "
                  f"ComputeDegreesPS -> k = {k}, m = {m}; input (E0, NoiseLevel {nl0})")
            print(f"  power basis T_1..T_k (level - E0, NoiseLevel): {[(l - E0, d) for l, d in powers]}")
            print(f"  T_k*2^i (level - E0, NoiseLevel) after the series: {[(l - E0, d) for l, d in t2]}")
            print(f"  series output (level - E0, NoiseLevel) = {after_series}; after {R} double angles = ({ct.lvl - E0}, {ct.nl}); "
                  f"levels consumed, settled input to settled output = {eff_in - eff_out}")
            print(f"  ct x ct multiplications: series {mult_series} + double angle {ST.mult - mult_series} = {ST.mult} per ciphertext; "
                  f"x2 ciphertexts (fully packed slots) = {2 * ST.mult} per bootstrap")
            print(f"  rescales {ST.rescale}, level-adjust scalar multiplications {ST.mscalar}, weighted sums {ST.wsum} over {ST.wsum_inputs} inputs")
            print(f"  PS recursion nodes (m, level_offset, deg c, deg q, deg s, node output (level - E0, NoiseLevel)): "
                  f"{[(a, b, c_, d, e, (f[0] - E0, f[1])) for a, b, c_, d, e, f in trace]}")


if __name__ == "__main__":
    main()
