#!/usr/bin/env python3
"""The library's own alternative for the uniform-ternary bootstrap: K_UNIFORMEXT = 768 with the degree-118 table
g_coefficientsUniformExt (OpenFHE 1.5.1 selects it only for composite scaling at N >= 2^17, "to achieve a reasonable
probability of failure"). Same arithmetic as boot_overflow_estimate.py: reproduce the table inside its range, find where
it breaks outside, and price the overflow probability at N = 2^17. Also the depth bucket (ckksrns-utils.cpp GetDepthByDegree:
degree 60..119 -> depth 8) for both tables. Arithmetic on library constants, not a measurement."""
import math, re, os
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
src = open(os.path.join(REPO, "vendor/openfhe-development/src/pke/include/scheme/ckksrns/ckksrns-fhe.h")).read()
def table(name): return [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", re.search(name + r"\s*\{(.*?)\};", src, re.S).group(1))]
R = int(re.search(r"R_UNIFORM\s*=\s*(\d+)", src).group(1))
def make(coeffs, K):
    def cheb(t):
        Tkm, Tk = 1.0, t; s = coeffs[0] / 2 + coeffs[1] * t
        for c in coeffs[2:]: Tkm, Tk = Tk, 2 * t * Tk - Tkm; s += c * Tk
        return s
    def g(t):
        y = cheb(t)
        for i in range(1 - R, 1): y = 2 * y * y - (2 * math.pi) ** (-(2.0 ** i))
        return y
    return g
def Q(x): return 0.5 * math.erfc(x / math.sqrt(2))
def depth_bucket(deg):
    for hi, d in ((0,0),(1,1),(2,2),(4,3),(5,4),(13,5),(27,6),(59,7),(119,8),(247,9)):
        if deg <= hi: return d
N = 1 << 17; sig = math.sqrt((2 * N / 3 + 1) / 12)
for name, Kname in (("g_coefficientsUniform", "K_UNIFORM"), ("g_coefficientsUniformExt", "K_UNIFORMEXT")):
    c = table(name); K = int(re.search(r"\b" + Kname + r"\s*=\s*(\d+)", src).group(1)); g = make(c, K)
    inside = max(abs(g(i / K) - math.sin(2 * math.pi * i) / (2 * math.pi)) for i in range(-K, K + 1))
    inside_f = max(abs(g((i + 0.013) / K) - math.sin(2 * math.pi * (i + 0.013)) / (2 * math.pi)) for i in range(-K, K))
    thr = {}
    for I in range(K, 2 * K):
        e = abs(g(I / K))
        for t in (1e-6, 1e-4, 1e-2):
            if e > t and t not in thr: thr[t] = I
        if len(thr) == 3: break
    print(f"{name}: degree {len(c) - 1} (depth bucket {depth_bucket(len(c) - 1)} + R = {R} double angles), K = {K} = {K / sig:.2f} sigma at N = 2^17 (sigma_I {sig:.2f})")
    print(f"  reproduces (1/2pi) sin(2 pi x) inside |I| <= K to {inside:.2e} at integers, {inside_f:.2e} off-integer")
    print(f"  first overflow where the injected error exceeds 1e-6 / 1e-4 / 1e-2: {thr[1e-6]} / {thr[1e-4]} / {thr[1e-2]}  (overshoot {thr[1e-4] / K - 1:+.1%} at 1e-4)")
    for B in (K, thr[1e-4], thr[1e-2]):
        p1 = 2 * Q((B + 0.5) / sig); pb = -math.expm1(N * math.log1p(-p1)) if p1 < 1 else 1.0; pt = -math.expm1(480 * math.log1p(-pb))
        print(f"  P(some |I_j| > {B}) = {pb:.2e} per bootstrap, {pt:.2e} per 480-boot tick" + (f" (mean {1 / pt:.3g} ticks to the first event)" if pt > 0 else ""))
