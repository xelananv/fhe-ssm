#!/usr/bin/env python3
"""How likely is a CKKS bootstrap to fail by coefficient overflow at the demo's parameters, from the library's own
constants? (OpenFHE v1.5.1 ckksrns-fhe.h: UNIFORM_TERNARY secret -> K_UNIFORM = 512, R_UNIFORM = 6 double-angle
iterations, Chebyshev table g_coefficientsUniform on t = x/K in [-1, 1].)

After ModRaise the raised polynomial is m + q0*I with I_j = the integer overflow of coefficient j. For a uniform-ternary
secret with h nonzeros, I_j is (up to rounding) a sum of h independent uniforms on (-1/2, 1/2): sigma_I = sqrt((h+1)/12).
The approximate modular reduction is a polynomial fitted on |I| <= K only; outside, it diverges. This script
  1. evaluates the library's own approximation g(t) = DoubleAngle^R(Chebyshev(t)) at integer overflows I = K+1.. and
     prints its error against the exact sin(2*pi*I) = 0 (the message part is ~0 there): the tolerance to overshoot;
  2. prints the Gaussian tail P(max_j |I_j| > B) per bootstrap over N coefficients, for the bounds B found in 1;
  3. converts it to an expected number of ticks to the first failure at `boots` bootstraps per tick.
Nothing here is a measurement of the pod; it is arithmetic on the library's constants."""
import math, re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
H = os.path.join(REPO, "vendor/openfhe-development/src/pke/include/scheme/ckksrns/ckksrns-fhe.h")
src = open(H).read()
m = re.search(r"g_coefficientsUniform\{(.*?)\};", src, re.S); coeffs = [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", m.group(1))]
K = int(re.search(r"K_UNIFORM\s*=\s*(\d+)", src).group(1)); R = int(re.search(r"R_UNIFORM\s*=\s*(\d+)", src).group(1))
print(f"library constants: K_UNIFORM = {K}, R_UNIFORM = {R}, Chebyshev degree {len(coeffs) - 1}")
def cheb(t):   # OpenFHE's EvalChebyshevSeries convention: c0/2 + sum_{k>=1} c_k T_k(t)
    Tkm, Tk = 1.0, t; s = coeffs[0] / 2 + coeffs[1] * t
    for c in coeffs[2:]:
        Tkm, Tk = Tk, 2 * t * Tk - Tkm; s += c * Tk
    return s
def g(t):   # ckksrns-fhe.cpp ApplyDoubleAngleIterations: for i = 1-R..0: y <- 2 y^2 - (2 pi)^(-2^i)   (the scaled double angle)
    y = cheb(t)
    for i in range(1 - R, 1): y = 2 * y * y - (2 * math.pi) ** (-(2.0 ** i))
    return y
def target(x): return math.sin(2 * math.pi * x) / (2 * math.pi)   # the library's comment: the series interpolates 1/(2 Pi) Sin(2 Pi K x)
# sanity inside the interval: worst error at integer and half-integer-offset points
worst_in = max(abs(g(i / K) - target(i)) for i in range(-K, K + 1))
worst_in_frac = max(abs(g((i + 0.01) / K) - target(i + 0.01)) for i in range(-K, K))
print(f"inside |I| <= K: worst |g - sin| at integers {worst_in:.3e}; at I+0.01 {worst_in_frac:.3e} (sanity: the table reproduces the target)")
print("\noverflow I | t = I/K | |g(t) - sin(2 pi I)| (error injected into ONE coefficient, in units of the sine output)")
bounds = {}
for I in list(range(K, K + 12)) + [K + 15, K + 20, K + 26, K + 32, K + 40, K + 51]:
    e = abs(g(I / K) - target(I)); print(f"{I:6d} | {I / K:.4f} | {e:.3e}   (the message itself is ~ m/q0 <= ~1e-3 in these units)")
    for thr in (1e-6, 1e-4, 1e-2):
        if e > thr and thr not in bounds: bounds[thr] = I
print("\nfirst overflow at which the injected error exceeds a threshold:", {f"{k:g}": v for k, v in bounds.items()})
def Q(x): return 0.5 * math.erfc(x / math.sqrt(2))
for N in (1 << 17, 1 << 16):
    h = 2 * N / 3; sig = math.sqrt((h + 1) / 12)
    print(f"\nring N = {N}: uniform ternary h ~ {h:.0f}, sigma_I = {sig:.2f}, K/sigma = {K / sig:.2f}")
    for B in sorted(set([K] + list(bounds.values()))):
        p1 = 2 * Q((B + 0.5) / sig); pb = 1 - (1 - p1) ** N
        line = f"  P(|I_j| > {B}) = {p1:.2e} per coefficient -> {pb:.2e} per bootstrap"
        for boots in (480,):
            pt = 1 - (1 - pb) ** boots; line += f" -> {pt:.2e} per tick at {boots} boots (mean ticks to first event {1 / pt:.0f})" if pt > 0 else ""
        print(line)
