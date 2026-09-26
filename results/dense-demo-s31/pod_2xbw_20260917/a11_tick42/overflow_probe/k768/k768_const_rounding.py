#!/usr/bin/env python3
"""ARITHMETIC (not a measurement): the relative rounding error of the scalar constant pre/(K*N) that EvalBootstrap
multiplies the raised ciphertext by (ckksrns-fhe.cpp:638, `cc->EvalMultInPlace(raised, pre * (1.0 / (k * N)))`).

OpenFHE (64-bit build) encodes a double constant c for EvalMult as the INTEGER round(c * SF) and the product carries
the scaling factor SF * SF (ckksrns-leveledshe.cpp:441-..., `large = static_cast<DoubleInteger>(operand / approxFactor *
scFactor + 0.5)`), so the constant actually applied is round(c*SF)/SF = c * (1 + eta), eta = (round(c*SF) - c*SF)/(c*SF).
The input of the approximate modular reduction is then x * (1 + eta): a coefficient with overflow I is seen as
I * (1 + eta), the sine returns ~ I * eta instead of 0, and the plaintext coefficient gets the error I * eta * 2^F.

SF at that point is the level-0 scaling factor of a FLEXIBLEAUTO context = the LAST modulus of the chain
(ckksrns-cryptoparameters.cpp:103) = FirstPrime(scalingModSize, 2N) (ckksrns-parametergeneration.cpp:419-420) = the
smallest prime 2^p + 1 + j*2N, j >= 0 (core/include/math/nbtheory-impl.h:329-347). pre = 2^-deg, deg =
round(log2(q0 / 2^p)) (ckksrns-fhe.cpp:532-545) = 1 for a 60-bit first modulus and p = 59.
FIDESlib's GPU code builds the same constant the same way (vendor/FIDESlib-K/src/CKKS/Bootstrap.cu:95-99 and :238-247,
`constantEvalMult = pre * (1.0 / (k * cc.N)); ctxt.multScalar(...)`; Context.cu:366-..., ElemForEvalMult, `large =
static_cast<DoubleInteger>(operand / approxFactor * scFactor + 0.5)`): SOURCE READING ONLY, nothing was run on a GPU, and
its scFactor at that level was not checked against OpenFHE's.
"""
from fractions import Fraction
import math


def is_prime(n):   # deterministic Miller-Rabin for n < 3.3e24
    if n < 2: return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0: return n == p
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else: return False
    return True


def first_prime(nbits, m):
    q = 1 << nbits; r = q % m; qn = q + 1 - r
    if r > 0: qn += m
    while not is_prime(qn): qn += m
    return qn


def eta(K, N, p=59, deg=1):
    SF = first_prime(p, 2 * N)
    c = Fraction(1, (1 << deg) * K * N)           # pre / (K * N), exact
    cSF = c * SF
    large = math.floor(cSF + Fraction(1, 2))       # static_cast<int128>(c*SF + 0.5)
    return SF, float(cSF), large, float((large - cSF)), float((large - cSF) / cSF)


print("scalingModSize 59, firstModSize 60 (deg = 1, pre = 1/2), FLEXIBLEAUTO; SF = FirstPrime(59, 2N)")
print(f"{'N':>8} {'K':>4} {'SF - 2^59':>12} {'c*SF':>20} {'round - c*SF':>13} {'eta = relative error of the constant':>38}"
      f" {'sigma_I (h = 2N/3)':>19} {'rms over coefficients of I*eta':>31} {'... * 2^F (F)':>22}")
for logN in (12, 13, 14, 15, 16, 17):
    N = 1 << logN
    F = {12: 11, 13: 10}.get(logN)               # measured by the probe headers at 2^12 / 2^13; else the library's default formula
    if F is None:
        F = min(14, max(7, round(-0.2419 * (2 * logN + (logN - 1)) + 19.081)))
    sig = math.sqrt((2 * N / 3 + 1) / 12)
    for K in (512, 768):
        SF, cSF, large, err, e = eta(K, N)
        print(f"{'2^' + str(logN):>8} {K:>4} {SF - (1 << 59):>12} {cSF:>20.4f} {err:>+13.4f} {e:>+38.4e} {sig:>19.2f} {abs(e) * sig:>31.3e} {abs(e) * sig * 2 ** F:>16.3e} ({F})")
print()
print("the split used by variant B (OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1): scalar constant pre/(256*N), the factor 1/3 inside the CoeffsToSlots plaintexts")
for logN in (12, 13, 17):
    N = 1 << logN
    SF, cSF, large, err, e = eta(256, N)
    print(f"{'2^' + str(logN):>8}  256 {SF - (1 << 59):>12} {cSF:>20.4f} {err:>+13.4f} {e:>+38.4e}")
