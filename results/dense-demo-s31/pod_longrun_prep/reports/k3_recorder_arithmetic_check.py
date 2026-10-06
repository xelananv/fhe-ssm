#!/usr/bin/env python3
"""Check of the ARITHMETIC of the --boot-record callback (hpc_gpu_port/experimental/gpu_real_model_x.cu, struct
FheSsmNegacyclic): a line-for-line Python transliteration of the C++ (same bit-reversal table, same twiddles
exp(-2 pi i k/N), same butterfly order, same twist zeta^k = exp(i pi k/N), same centering v > (q0 >> 1) -> v - q0,
same division by q0 in double precision) compared with EXACT integer arithmetic.

It does NOT compile or run the C++; it shows the algorithm the C++ spells out gives the right I = round((c0 + c1*s)/q0).
  part 1: N = 256, 1024, 4096: every coefficient against the exact negacyclic product (Python integers).
  part 2: N = 2^17, q0 a 60-bit odd number, uniform c0/c1, uniform ternary s: 64 probed coefficients against the exact
          integer value (one O(N) integer dot product each), the fractional-part self check, and rms(I) against the
          prediction sqrt((h + 1) / 12).
The stages are vectorised with numpy for speed only; the index arithmetic is the C++ loop's.
"""
import math, random, sys
import numpy as np


class Negacyclic:
    def __init__(self, n):
        assert n >= 2 and (n & (n - 1)) == 0
        self.N = n
        self.logN = n.bit_length() - 1
        k = np.arange(n, dtype=np.float64)
        self.twRe = np.cos(math.pi * k / n); self.twIm = np.sin(math.pi * k / n)
        h = np.arange(n // 2, dtype=np.float64)
        self.wRe = np.cos(2.0 * math.pi * h / n); self.wIm = -np.sin(2.0 * math.pi * h / n)
        rev = np.zeros(n, dtype=np.int64)
        for i in range(n):  # C++: rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (logN - 1))
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (self.logN - 1))
        self.rev = rev

    def fft(self, re, im, inverse):
        n = self.N
        re[:] = re[self.rev]; im[:] = im[self.rev]        # C++: swap(i, rev[i]) for i < rev[i]  (an involution: same result)
        length = 2
        while length <= n:
            half, step = length >> 1, n // length
            wr = self.wRe[0:n // 2:step][:half]
            wi = self.wIm[0:n // 2:step][:half]
            if inverse:
                wi = -wi
            R = re.reshape(-1, length); I = im.reshape(-1, length)
            xr = R[:, half:] * wr - I[:, half:] * wi
            xi = R[:, half:] * wi + I[:, half:] * wr
            R[:, half:] = R[:, :half] - xr; I[:, half:] = I[:, :half] - xi
            R[:, :half] += xr; I[:, :half] += xi
            length <<= 1
        if inverse:
            re *= 1.0 / n; im *= 1.0 / n

    def set_secret(self, s):
        self.sRe = s.astype(np.float64) * self.twRe; self.sIm = s.astype(np.float64) * self.twIm
        self.fft(self.sRe, self.sIm, False)

    @staticmethod
    def centered_frac(v, q0):          # C++: v > halfQ ? -(double)(q0 - v) * invQ : (double)v * invQ
        half = q0 >> 1
        out = np.empty(len(v), dtype=np.float64)
        inv = 1.0 / float(q0)
        for i, x in enumerate(v):
            x %= q0
            out[i] = -float(q0 - x) * inv if x > half else float(x) * inv
        return out

    def overflow(self, c0, c1, q0):
        a = self.centered_frac(c1, q0)
        aRe = a * self.twRe; aIm = a * self.twIm
        self.fft(aRe, aIm, False)
        pr = aRe * self.sRe - aIm * self.sIm; pi = aRe * self.sIm + aIm * self.sRe
        self.fft(pr, pi, True)
        conv = pr * self.twRe + pi * self.twIm
        x = self.centered_frac(c0, q0) + conv
        r = np.rint(x)
        return r.astype(np.int64), np.abs(x - r)


def exact_coeff(c0, c1, s_idx_sign, q0, k, n):
    """exact t_k = [c0_k] + sum_j s_j [c1]_{k-j} (negacyclic), then I_k = round(t_k / q0), all in Python integers"""
    half = q0 >> 1
    cen = lambda v: v - q0 if v > half else v
    t = cen(c0[k])
    for j, sg in s_idx_sign:
        i = k - j
        if i >= 0:
            t += sg * cen(c1[i])
        else:
            t -= sg * cen(c1[i + n])
    I = (2 * t + q0) // (2 * q0)          # round to nearest (ties cannot occur: q0 is odd)
    return I, abs(t - I * q0) / q0


def run(n, q0, rng, probes=None):
    c0 = [rng.randrange(q0) for _ in range(n)]
    c1 = [rng.randrange(q0) for _ in range(n)]
    s = np.array([rng.choice((-1, 0, 1)) for _ in range(n)], dtype=np.int64)
    sidx = [(j, int(s[j])) for j in range(n) if s[j] != 0]
    ng = Negacyclic(n); ng.set_secret(s)
    I, frac = ng.overflow(c0, c1, q0)
    ks = range(n) if probes is None else [rng.randrange(n) for _ in range(probes)]
    bad = 0; worst = 0.0
    for k in ks:
        Ie, fe = exact_coeff(c0, c1, sidx, q0, k, n)
        if Ie != int(I[k]):
            bad += 1
        worst = max(worst, abs(fe - float(frac[k])))
    h = len(sidx)
    return dict(N=n, h=h, checked=len(list(ks)) if probes is None else probes, mismatches=bad,
                worst_frac_error=worst, maxI=int(np.abs(I).max()), rmsI=float(np.sqrt(np.mean(I.astype(np.float64) ** 2))),
                rms_pred=math.sqrt((h + 1) / 12.0))


def main():
    rng = random.Random(20260918)
    q0 = (1 << 60) - 93                      # an odd 60-bit modulus; primality is irrelevant to this arithmetic
    for n in (256, 1024, 4096):
        r = run(n, q0, rng)
        print(f"part 1  N = {r['N']:6d}  h = {r['h']:6d}  coefficients checked {r['checked']:6d}  I mismatches {r['mismatches']}  "
              f"worst |frac(float) - frac(exact)| {r['worst_frac_error']:.3e}  max|I| {r['maxI']}  rms I {r['rmsI']:.3f} (pred {r['rms_pred']:.3f})")
    r = run(1 << 17, q0, rng, probes=64)
    print(f"part 2  N = {r['N']:6d}  h = {r['h']:6d}  coefficients checked {r['checked']:6d}  I mismatches {r['mismatches']}  "
          f"worst |frac(float) - frac(exact)| {r['worst_frac_error']:.3e}  max|I| {r['maxI']}  rms I {r['rmsI']:.3f} (pred {r['rms_pred']:.3f})")
    print("note: c0, c1 here are uniform, so the exact fractional part is uniform too; on a real ciphertext it is the message m/q0, small."
          " The check is that the float fraction equals the exact one to ~1e-10, i.e. the double-precision FFT resolves I exactly.")


if __name__ == "__main__":
    main()
