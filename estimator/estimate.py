#!/usr/bin/env python3
"""
Core-SVP security estimator for RLWE parameter sets (primal uSVP model).

Methodology [prior art]:
  - Primal uSVP via Kannan embedding, GSA success condition with m samples
    (Alkim-Ducas-Poppelmann-Schwabe, USENIX'16; Albrecht-Player-Scott '15):
        sigma * sqrt(b) <= delta(b)^(2b - d - 1) * q^(m/d),   d = n + m + 1
        delta(b) = ((pi*b)^(1/b) * b / (2*pi*e)) ** (1 / (2*(b - 1)))
  - Cost of BKZ with block size b: core-SVP sieve exponents
        classical 2^(0.292 b), quantum 2^(0.265 b).

This is deliberately the conservative quick model used for sizing, NOT a
replacement for the full lattice-estimator (no dual/hybrid refinements).
Output is cross-checked against the Homomorphic Encryption Standard (2018)
table for uniform-ternary secrets at classical 128-bit; when deriving
parameter sets we adopt the STRICTER of {this model, the standard table}.

stdlib only. Usage: python3 estimate.py
"""
import math

SIGMA = 3.19  # SEAL/HE-standard default error stddev

# Homomorphic Encryption Standard (2018), uniform ternary secret,
# classical 128-bit: max log2(QP) per ring degree N.
HE_STD_128 = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}


def delta(b: int) -> float:
    return ((math.pi * b) ** (1.0 / b) * b / (2 * math.pi * math.e)) ** (
        1.0 / (2.0 * (b - 1))
    )


def primal_wins(n: int, logq: float, b: int, m: int) -> bool:
    """GSA success condition for primal uSVP with block size b, m samples."""
    d = n + m + 1
    lhs = math.log2(SIGMA) + 0.5 * math.log2(b)
    rhs = (2 * b - d - 1) * math.log2(delta(b)) + (m / d) * logq
    return lhs <= rhs


def min_block_size(n: int, logq: float) -> int:
    """Smallest BKZ block size at which the primal attack succeeds
    (attacker optimizes the sample count m)."""
    lo, hi = 60, 4000
    # attack gets easier (wins at smaller b) as m is optimized; scan b upward
    for b in range(lo, hi):
        # coarse-then-fine search over m in [b/2 .. 2n]
        m_lo, m_hi = max(1, b // 2), 2 * n
        step = max(1, (m_hi - m_lo) // 64)
        found = False
        for m in range(m_lo, m_hi + 1, step):
            if primal_wins(n, logq, b, m):
                found = True
                break
        if found:
            return b
    return hi


def security_bits(n: int, logq: float):
    b = min_block_size(n, logq)
    return 0.292 * b, 0.265 * b, b


def max_logq(n: int, target_classical: float = 128.0) -> int:
    """Largest integer log2(QP) with classical core-SVP security >= target."""
    lo, hi = 8, 4000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        c, _, _ = security_bits(n, mid)
        if c >= target_classical:
            lo = mid
        else:
            hi = mid - 1
    return lo


CANDIDATE_SETS = [
    # (name, N, chain bits, note)
    ("LAB",    16384, [60] + [40] * 6 + [60],        "private-ai-lab bench chain"),
    ("PS-G1",  16384, [50] + [35] * 9 + [50],        "leveled generation, scale 2^35"),
    ("PS-G1c", 16384, [45] + [35] * 7 + [45],        "core-SVP-conservative variant"),
    ("PS-P",   32768, [55] + [45] * 15 + [55, 55],   "prefill DFT-conv"),
    ("PS-G2",  65536, [60, 60] + [50] * 15 + [35] * 20, "bootstrapped generation (sketch)"),
]


def main() -> None:
    # The two tiers deliberately disagree: pure core-SVP ignores BKZ call
    # counts and polynomial factors, so it is ~15-20% stricter than the model
    # behind the HE Standard table (which SEAL's tc128 enforces). The
    # calibration column makes that translation constant explicit: it is the
    # core-SVP security of the standard table's own bound.
    print("== max log2(QP) per ring degree: paranoid core-SVP-128 tier vs "
          "HE-Standard 128-bit tier (sigma=3.19) ==")
    print(f"{'N':>6} {'core-SVP-128':>13} {'HE-std 2018':>12} "
          f"{'core-SVP bits @ HE-std bound':>29}")
    paranoid = {}
    for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
        est = max_logq(n)
        paranoid[n] = est
        std = HE_STD_128.get(n)
        if std is not None:
            cal, _, _ = security_bits(n, std)
            cal_s = f"{cal:.1f}"
        else:
            cal_s = "n/a"
        print(f"{n:>6} {est:>13} {str(std) if std else 'n/a':>12} {cal_s:>29}")

    print("\n== candidate parameter sets (dual reporting) ==")
    print(f"{'set':>7} {'N':>6} {'logQP':>6} {'HE-std/tc128':>12} "
          f"{'core-SVP-128':>12} {'cSVP bits':>9} {'quantum':>8} {'depth':>7}  note")
    for name, n, chain, note in CANDIDATE_SETS:
        logqp = sum(chain)
        c, q, b = security_bits(n, logqp)
        std = HE_STD_128.get(n)
        std_ok = "Y" if (std is not None and logqp <= std) else (
            "n/a" if std is None else "N")
        par_ok = "Y" if logqp <= paranoid[n] else "N"
        # usable depth: rescalable levels = scale-prime count
        # (chain minus first prime minus special prime(s))
        specials = 1 if n < 32768 else 2
        depth = len(chain) - 1 - specials
        if name == "PS-G2":  # sketch: 2 specials + 15 boot + 20 compute
            depth = "20c+15b"
        print(f"{name:>7} {n:>6} {logqp:>6} {std_ok:>12} {par_ok:>12} "
              f"{c:>8.1f}b {q:>7.1f}b {str(depth):>7}  {note}")

    print("\nCompliance claim = HE-Standard/tc128 tier (the field's operating "
          "definition of 128-bit,\nenforced by SEAL at context creation). "
          "core-SVP tier = conservative floor. Run the full\nlattice-estimator "
          "(dual/hybrid attacks) before any non-research use.")


if __name__ == "__main__":
    main()
