# run_cachemir_1763.py — estimate Cachemir's published parameter point.
#
# Cachemir (arXiv:2602.11470v1, Sec. 7.1 "Cryptographic Configuration")
# states: "we set the polynomial degree to N'=2^16 ... with a 1763-bit
# ciphertext modulus, which achieves 128-bit security according to the
# Homomorphic Encryption Standard (Albrecht et al., 2022)" and separately
# uses "a sparse secret key (Hamming weight 192)" with sparse secret
# encapsulation (Bossuat et al., 2022). The published HE Standard table
# (v1.1, 2018-11-21) stops at n=2^15 (ternary, 128-bit: logq 881); OpenFHE's
# stdlatticeparms.cpp extends it to 1747 at 2^16. This script measures 1763
# directly, with 1747 re-run in the same invocation as the anchor
# (fest_battery.jsonl:6 recorded 1747 -> 128.20 bits).
#
# Same method as run_fest_battery.py: sigma 3.19, primal_usvp + dual only
# (two-attack; same caveat as the F-est battery).
#
# Run from estimator/:
#   MAMBA_ROOT_PREFIX=$PWD/.micromamba ./bin/micromamba run \
#     -n sage-estimator sage run_cachemir_1763.py
import json
import math
import os
import sys
import time

sys.path.insert(0, "lattice-estimator-src")
from estimator import *  # noqa: F401,F403
import logging
logging.getLogger("estimator").setLevel(logging.WARNING)

SIGMA = 3.19
OUT = os.path.join("..", "results", "estimator-fest")
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, "cachemir_1763.jsonl")

SETS = [
    ("R16-CEILING-ANCHOR", 65536, 1747, "uniform",
     "OpenFHE HEStd_128_classic ternary ceiling at 2^16; anchor vs fest_battery.jsonl:6 (128.20)"),
    ("R16-1762", 65536, 1762, "uniform",
     "2 x the published 2^15 ternary bound (881) — naive doubling"),
    ("CACHEMIR-1763-UNIFORM", 65536, 1763, "uniform",
     "Cachemir's stated modulus at 2^16, uniform ternary secret"),
    ("CACHEMIR-1763-SPARSE-H192", 65536, 1763, "sparse192",
     "Cachemir's stated modulus with its stated sparse secret (Hamming weight 192), informational"),
]


def secret_dist(kind, n):
    if kind == "uniform":
        return ND.UniformMod(3), "uniform_ternary"
    if kind == "sparse192":
        return ND.SparseTernary(96, 96, n), "sparse_ternary_h192"
    raise ValueError(kind)


def main():
    jl = open(JSONL, "a")
    for name, n, logqp, skind, why in SETS:
        Xs, slabel = secret_dist(skind, n)
        params = LWE.Parameters(n=n, q=2 ** logqp, Xs=Xs,
                                Xe=ND.DiscreteGaussian(SIGMA), tag=name)
        rec = {"set": name, "N": n, "logQP": logqp, "secret": slabel,
               "why": why, "sigma": SIGMA}
        t0 = time.time()
        try:
            res = {"primal_usvp": LWE.primal_usvp(params),
                   "dual": LWE.dual(params)}
            for attack, cost in res.items():
                rec[attack] = {k: str(v) for k, v in dict(cost).items()}
            weakest = min(cost["rop"] for cost in res.values())
            bits = float(math.log(float(weakest), 2))
            rec["weakest_rop"] = str(weakest)
            rec["security_bits"] = float(round(float(bits), 2))
            rec["clears_128"] = bool(float(bits) >= 128.0)
            print(f"{name}: N={n} logQP={logqp} {slabel} -> 2^{bits:.2f} "
                  f"{'CLEARS' if bits >= 128 else 'FAILS'}", flush=True)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"{name}: ERROR {type(e).__name__}: {e}", flush=True)
        rec["seconds"] = float(round(float(time.time() - t0), 1))
        jl.write(json.dumps(rec, default=str) + "\n")
        jl.flush()
    jl.close()


# no __main__ guard on purpose — `sage <file>` runs with __name__ == "sage.all"
main()
