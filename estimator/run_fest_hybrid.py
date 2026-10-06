# run_fest_hybrid.py — the FULL LWE.estimate() battery on the F-est sets that
# the two-attack run cannot honestly settle.
#
# WHY THIS EXISTS. run_fest_battery.py models primal_usvp + dual only (that is
# what run_lattice_estimator.py has always done, and what the five committed
# 2026-07-16 sets used). For a UNIFORM ternary secret that is a defensible
# approximation. For a SPARSE secret it is NOT: low Hamming weight is exactly
# the structure that hybrid / meet-in-the-middle attacks exploit, so quoting
# "h'=32 clears 128 bits" from primal_usvp+dual alone would be an artifact of
# not having run the attack that threatens it. LWE.estimate() adds bdd,
# dual_hybrid, bdd_hybrid, bdd_mitm_hybrid and arora_gb.
#
# Run from estimator/:
#   MAMBA_ROOT_PREFIX=$PWD/.micromamba ./bin/micromamba run \
#     -n sage-estimator sage run_fest_hybrid.py
#
# NO __main__ GUARD — `sage <file>` sets __name__ == "sage.all" (see
# run_fest_battery.py's note). float() every numeric before json (Sage's
# round() returns RealDoubleElement_gsl).
import json
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
JSONL = os.path.join(OUT, "fest_hybrid.jsonl")
LOG = os.path.join(OUT, "fest_hybrid.txt")

# The sets where the two-attack answer is not sufficient:
#  - both sparse sets (hybrid attacks are THE threat to a sparse secret)
#  - the three marginal uniform sets (127-131 bits: too close to 128 to rest
#    on a partial attack list)
# 2026-08-18: the four N=2^17 sets OOM-KILLED the process (exit 137) on the
# FIRST one. LWE.estimate()'s hybrid attacks at n=131072 exceed this Mac's
# memory. Reduced to the N=2^16 sets, which are half the dimension and do fit.
# THE N=2^17 SETS REMAIN UNRUN under the full battery — in particular both
# SPARSE sets, which is exactly where hybrid attacks matter. Recorded as a gap,
# not quietly dropped.
SETS = [
    ("R16-1711", 65536, 1711, "uniform",
     "F-est (b): 2-attack said 130.97 - only 3 bits over, check the full list"),
    ("R16-SPARSE-H32", 65536, 1711, "sparse32",
     "sparse-vs-uniform hybrid gap at the ring that fits in memory"),
    ("R16-CEILING", 65536, 1747, "uniform",
     "the 2^16 ceiling constant: 2-attack said 128.20"),
]


def secret_dist(kind, n):
    if kind == "uniform":
        return ND.UniformMod(3), "uniform_ternary"
    if kind == "sparse32":
        return ND.SparseTernary(16, 16, n), "sparse_ternary_h32"
    raise ValueError(kind)


log = open(LOG, "a")
jl = open(JSONL, "a")


def emit(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


emit(f"=== F-est FULL LWE.estimate() battery, sigma={SIGMA} ===")

for name, n, logqp, skind, why in SETS:
    Xs, slabel = secret_dist(skind, n)
    params = LWE.Parameters(n=n, q=2 ** logqp, Xs=Xs,
                            Xe=ND.DiscreteGaussian(SIGMA), tag=name)
    emit(f"\n=== {name} (N={n}, logQP={logqp}, secret={slabel}) ===")
    emit(f"    why: {why}")
    rec = {"set": name, "N": n, "logQP": logqp, "secret": slabel,
           "why": why, "sigma": SIGMA, "battery": "LWE.estimate"}
    t0 = time.time()
    try:
        res = LWE.estimate(params)
        import math
        per = {}
        for attack, cost in res.items():
            rop = float(cost["rop"])
            bits = float(math.log(rop, 2)) if rop > 0 else float("inf")
            per[str(attack)] = round(float(bits), 2)
            emit(f"  {str(attack):20s} 2^{bits:7.2f}")
        rec["per_attack_bits"] = per
        weakest_name = min(per, key=per.get)
        rec["weakest_attack"] = weakest_name
        rec["security_bits"] = float(per[weakest_name])
        rec["clears_128"] = bool(per[weakest_name] >= 128.0)
        emit(f"  --> WEAKEST: {weakest_name} at 2^{per[weakest_name]:.2f}  "
             f"{'CLEARS' if per[weakest_name] >= 128 else '*** FAILS ***'} 128-bit")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        emit(f"  ERROR: {type(e).__name__}: {e}")
    rec["seconds"] = float(round(float(time.time() - t0), 1))
    emit(f"  ({rec['seconds']} s)")
    jl.write(json.dumps(rec, default=str) + "\n")
    jl.flush()

emit("\n=== hybrid battery complete ===")
log.close()
jl.close()
