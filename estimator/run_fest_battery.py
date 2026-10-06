# run_fest_battery.py — the F-est cases, none of which this repo has ever run.
#
# Verified UNRUN 2026-08-18 by a records-only R1 sweep (MEASUREMENTS_DETAIL.md,
# then MEASUREMENTS_INVALID.md, then the tree): the largest ring ever estimated
# here is N=2^16 (PS-G2), every estimator script hardcodes a UNIFORM ternary
# secret, and `1711` appears only in derivations and to-do rows. Corroborated
# by RUN_LEDGER.md:1300 (regenerated 2026-08-18).
#
# Run from estimator/ (the MAMBA_ROOT_PREFIX quirk is load-bearing):
#   MAMBA_ROOT_PREFIX=$PWD/.micromamba ./bin/micromamba run \
#     -n sage-estimator sage run_fest_battery.py
#
# Writes incrementally to ../results/estimator-fest/ so a killed run keeps
# whatever finished. N=2^17 sets are SLOW (dimension is 2x the largest set
# ever run here) — that is expected, not a hang.
#
# EVERY logQP BELOW IS TRANSCRIBED, NOT DERIVED HERE (R5). Sources:
#   2371  = the DEPLOYED chain at N=2^17. logQ = 60+29*59 = 1771; digit
#           ceil(30/3)=10 towers = 591 bits; logQP ~ 2371.
#           MEASUREMENTS_DETAIL.md:1691; WALL_SYNTHESIS.md:99.
#   3523  = HEStd_128_classic uniform-ternary logQP ceiling at N=2^17
#           (881 at 2^15 / 1747 at 2^16 / 3523 at 2^17).
#           MEASUREMENTS_DETAIL.md:1707, verified by 3 auditors against
#           vendor/.../stdlatticeparms.cpp:184,187,190. NOTE the proposals'
#           "~3576" was REFUTED, and amortize.audit.md uses 3540 throughout —
#           BOTH differ from the ledger's 3523. This battery estimates the
#           ceiling directly so the constant stops being a table lookup.
#   1711  = lever 3's encapsulated short chain at N=2^16 (D=26, delta=2^50,
#           dnum=4), claimed to pass because 1711 <= 1747 with 9 bits of
#           P-margin. MEASUREMENTS_DETAIL.md:1727; boot-cost.audit.md:47.
#   3187  = lever 4's D=45 chain at N=2^17: logQ = 60+45*59 = 2715, plus 8
#           specials -> ~3187. amortize.audit.md:13.
#   3541  = D=53 at dnum=10: logQ = 60+53*59 = 3187, plus 6 specials (~354)
#           -> ~3541, which BREACHES the ceiling. amortize.audit.md:37.
#   3482  = D=53 at dnum>=12: 5 specials -> ~3482, the forced repair.
#           amortize.audit.md:37,75.
#   h'=32 = the sparse-secret Hamming weight of lever 2's encapsulated
#           bootstrap. ND.SparseTernary(p, m, n) puts p ones and m minus-ones,
#           so h' = p+m => SparseTernary(16, 16, n).
#           MEASUREMENTS_DETAIL.md:1583,1725.
#
# Coincidence worth not tripping over: D=45's logQP and D=53's logQ are BOTH
# 3187. They are different quantities that happen to collide.
import json
import os
import sys
import time

sys.path.insert(0, "lattice-estimator-src")
from estimator import *  # noqa: F401,F403
import logging
logging.getLogger("estimator").setLevel(logging.WARNING)

SIGMA = 3.19          # matches estimate.py and run_lattice_estimator.py
OUT = os.path.join("..", "results", "estimator-fest")
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, "fest_battery.jsonl")
LOG = os.path.join(OUT, "fest_battery.txt")

# (name, N, logQP, secret, why)
SETS = [
    # --- the broader gap: ring 2^17 has never been estimated AT ALL, and it
    # is the ring every 128-bit-secure measurement in the paper uses.
    ("R17-DEPLOYED", 131072, 2371, "uniform",
     "the deployed chain; paper's own operating point, never estimated"),
    ("R17-CEILING", 131072, 3523, "uniform",
     "HEStd_128_classic ceiling at 2^17 - estimate the table constant itself"),
    # --- F-est case (a): lever 2, sparse-secret encapsulated at 2^17
    ("R17-SPARSE-H32", 131072, 2371, "sparse32",
     "F-est (a): lever 2 encapsulated bootstrap, h'=32, at the deployed chain"),
    ("R17-SPARSE-H32-CEIL", 131072, 3523, "sparse32",
     "F-est (a): same secret at the ceiling - sizes the sparse margin"),
    # --- F-est case (b): lever 3's short chain
    ("R16-1711", 65536, 1711, "uniform",
     "F-est (b): lever 3 short chain, claimed 1711 <= 1747 with 9 bits spare"),
    ("R16-CEILING", 65536, 1747, "uniform",
     "control for (b): the 2^16 ceiling constant"),
    # --- F-est case (c): lever 4's depth choices
    ("D45", 131072, 3187, "uniform",
     "F-est (c): lever 4 D=45 chain (logQ 2715 + 8 specials)"),
    ("D53-DNUM10", 131072, 3541, "uniform",
     "F-est (c): D=53 at dnum=10 - expected to BREACH (3541 > 3523)"),
    ("D53-DNUM12", 131072, 3482, "uniform",
     "F-est (c): D=53 at dnum>=12, the forced repair"),
]


def secret_dist(kind, n):
    if kind == "uniform":
        return ND.UniformMod(3), "uniform_ternary"
    if kind == "sparse32":
        # h' = p + m = 32
        return ND.SparseTernary(16, 16, n), "sparse_ternary_h32"
    raise ValueError(kind)


def main():
    log = open(LOG, "a")
    jl = open(JSONL, "a")

    def emit(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    emit(f"=== F-est battery, sigma={SIGMA}, "
         f"lattice-estimator in lattice-estimator-src ===")
    emit(f"{'set':>20} {'N':>7} {'logQP':>6} {'secret':>20}")

    for name, n, logqp, skind, why in SETS:
        Xs, slabel = secret_dist(skind, n)
        params = LWE.Parameters(n=n, q=2 ** logqp, Xs=Xs,
                                Xe=ND.DiscreteGaussian(SIGMA), tag=name)
        emit(f"\n=== {name} (N={n}, logQP={logqp}, secret={slabel}) ===")
        emit(f"    why: {why}")
        rec = {"set": name, "N": n, "logQP": logqp, "secret": slabel,
               "why": why, "sigma": SIGMA}
        t0 = time.time()
        try:
            res = {"primal_usvp": LWE.primal_usvp(params),
                   "dual": LWE.dual(params)}
            for attack, cost in res.items():
                emit(f"  {attack:20s} {cost}")
                rec[attack] = {k: str(v) for k, v in dict(cost).items()}
            rops = [cost["rop"] for cost in res.values()]
            weakest = min(rops)
            # log2 of the weakest attack cost = the security claim
            import math
            bits = float(math.log(float(weakest), 2))
            # float() EVERYWHERE, deliberately: `sage <file>` injects Sage's
            # own round() into globals, which returns RealDoubleElement_gsl —
            # not JSON-serializable. Measured 2026-08-18: the battery ran the
            # estimator correctly and then died on the first jl.write().
            rec["weakest_rop"] = str(weakest)
            rec["security_bits"] = float(round(float(bits), 2))
            rec["clears_128"] = bool(float(bits) >= 128.0)
            emit(f"  --> weakest attack: {weakest}  = 2^{bits:.2f}  "
                 f"{'CLEARS' if bits >= 128 else 'FAILS'} 128-bit")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            emit(f"  ERROR: {type(e).__name__}: {e}")
        rec["seconds"] = float(round(float(time.time() - t0), 1))
        emit(f"  ({rec['seconds']} s)")
        # default=str is the backstop for any other Sage numeric that leaks in
        jl.write(json.dumps(rec, default=str) + "\n")
        jl.flush()

    emit("\n=== battery complete ===")
    log.close()
    jl.close()


# NO `if __name__ == "__main__"` GUARD — DELIBERATE, DO NOT "FIX".
# `sage <file>` executes the script with __name__ == "sage.all", so a __main__
# guard NEVER FIRES: the run exits 0, prints nothing, writes no record, and
# looks exactly like a completed run that found nothing. Measured 2026-08-18.
# This is a sibling of the MAMBA_ROOT_PREFIX quirk in the header. Top-level
# invocation is also what run_lattice_estimator.py does.
main()
