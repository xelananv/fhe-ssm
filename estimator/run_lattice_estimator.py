# run_lattice_estimator.py — the REAL Albrecht et al. lattice-estimator
# (github.com/malb/lattice-estimator, commit 3e48ef4), run against this
# project's actual candidate parameter sets from estimate.py's
# CANDIDATE_SETS. This script models the primal-uSVP and dual attacks
# (the full LWE.estimate() battery incl. hybrid/BDD attacks is
# run_full_battery.py — much slower). sigma=3.19, uniform-ternary secret,
# matching estimate.py exactly. q approximated as 2^logQP (standard
# practice — hardness depends on log q, not the exact RNS factorization).
# Run from estimator/:
#   MAMBA_ROOT_PREFIX=$PWD/.micromamba ./bin/micromamba run \
#     -n sage-estimator sage run_lattice_estimator.py
# Output committed at lattice_estimator_output.txt (2026-07-16 run).
import sys
sys.path.insert(0, "lattice-estimator-src")
from estimator import *  # noqa: F401,F403
import logging
logging.getLogger("estimator").setLevel(logging.DEBUG)

SIGMA = 3.19

CANDIDATE_SETS = [
    ("LAB", 16384, 360),
    ("PS-G1", 16384, 415),
    ("PS-G1c", 16384, 335),
    ("PS-P", 32768, 840),
    ("PS-G2", 65536, 1570),
]

print(f"{'set':>7} {'N':>6} {'logQP':>6}  full lattice-estimator result")
for name, n, logqp in CANDIDATE_SETS:
    q = 2 ** logqp
    params = LWE.Parameters(
        n=n, q=q,
        Xs=ND.UniformMod(3),           # uniform ternary secret
        Xe=ND.DiscreteGaussian(SIGMA), # sigma=3.19, matches estimate.py
        tag=name,
    )
    print(f"\n=== {name} (N={n}, logQP={logqp}) ===")
    try:
        res = {
            "primal_usvp": LWE.primal_usvp(params),
            "dual": LWE.dual(params)
        }
        for attack, cost in res.items():
            print(f"  {attack:20s} {cost}")
        # the security claim is the MINIMUM rop cost over every attack tried
        rops = [cost["rop"] for cost in res.values()]
        print(f"  --> weakest attack: {min(rops)}")
    except Exception as e:
        print(f"  ERROR: {e}")
