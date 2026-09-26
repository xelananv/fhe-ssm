# run_fest_hybrid_r17.py — the FULL LWE.estimate() battery on the N=2^17 sets
# that run_fest_hybrid.py DROPPED after they OOM-killed the 16 GB Mac (2026-08-18,
# results/OFFLINE_BATTERY_20260818.md: fest_hybrid.jsonl has 0 rows). These are the
# DEPLOYED ring's sets, including both SPARSE (h'=32) ones -- hybrid / MITM is
# exactly the sparse-secret threat, so the 128-bit claim for the encapsulated
# bootstrap (lever 2) rests on these rows. RAM-bound: schedule on a >=96 GB box
# (S2.9 explore.json x_est_hybrid_r17, gpus:0).
#
# logQP values are TRANSCRIBED from estimator/run_fest_battery.py (R5): 2371 = the
# deployed chain at N=2^17; 3523 = the HEStd_128_classic uniform-ternary ceiling.
# Run from estimator/:
#   MAMBA_ROOT_PREFIX=$PWD/.micromamba ./bin/micromamba run -n sage-estimator sage run_fest_hybrid_r17.py
# NO __main__ GUARD (`sage <file>` sets __name__ == "sage.all"); float() every numeric before json.
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
JSONL = os.path.join(OUT, "fest_hybrid_r17.jsonl")
LOG = os.path.join(OUT, "fest_hybrid_r17.txt")

SETS = [
    ("R17-DEPLOYED", 131072, 2371, "uniform", "the deployed chain: 2-attack 195.39 bits (fest_battery.jsonl)"),
    ("R17-SPARSE-H32", 131072, 2371, "sparse32", "lever 2 (encapsulated, h'=32) at the deployed chain: 2-attack 195.08 -- the hybrid attacks are THE threat"),
    ("R17-CEILING", 131072, 3523, "uniform", "the HEStd 2^17 ceiling constant: 2-attack 128.09"),
    ("R17-SPARSE-H32-CEIL", 131072, 3523, "sparse32", "sparse at the ceiling: 2-attack FAILS (127.82); hybrid can only make it worse -- recorded for completeness"),
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


emit(f"=== F-est FULL LWE.estimate() battery, N=2^17 sets, sigma={SIGMA} ===")
for name, n, logqp, skind, why in SETS:
    Xs, slabel = secret_dist(skind, n)
    params = LWE.Parameters(n=n, q=2 ** logqp, Xs=Xs, Xe=ND.DiscreteGaussian(SIGMA), tag=name)
    emit(f"\n=== {name} (N={n}, logQP={logqp}, secret={slabel}) -- {why} ===")
    t0 = time.time()
    row = {"set": name, "N": n, "logQP": logqp, "secret": slabel, "why": why, "sigma": SIGMA, "battery": "LWE.estimate() full"}
    try:
        res = LWE.estimate(params)
        costs = {}
        for attack, cost in res.items():
            try:
                rop = float(cost["rop"])
                costs[attack] = {"rop_log2": float(__import__("math").log2(rop)) if rop > 0 else None, "raw": str(cost)}
            except Exception as e:  # noqa: BLE001
                costs[attack] = {"error": str(e), "raw": str(cost)}
        finite = [c["rop_log2"] for c in costs.values() if c.get("rop_log2") is not None]
        row["attacks"] = costs
        row["min_bits_all_attacks"] = min(finite) if finite else None
        row["weakest_attack"] = min((k for k in costs if costs[k].get("rop_log2") is not None), key=lambda k: costs[k]["rop_log2"]) if finite else None
        row["legal128"] = bool(row["min_bits_all_attacks"] is not None and row["min_bits_all_attacks"] >= 128.0)
        emit(f"  weakest: {row['weakest_attack']} at {row['min_bits_all_attacks']} bits -> legal128={row['legal128']}")
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)
        emit(f"  ERROR: {e}")
    row["seconds"] = float(time.time() - t0)
    jl.write(json.dumps(row) + "\n")
    jl.flush()
emit("=== done ===")
