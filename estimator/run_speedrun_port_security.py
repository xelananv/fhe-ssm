# run_speedrun_port_security.py — the key set of the fhe-ssm port on the speedrun runtime (perseus-core, chain n32), as
# the 2026-11-02 publication run's bundle made it (code 50666f6f; pc-core / pc-client unchanged to 85971fb), priced with
# lattice-estimator 3e48ef4 under its default cost model. One line per result in battery.jsonl as it lands.
#
# Instances (moduli exact, from ../results/estimator-speedrun-port-20261103/chain_moduli.jsonl; secrets as sampled,
# sampler_weights.jsonl; sigma: pc-core keys.rs DiscreteGaussian::SIGMA 3.19, pcmm_keys.rs SIGMA 3.2):
#   MAIN-QP       main secret, uniform ternary, n 2^16, every rotation / conj / relin key at Q*P (log2 1705.34, dnum 6)
#   ENC-H32-QP    encapsulation secret, Hamming weight 32, the dense->sparse key, which pc-client only accepts in the
#                 bottom shape q0 q1 x P' (spec.rs:152-154; log2 307.64)
#   ENC-H32-PROJ  the same key reduced mod P' (the gadget term vanishes there): a clean RLWE sample, log2 251.65
#   PCMM-S1-QP    the PCMM ring-switch secret s', uniform ternary, n 2^12 (fwd: s -> s'(X^16) decomposes into 16 RLWE
#                 samples in ring 2^12 under s'; pack: s'_l -> s'), keys over q0 q1 P (log2 86.98)
#   PCMM-S1-PROJ  digit j reduced mod P q_{1-j} (pcmm_keys.rs ks_key: g = P q_{1-j}), a clean sample, log2 59.00
#   MAIN-N32A4    reference only: the main key on upstream's later chain n32a4 (log2 1733.26), for a future rebase
# The conventional modulus (the key's whole QP) is the defender's worst case: a larger q makes LWE easier.
#
# Phases (argv, default all, in this order; fast first so the cheap answers land first):
#   pcmm      PCMM-S1-QP, PCMM-S1-PROJ: LWE.estimate (arora-gb denied) + the attacks one by one
#   enc       ENC-H32-QP, ENC-H32-PROJ: the attacks one by one, as run_perseus_encaps3.py
#   encgrid   ENC-H32-QP: dual_hybrid on a fixed zeta grid with h1 optimised (the 2026-10-14 reviewer's method:
#             the estimator's own zeta search can stop at zeta 0 on n = 65,536)
#   main      MAIN-QP: usvp, dual, bdd, dual_hybrid, bdd_hybrid
#   maingrid  MAIN-QP: bdd_mitm_hybrid on a fixed zeta grid (the open search does not return: run_perseus_encaps6.py)
#   n32a4     MAIN-N32A4: usvp, dual, bdd
#   mainopen  MAIN-QP: bdd_mitm_hybrid with the estimator's own zeta search, capped (the long shot)
# Every call runs under signal.alarm; a cap is logged as a cap, never as a number.
# Run from estimator/:  MAMBA_ROOT_PREFIX=$PWD/.micromamba ./bin/micromamba run -n sage-estimator sage run_speedrun_port_security.py [phases]
import json
import math
import os
import signal
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "estimator", "lattice-estimator-src"))
from estimator import *  # noqa: F401,F403,E402
from estimator.lwe_dual import DH  # noqa: E402
from estimator.lwe_guess import exhaustive_search, guess_composition  # noqa: E402
import logging  # noqa: E402
logging.getLogger("estimator").setLevel(logging.WARNING)

OUT = os.path.join(ROOT, "results", "estimator-speedrun-port-20261103")
CHAIN = {}
for line in open(os.path.join(OUT, "chain_moduli.jsonl")):
    d = json.loads(line)
    CHAIN[d["preset"]] = d
N32, N32A4 = CHAIN["n32"], CHAIN["n32a4"]


def prod(xs):
    r = 1
    for x in xs:
        r *= int(x)
    return r


Q_MAIN = prod(N32["q"]) * prod(N32["p"])
Q_ENC = prod(N32["q"][: N32["level_unit"]]) * prod(N32["p"][: N32["p_sparse"]])
Q_ENC_PROJ = prod(N32["p"][: N32["p_sparse"]])
PC0, PC1, PCP = (int(x) for x in N32["pcmm_primes"])
Q_PCMM = PC0 * PC1 * PCP
Q_PCMM_PROJ = max(PCP * PC1, PCP * PC0)
Q_N32A4 = prod(N32A4["q"]) * prod(N32A4["p"])

SIG_MAIN, SIG_PCMM = 3.19, 3.2
N, N1 = 65536, 4096
INST = {
    "MAIN-QP": dict(n=N, q=Q_MAIN, xs="uniform", sigma=SIG_MAIN),
    "ENC-H32-QP": dict(n=N, q=Q_ENC, xs="h32", sigma=SIG_MAIN),
    "ENC-H32-PROJ": dict(n=N, q=Q_ENC_PROJ, xs="h32", sigma=SIG_MAIN),
    "PCMM-S1-QP": dict(n=N1, q=Q_PCMM, xs="uniform", sigma=SIG_PCMM),
    "PCMM-S1-PROJ": dict(n=N1, q=Q_PCMM_PROJ, xs="uniform", sigma=SIG_PCMM),
    "MAIN-N32A4": dict(n=N, q=Q_N32A4, xs="uniform", sigma=SIG_MAIN),
}


def params(name):
    i = INST[name]
    xs = ND.UniformMod(3) if i["xs"] == "uniform" else ND.SparseTernary(16, 16, i["n"])
    return LWE.Parameters(n=i["n"], q=i["q"], Xs=xs, Xe=ND.DiscreteGaussian(i["sigma"]))


class Cap(Exception):
    pass


def on_alarm(*_):
    raise Cap()


signal.signal(signal.SIGALRM, on_alarm)
log = open(os.path.join(OUT, "battery.txt"), "a")
jl = open(os.path.join(OUT, "battery.jsonl"), "a")


def emit(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def bits(cost):
    rop = float(cost["rop"])
    return round(float(math.log(rop, 2)), 2) if 0 < rop < float("inf") else float("inf")


def run(phase, inst, attack, fn, cap_s, extra=None):
    t0 = time.time()
    i = INST[inst]
    rec = {"phase": phase, "set": inst, "n": i["n"], "log2q": round(float(math.log(i["q"], 2)), 4), "xs": i["xs"],
           "sigma": i["sigma"], "attack": attack, "cap_s": cap_s, **(extra or {})}
    signal.alarm(cap_s)
    try:
        cost = fn()
        if isinstance(cost, dict) and "rop" not in cost:  # LWE.estimate: one Cost per attack
            rec["by_attack"] = {str(k): bits(v) for k, v in cost.items()}
            rec["bits"] = min(rec["by_attack"].values())
            emit(f"  {inst:13s} {attack:22s} min 2^{rec['bits']}  {time.time() - t0:.0f} s  {rec['by_attack']}")
        else:
            rec["bits"] = bits(cost)
            rec["detail"] = {str(k): str(v) for k, v in dict(cost).items()}
            emit(f"  {inst:13s} {attack:22s} 2^{rec['bits']}  {time.time() - t0:.0f} s  {cost!r}"[:300])
    except Cap:
        rec["error"] = f"cap {cap_s} s"
        emit(f"  {inst:13s} {attack:22s} HIT THE {cap_s} s CAP")
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        emit(f"  {inst:13s} {attack:22s} ERROR {rec['error'][:200]}")
    signal.alarm(0)
    rec["seconds"] = round(float(time.time() - t0), 1)
    jl.write(json.dumps(rec, default=str) + "\n")
    jl.flush()


ONE_BY_ONE = [
    ("usvp", lambda p: LWE.primal_usvp(p)),
    ("dual", lambda p: LWE.dual(p)),
    ("bdd", lambda p: LWE.primal_bdd(p)),
    ("dual_hybrid", lambda p: LWE.dual_hybrid(p)),
    ("bdd_hybrid", lambda p: LWE.primal_hybrid(p, mitm=False, babai=False)),
    ("bdd_mitm_hybrid", lambda p: LWE.primal_hybrid(p, mitm=True, babai=True)),
]
SPARSE_ONLY = [("usvp+guessing", lambda p: guess_composition(LWE.primal_usvp)(p))]


def phase_pcmm():
    for inst in ("PCMM-S1-QP", "PCMM-S1-PROJ"):
        p = params(inst)
        run("pcmm", inst, "LWE.estimate(-arora-gb)", lambda: LWE.estimate(p, deny_list=("arora-gb",)), 7200)
        for att, fn in ONE_BY_ONE:
            run("pcmm", inst, att, lambda: fn(p), 3600)


def phase_enc():
    for inst in ("ENC-H32-QP", "ENC-H32-PROJ"):
        p = params(inst)
        for att, fn in ONE_BY_ONE + SPARSE_ONLY:
            run("enc", inst, att, lambda: fn(p), 3600)


def phase_encgrid():
    from estimator.cost import Cost
    from estimator.util import local_minimum
    Cost.register_impermanent(rop=True, mem=False, red=True, beta=False, delta=False, m=True, d=False,
                              zeta=False, t=False, h1=False)
    p = params("ENC-H32-QP").normalize()
    h = p.Xs.hamming_weight

    def best_h1(zeta):
        h1_min, h1_max = max(0, h - (p.n - zeta)), min(zeta, h)
        if h1_min == h1_max:
            h1_max = h1_min + 1
        with local_minimum(h1_min, h1_max) as it:
            for h1 in it:
                it.update(DH.optimize_blocksize(solver=exhaustive_search, params=p, zeta=zeta, h1=h1))
            return it.y

    for zeta in [0, 8192, 16384, 24576, 32768, 40960, 49152, 53248, 57344, 61440, 63488, 64512, 65024]:
        run("encgrid", "ENC-H32-QP", f"dual_hybrid zeta={zeta}", lambda: best_h1(zeta), 1800, {"zeta": zeta})


def phase_main():
    p = params("MAIN-QP")
    for att, fn in ONE_BY_ONE[:5]:
        run("main", "MAIN-QP", att, lambda: fn(p), 3600)


def phase_maingrid():
    p = params("MAIN-QP")
    for zeta in (0, 64, 96, 112, 128, 144, 160, 192):
        run("maingrid", "MAIN-QP", f"bdd_mitm_hybrid zeta={zeta}",
            lambda: LWE.primal_hybrid(p, zeta=zeta, mitm=True, babai=True), 600, {"zeta": zeta})


def phase_n32a4():
    p = params("MAIN-N32A4")
    for att, fn in ONE_BY_ONE[:3]:
        run("n32a4", "MAIN-N32A4", att, lambda: fn(p), 3600)


def phase_mainopen():
    p = params("MAIN-QP")
    run("mainopen", "MAIN-QP", "bdd_mitm_hybrid (open zeta search)",
        lambda: LWE.primal_hybrid(p, mitm=True, babai=True), 5 * 3600)


def phase_smoke():
    for inst in ("PCMM-S1-QP", "ENC-H32-QP", "MAIN-QP"):
        run("smoke", inst, "usvp", lambda: LWE.primal_usvp(params(inst)), 600)


PHASES = {"pcmm": phase_pcmm, "enc": phase_enc, "encgrid": phase_encgrid, "main": phase_main,
          "maingrid": phase_maingrid, "n32a4": phase_n32a4, "mainopen": phase_mainopen}
want = sys.argv[1:] or list(PHASES)
PHASES["smoke"] = phase_smoke  # only when asked for
emit(f"\n=== speedrun-port key set, phases {want}, started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}; "
     f"log2 q: " + ", ".join(f"{k} {math.log(v['q'], 2):.4f}" for k, v in INST.items()) + " ===")
for ph in want:
    emit(f"\n--- phase {ph}, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    PHASES[ph]()
emit(f"\n=== complete {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
