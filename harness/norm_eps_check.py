#!/usr/bin/env python3
"""norm_eps_check.py -- the RMSNorm eps fix (S3.7 V1, 2026-09-05), plaintext twin.

Reproduces, in float64 and with no FHE, the exact Newton chain the harness
rmsnorm() runs (gpu_real_model.cu, harness/cpu_real_model.cpp) in BOTH frames:

  noeps : ms = mean(x^2)            -- the circuit before V1 (T18 section 3)
  eps   : ms = mean(x^2) + eps      -- the model's own definition
          (train_fhe_native_ssm.py :153-:155, :834, :927-:932) and V1's circuit

and compares each against the reference 1/sqrt(mean + eps) the harness selftests
use, so every selftest number has a predicted value to be checked against:

  gpu   : the pod selftest (gpu_real_model.cu, "selftest":"rmsnorm"): v[i] =
          sin(0.1 i)/2, d = 1024, generic seed (1, 0), 4 Newton steps. Record:
          results/dense-demo-s31/pod_rtx6000x5_20260904/pod_pull/
          mgpu_selftest_r17_nccl.log:24 -> err 0.0310171 (no eps). With eps the
          same chain gives 0.0310645: still above the 1e-2 bar, because the
          generic seed is under-converged after 4 steps -- not because of eps.
  cpu   : harness/cpu_real_model.cpp "rmsnorm_newton": v[i] = sin(0.37 i)/2,
          d = 48, seed (1.5/sqrt(ms), -0.5/ms^1.5), 3 steps.
  seeded: the V1 GPU selftest addition "rmsnormSeeded": the bundle's L0.tm seed
          (a, b, iters) on v scaled so mean(v^2) = -a/(b Kt), Kt = 16 (the
          seed's zero crossing -a/b sits at 1.34-1.45x the design window's top,
          T18 section 4, so -a/(16 b) is inside the window near its low end:
          L0.tm 2.4e-4 in [1.1e-4, 2.5e-3]); there eps/ms = 4 %, so the no-eps
          chain misses the 1e-2 bar (0.028) and the eps chain is at the Newton
          floor -- a real pass/fail. --sigma K also runs it re-parametrised.

Optionally --sigma S runs the chain in the --ms-norm re-parametrisation
(msn = S*ms, seed a/sqrt(S), b/S^1.5, tail * sqrt(S)) to show it is exact.
Writes one JSON line per case to stdout (and --out FILE).
"""
import argparse, json, math, sys

def newton_chain(v, a0, b0, iters, eps_in_circuit, eps=1e-5, sigma=1.0):
    d = len(v)
    mean = sum(x * x for x in v) / d
    ms = mean + (eps if eps_in_circuit else 0.0)
    # --ms-norm re-parametrisation (gpu_real_model.cu rmsnorm(): sigma = -K b/a)
    msn = sigma * ms
    a_n = a0 / math.sqrt(sigma); b_n = b0 / (sigma * math.sqrt(sigma))
    y = a_n + b_n * msn
    for _ in range(iters):
        y = y * (1.5 - 0.5 * msn * y * y)
    y *= math.sqrt(sigma)                 # tail: rsqrt(ms) = sqrt(sigma) rsqrt(msn)
    ref = 1.0 / math.sqrt(mean + eps)
    err = max(abs(x * y - x * ref) for x in v)
    return {"mean": mean, "msCircuit": ms, "yFinal": y, "yRef": ref,
            "relErrY": y / ref - 1.0, "maxAbsErr": err, "finite": math.isfinite(y)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", help="pbd430a_seeds.json (read-only) for the seeded case")
    ap.add_argument("--sigma", type=float, default=0.0, help="also run under --ms-norm K = this")
    ap.add_argument("--out")
    args = ap.parse_args()
    eps = 1e-5
    out = []
    # --- gpu selftest chain
    vg = [math.sin(0.1 * i) * 0.5 for i in range(1024)]
    for frame in ("noeps", "eps"):
        r = newton_chain(vg, 1.0, 0.0, 4, frame == "eps", eps)
        out.append({"case": "gpu_selftest", "frame": frame, "d": 1024, "seed": [1.0, 0.0], "iters": 4,
                    "passBar": 1e-2, "passes": r["maxAbsErr"] < 1e-2, **r})
    # --- cpu twin chain
    vc = [math.sin(0.37 * i) * 0.5 for i in range(48)]
    msc = sum(x * x for x in vc) / 48
    a0c, b0c = 1.5 / math.sqrt(msc), -0.5 / (msc * math.sqrt(msc))
    for frame in ("noeps", "eps"):
        r = newton_chain(vc, a0c, b0c, 3, frame == "eps", eps)
        out.append({"case": "cpu_twin_rmsnorm_newton", "frame": frame, "d": 48, "seed": [a0c, b0c], "iters": 3,
                    "passBar": 5e-3, "passes": r["maxAbsErr"] < 5e-3, **r})
    # --- seeded GPU selftest (V1 addition), needs the bundle's seeds
    if args.seeds:
        sd = json.load(open(args.seeds))["L0.tm"]
        a, b, it = sd["a"], sd["b"], int(sd["iters"])
        Kt = 16.0                                  # the operating point gpu_real_model.cu test 4b uses
        msTest = -a / (b * Kt)
        lo, hi = sd.get("ms_range", [float("nan"), float("nan")])
        # the CPU twin's mirror (cpu_real_model.cpp --norm-seed A B ITERS): same chain on
        # its own d = 48 vector, so its FHE number has a prediction too
        scc = math.sqrt(msTest / msc)
        for frame in ("noeps", "eps"):
            r = newton_chain([x * scc for x in vc], a, b, it, frame == "eps", eps)
            out.append({"case": "cpu_twin_seeded_L0tm", "frame": frame, "d": 48, "seed": [a, b], "iters": it,
                        "Kt": Kt, "msTest": msTest, "scale": scc, "passBar": 1e-2, "passes": r["maxAbsErr"] < 1e-2, **r})
        msv = sum(x * x for x in vg) / 1024
        sc = math.sqrt(msTest / msv)
        vs = [x * sc for x in vg]
        for frame in ("noeps", "eps"):
            for sig in ([1.0] + ([args.sigma] if args.sigma > 0 else [])):
                sigma = 1.0 if sig == 1.0 else -sig * b / a    # --ms-norm K: sigma = -K b / a
                r = newton_chain(vs, a, b, it, frame == "eps", eps, sigma)
                out.append({"case": "gpu_selftest_seeded_L0tm", "frame": frame, "d": 1024,
                            "seed": [a, b], "iters": it, "Kt": Kt, "msTest": msTest,
                            "msTestInDesignRange": bool(lo <= msTest <= hi), "designRange": [lo, hi],
                            "scale": sc, "msNormK": sig if sig != 1.0 else 0.0, "sigma": sigma,
                            "passBar": 1e-2, "passes": r["maxAbsErr"] < 1e-2, **r})
    for o in out:
        print(json.dumps(o))
    if args.out:
        with open(args.out, "w") as f:
            for o in out: f.write(json.dumps(o) + "\n")

if __name__ == "__main__":
    main()
