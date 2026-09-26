#!/usr/bin/env python3
"""Numeric instantiation of the tick bounds in TICK_SCALING_BOUNDS_20260903.md.

Counts come from tick_level_trace.py (a replay of gpu_real_model.cu's own
statements); every unit cost below is an ASSUMPTION, labelled as such in the
report. Nothing here reads results/.

Usage: python3 tick_bounds_tables.py [--md]
"""
import argparse
import json
import math
import sys

import tick_level_trace as T

# ---------------------------------------------------------------------------
# Security coupling (OpenFHE rules, cited in the report §3):
#   logQ  = firstMod + D*scaleBits + 1                (ckksrns-parametergeneration.cpp:126-131)
#   digits: numPerPartQ = ceil((D+1)/dnum)            (rns-cryptoparameters.cpp:436)
#   maxBits = bits of the largest digit (+1)          (rns-cryptoparameters.cpp:455-470)
#   sizeP = ceil(maxBits/auxBits), logP = sizeP*auxBits, auxBits = 60 (:472-475, AUXMODSIZE)
#   bound: logQP <= beta(N) from the HEStd_ternary/128_classic table (stdlatticeparms.cpp:182-188)
# ---------------------------------------------------------------------------
FIRST, SCALE, AUX = 60, 59, 60
BETA = {15: 881, 16: 1747, 17: 3523}          # table rows; 2^18 has NO row (FindRingDim returns 2n)
BETA_ASSUMED = {18: 7000}                     # ASSUMED extrapolation (ratios 1.98, 2.02 per doubling)


def log_qp(D, dnum):
    nprimes = D + 1
    per = math.ceil(nprimes / dnum)
    sizes = [FIRST] + [SCALE] * (nprimes - 1)
    maxbits = 0
    for j in range(dnum):
        seg = sizes[j * per:(j + 1) * per]
        if seg:
            maxbits = max(maxbits, sum(seg))
    if maxbits != AUX:
        maxbits += 1
    sizeP = math.ceil(maxbits / AUX)
    logQ = FIRST + D * SCALE + 1
    return logQ + sizeP * AUX, sizeP


def d_max(logN, dnum=3):
    beta = BETA.get(logN, BETA_ASSUMED.get(logN))
    D = 0
    while log_qp(D + 1, dnum)[0] <= beta:
        D += 1
    return D


# ---------------------------------------------------------------------------
# ASSUMED unit costs. Reference point: N_ref = 2^16, l_ref = 14 live limbs, an
# A100-class GPU. The magnitudes are of the order reported for GPU CKKS
# libraries in the open literature (they are NOT measurements of this repo's
# binaries). Scaling model: ct x pt ∝ l*N; rotation/rescale ∝ l*N*log2(N)
# (NTT-bound, hybrid key switching at fixed dnum); bootstrap ∝ N*log2(N)*(D+1).
# ---------------------------------------------------------------------------
ASSUMED = {
    "c_pt_ref_ms": 0.18,     # ct x pt multiply, 2^16, 14 limbs
    "c_rot_ref_ms": 0.61,    # rotation (key switch), 2^16, 14 limbs
    "c_rs_ref_ms": 0.10,     # rescale, 2^16, 14 limbs
    "c_add_ref_ms": 0.02,    # ct + ct, 2^16, 14 limbs
    "c_boot_ref_ms": 126.0,  # bootstrap, 2^16, chain of 28 primes (D_ref = 27)
    "bw_h2d_GBs": 20.0,      # host->device copy bandwidth
    "bw_hbm_GBs": 2000.0,    # device memory bandwidth
    "c_enc_ref_ms": 5.0,     # host encode of one full plaintext, 2^15, ~20 limbs (no-store path only)
}
N_REF, L_REF, D_REF = 2 ** 16, 14, 27


def c_pt(N, l):
    return ASSUMED["c_pt_ref_ms"] * (l * N) / (L_REF * N_REF)


def ks_factor(l, dnum):
    """ASSUMED hybrid key-switch cost model: NTT-limb operations ∝ dnum*(l + ceil(l/dnum)),
    normalised to dnum = 3 (the harness default, scheme-utils.h:38)."""
    return (dnum * (l + math.ceil(l / dnum))) / (3 * (l + math.ceil(l / 3)))


def c_rot(N, l, dnum=3):
    return ASSUMED["c_rot_ref_ms"] * (l * N * math.log2(N)) / (L_REF * N_REF * math.log2(N_REF)) * ks_factor(l, dnum)


def c_rs(N, l):
    return ASSUMED["c_rs_ref_ms"] * (l * N * math.log2(N)) / (L_REF * N_REF * math.log2(N_REF))


def c_add(N, l):
    return ASSUMED["c_add_ref_ms"] * (l * N) / (L_REF * N_REF)


def c_boot(N, D, dnum=3):
    return (ASSUMED["c_boot_ref_ms"] * (N * math.log2(N) * (D + 1)) / (N_REF * math.log2(N_REF) * (D_REF + 1))
            * ks_factor(D + 1, dnum))


class A:
    pass


def trace(d, L, D, lam, F, margin, iters, lanes_block, lazy, seeds=None, ticks=12, shift=True):
    a = A()
    a.depth = D; a.extra_depth = 0; a.dboot = 0; a.lam = lam; a.boot_floor = F; a.margin = margin
    a.dpad = d; a.d = d; a.dff = 4 * d; a.layers = L; a.iters = iters; a.seeds = seeds
    a.lanes_block = lanes_block; a.shift_mix = shift; a.lazy = lazy; a.final_boot = False
    a.ticks = ticks; a.json = None; a.sites = False
    o = T.run(a)
    warm = o["per_tick"][4:]                       # steady state: skip the cold-carry ticks
    keys = ["boots", "rot", "ctpt_diag", "ctpt_elem", "ctct", "rescale", "add", "diag_limbs", "rot_limbs"]
    mean = {k: sum(t[k] for t in warm) / len(warm) for k in keys}
    mean["boots_min"] = min(t["boots"] for t in warm)
    mean["boots_max"] = max(t["boots"] for t in warm)
    mean["K"] = o["params"]["K"]; mean["BS"] = o["params"]["BS"]; mean["G"] = o["params"]["G"]
    return mean


def d_min_layer(I_tm, I_cm):
    """Minimal multiplicative depth of one layer of the circuit (report Lemma 11)."""
    return 2 * (I_tm + I_cm) + 16


def c_path_layer(I_tm, I_cm, lanes_block):
    """Level consumption along the residual path per layer as implemented (report Lemma 8)."""
    return 4 * (I_tm + I_cm) + 24 + (2 if lanes_block else 0)


def cell(d, L, logN, dboot=19, F=3, margin=4, iters=3, lanes_block=True, lazy=False, seeds=None,
         lanes_used=None, dnum=3, depth=None):
    N = 2 ** logN
    Dpad = d
    REP = N // (2 * Dpad)
    D = d_max(logN, dnum) if dnum != "max" else d_max_dnum_max(logN)
    if dnum == "max":
        dnum = D + 1
    if depth is not None:               # explicit chain (e.g. the demo's depth 41), must be legal
        assert log_qp(depth, dnum)[0] <= BETA.get(logN, BETA_ASSUMED.get(logN)), "depth exceeds beta(N)"
        D = depth
    lam = D - dboot
    feasible = lam >= 5 + F           # the Newton refresh (need 5) must be satisfiable right after a boot
    P = L * (2 * d * d + 3 * d * 4 * d)
    out = {"d": d, "L": L, "logN": logN, "N": N, "REP": REP, "D": D, "lam": lam, "P": P, "feasible": feasible,
           "dnum": dnum}
    if not feasible:
        return out
    tr = trace(d, L, D, lam, F, margin, iters, lanes_block, lazy, seeds)
    out.update(tr)
    h = 2 if lanes_block else 1
    # ---- lower bound (Theorem 1) ----
    c = REP if lanes_used is None else lanes_used
    M_min = math.ceil(2 * c * P / N)                      # slot counting
    Dmin = L * d_min_layer(iters, iters) + (2 * iters + 3)  # + output norm
    B_min = max(0, math.ceil((Dmin - D) / lam)) + L / lam   # path + amortised carry
    R_min = L * (5 + tr["K"]) * int(math.log2(Dpad))       # (3+K) matvec inputs + 2 norms, each >= log2 Dpad
    l_min = 2                                              # a multiply followed by a rescale needs >= 2 limbs
    T_lb = B_min * c_boot(N, D, dnum) + M_min * c_pt(N, l_min) + R_min * c_rot(N, l_min, dnum)
    out.update({"M_min": M_min, "B_min": B_min, "R_min": R_min, "T_lb_s": T_lb / 1000})
    # ---- upper bound (Theorem 2), implemented counts x unit costs ----
    ell_pt = tr["diag_limbs"] / tr["ctpt_diag"]           # limb-weighted mean operand limbs at the matvecs
    ell_rot = tr["rot_limbs"] / tr["rot"]
    ell_full = D + 1
    t_boot = tr["boots"] * c_boot(N, D, dnum)
    t_pt = tr["diag_limbs"] * c_pt(N, 1)                    # exact limb-weighted
    t_elem = (tr["ctpt_elem"] + tr["ctct"]) * c_pt(N, ell_full) * 1.0
    t_rot = tr["rot_limbs"] * c_rot(N, 1, dnum)
    t_rs = tr["rescale"] * c_rs(N, ell_full)
    t_add = tr["add"] * c_add(N, ell_pt)
    bytes_h2d = tr["diag_limbs"] * (N // REP) * 8         # store: N/REP words per limb per diagonal-half
    bytes_expand = tr["diag_limbs"] * N * 8               # expand kernel writes full limbs
    t_h2d = bytes_h2d / (ASSUMED["bw_h2d_GBs"] * 1e9) * 1000
    t_expand = bytes_expand / (ASSUMED["bw_hbm_GBs"] * 1e9) * 1000
    T_ub = t_boot + t_pt + t_elem + t_rot + t_rs + t_add + t_h2d + t_expand
    bytes_nostore = tr["diag_limbs"] * N * 8              # full plaintexts, REP x larger
    out.update({"ell_pt": ell_pt, "ell_rot": ell_rot, "t_boot_s": t_boot / 1000, "t_pt_s": t_pt / 1000,
                "t_rot_s": t_rot / 1000, "t_elem_s": t_elem / 1000, "t_rs_s": t_rs / 1000,
                "t_add_s": t_add / 1000, "bytes_h2d_GB": bytes_h2d / 1e9, "t_h2d_s": t_h2d / 1000,
                "bytes_expand_GB": bytes_expand / 1e9, "t_expand_s": t_expand / 1000,
                "bytes_nostore_GB": bytes_nostore / 1e9, "T_ub_s": T_ub / 1000,
                "T_ub_per_conv_s": T_ub / 1000 / REP, "T_lb_per_conv_s": T_lb / 1000 / REP})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--dboot", type=int, default=19)
    ap.add_argument("--lazy", action="store_true")
    args = ap.parse_args()

    print("# security coupling (dnum = 3 unless stated)")
    for logN in (15, 16, 17, 18):
        D = d_max(logN, 3)
        beta = BETA.get(logN, BETA_ASSUMED.get(logN))
        tag = "" if logN in BETA else " (beta ASSUMED)"
        print(f"log2 N = {logN}: beta = {beta}{tag}, D_max(dnum=3) = {D}, logQP = {log_qp(D, 3)[0]}, "
              f"lam = D_max - 19 = {D - 19}, D_max(dnum=D+1) = {d_max_dnum_max(logN)}")
    print()
    print("# demo point check: depth 41, dnum 3 ->", log_qp(41, 3))
    print()
    rows = []
    for logN in (16, 17, 18):
        for d in (1024, 2048, 4096):
            for L in (24, 32):
                r = cell(d, L, logN, dboot=args.dboot, lazy=args.lazy)
                rows.append(r)
                if logN == 16:      # the only 2^16 route at Delta = 59: one prime per digit
                    rows.append(cell(d, L, logN, dboot=args.dboot, lazy=args.lazy, dnum="max"))
    # the demo point: 2^17, depth 41 (extra-depth 12), bundle seeds, L = 24, d = 1024
    demo = cell(1024, 24, 17, dboot=args.dboot, lazy=args.lazy, seeds="../../ml-eval/artifacts/pbd430a_seeds.json")
    hdr = ["logN", "dnum", "d", "L", "REP", "D", "lam", "P(M)", "B_min", "boots(trace)", "M_min", "ctpt_diag",
           "rot", "ell_pt", "H2D GB", "T_lb s", "T_ub s", "boot s", "ctpt s", "rot s", "h2d s",
           "per-conv lb s", "per-conv ub s"]
    if args.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
    for r in rows:
        if not r.get("feasible"):
            vals = [r["logN"], r["dnum"], r["d"], r["L"], r["REP"], r["D"], r["lam"], f"{r['P']/1e6:.0f}"] + ["infeasible"] + [""] * (len(hdr) - 9)
        else:
            vals = [r["logN"], r["dnum"], r["d"], r["L"], r["REP"], r["D"], r["lam"], f"{r['P']/1e6:.0f}",
                    f"{r['B_min']:.1f}", f"{r['boots']:.0f} [{r['boots_min']}-{r['boots_max']}]",
                    f"{r['M_min']}", f"{r['ctpt_diag']:.0f}", f"{r['rot']:.0f}", f"{r['ell_pt']:.1f}",
                    f"{r['bytes_h2d_GB']:.0f}", f"{r['T_lb_s']:.0f}", f"{r['T_ub_s']:.0f}",
                    f"{r['t_boot_s']:.0f}", f"{r['t_pt_s']:.0f}", f"{r['t_rot_s']:.0f}", f"{r['t_h2d_s']:.0f}",
                    f"{r['T_lb_per_conv_s']:.2f}", f"{r['T_ub_per_conv_s']:.1f}"]
        if args.md:
            print("| " + " | ".join(str(v) for v in vals) + " |")
        else:
            print("\t".join(str(v) for v in vals))
    print()
    print("# demo point (2^17, depth 41 = extra-depth 12, dnum 3, bundle seeds, lanes-block, F 3, margin 4)")
    for k in ("D", "lam", "REP", "boots", "boots_min", "boots_max", "B_min", "ctpt_diag", "M_min", "rot",
              "R_min", "ell_pt", "bytes_h2d_GB", "bytes_expand_GB", "bytes_nostore_GB", "t_boot_s", "t_pt_s",
              "t_rot_s", "t_elem_s", "t_rs_s", "t_add_s", "t_h2d_s", "t_expand_s", "T_lb_s", "T_ub_s",
              "T_lb_per_conv_s", "T_ub_per_conv_s"):
        v = demo.get(k)
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"assumed": ASSUMED, "rows": rows, "demo": demo}, f, indent=1)


def demo_cell(**kw):
    return cell(1024, 24, 17, seeds="../../ml-eval/artifacts/pbd430a_seeds.json", **kw)


def d_max_dnum_max(logN):
    beta = BETA.get(logN, BETA_ASSUMED.get(logN))
    best = 0
    for D in range(1, 120):
        if log_qp(D, D + 1)[0] <= beta:
            best = D
    return best


if __name__ == "__main__":
    main()
