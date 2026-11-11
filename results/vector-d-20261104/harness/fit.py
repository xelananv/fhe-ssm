# SPDX-License-Identifier: MIT
"""Fit a Chebyshev polynomial per site on its calibrated interval; write the coefficients JSON that `poly` mode reads.

Interval: [p0.1, p99.9] of the calibration (--range pct, default) or [min, max] (--range minmax), widened on each side
by --margin x width. The polynomial is Chebyshev interpolation at the d+1 Chebyshev points of the first kind on the
interval (near-minimax: within a factor ~ (2/pi) log d + 1 of the best degree-d error). Degree: either fixed
(--degree, or per-kind --site-degree overrides), or the smallest degree <= --max-degree whose max error on a dense grid
meets --target-err (absolute, or relative to max|f| on the interval with --metric rel).

Error reported per site: max_abs_err and max_rel_err over 20001 grid points of the interval (the error OFF the
interval is unbounded unless poly mode clamps).

  python fit.py --ranges ranges.json --target-err 1e-3 --max-degree 63 --out coeffs.json
  python fit.py --ranges ranges.json --degree 15 --site-degree 'ffn.relu2=31' --site-degree 'kk_rsqrt=63' --out c.json
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

from rwkv7_model import CLIENT_SIDE, site_func, site_kind, site_short


def f_np(name: str, x: np.ndarray) -> np.ndarray:
    return site_func(name)(torch.from_numpy(x.astype(np.float64))).numpy()


def cheb_fit(name: str, lo: float, hi: float, deg: int) -> np.ndarray:
    k = np.arange(deg + 1)
    t = np.cos(np.pi * (k + 0.5) / (deg + 1))                       # Chebyshev points of the first kind
    x = 0.5 * (hi - lo) * t + 0.5 * (hi + lo)
    return np.polynomial.chebyshev.chebfit(t, f_np(name, x), deg)   # exact interpolation at deg+1 points


def errors(name: str, lo: float, hi: float, coeffs: np.ndarray, n: int = 20001):
    x = np.linspace(lo, hi, n)
    t = (2 * x - lo - hi) / (hi - lo)
    fx = f_np(name, x)
    e = np.abs(np.polynomial.chebyshev.chebval(t, coeffs) - fx)
    scale = max(np.abs(fx).max(), 1e-300)
    return float(e.max()), float(e.max() / scale)


def newton_fit(name: str, lo: float, hi: float, target_rel: float | None = None, steps: int | None = None,
               seed_degree: int = 3, max_steps: int = 30, n: int = 20001) -> dict:
    """1/sqrt(x + eps) on [lo, hi] as the circuit does it: u = (x + eps)/scale with scale = hi + eps, so u is in
    (seed_lo, 1]; a Chebyshev seed of 1/sqrt(u) of degree `seed_degree` on [seed_lo, 1]; then Newton steps. Steps: fixed,
    or the fewest <= max_steps whose max relative error on a log grid of [lo, hi] meets target_rel. A seed that would make
    Newton diverge somewhere on the grid (u y0^2 outside (0, 3)) falls back to lower degrees, down to the constant 1
    (always convergent for u <= 1, but slow where u is small: that cost is what the step count reports)."""
    from rwkv7_model import RSQRT_EPS, newton_rsqrt_eval
    eps = RSQRT_EPS[site_kind(name)]
    tiny = 1e-12
    lo = max(lo, tiny)
    hi = max(hi, lo * (1 + 1e-6))
    scale = hi + eps
    seed_lo = (lo + eps) / scale
    xg = np.geomspace(lo, hi, n)
    ug = (xg + eps) / scale
    fx = 1.0 / np.sqrt(xg + eps)
    seed, used = np.array([1.0]), 0
    for d in range(seed_degree, -1, -1):
        if d == 0:
            seed, used = np.array([1.0]), 0
            break
        k = np.arange(d + 1)
        t = np.cos(np.pi * (k + 0.5) / (d + 1))
        uu = 0.5 * (1 - seed_lo) * t + 0.5 * (1 + seed_lo)
        c = np.polynomial.chebyshev.chebfit(t, 1.0 / np.sqrt(uu), d)
        y0 = np.polynomial.chebyshev.chebval((2 * ug - seed_lo - 1) / (1 - seed_lo), c)
        if np.all(y0 > 0) and np.all(ug * y0 * y0 < 3.0):
            seed, used = c, d
            break
    p = {"eps": eps, "scale": scale, "seed_lo": seed_lo, "seed_coeffs": torch.tensor(seed, dtype=torch.float64)}
    xt = torch.from_numpy(xg)

    def rel_err(k):
        y = newton_rsqrt_eval(xt, {**p, "steps": k}).numpy()
        return float(np.max(np.abs(y - fx) / fx)), float(np.max(np.abs(y - fx)))

    met = None
    if steps is None:
        for k in range(0, max_steps + 1):
            er, ea = rel_err(k)
            if target_rel is not None and er <= target_rel:
                steps, met = k, True
                break
        if steps is None:
            steps, met = max_steps, False
    er, ea = rel_err(steps)
    if met is None and target_rel is not None:
        met = er <= target_rel
    return {"method": "newton", "eps": eps, "scale": scale, "seed_lo": seed_lo, "seed_coeffs": seed.tolist(),
            "seed_degree": used, "steps": steps, "lo": lo, "hi": hi, "max_abs_err": ea, "max_rel_err": er,
            "target_met": met}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--range", choices=["pct", "minmax"], default="pct")
    ap.add_argument("--pct-lo", default="p0.1")
    ap.add_argument("--pct-hi", default="p99.9")
    ap.add_argument("--margin", type=float, default=0.0, help="widen each side by this fraction of the width")
    ap.add_argument("--degree", type=int, default=None, help="fixed degree for all sites")
    ap.add_argument("--site-degree", action="append", default=[],
                    help="'REGEX=DEG', matched against the site name (re.search); later flags win")
    ap.add_argument("--target-err", type=float, default=None)
    ap.add_argument("--site-target", action="append", default=[],
                    help="'REGEX=ERR' or 'REGEX=ERR:abs|rel': a per-site target (re.search on the name); later flags win")
    ap.add_argument("--metric", choices=["abs", "rel"], default="abs")
    ap.add_argument("--max-degree", type=int, default=63)
    ap.add_argument("--client-side", default=",".join(CLIENT_SIDE),
                    help="comma list of sites left exact (evaluated on the client); '' for none")
    ap.add_argument("--rsqrt", choices=["newton", "newton-const", "cheb"], default="newton",
                    help="1/sqrt sites: seed + Newton steps (the circuit's form, default), Newton from a constant seed "
                         "over the min/max range widened --rsqrt-range-factor both ways (P6, rsqrt_const.py), or one "
                         "Chebyshev polynomial")
    ap.add_argument("--rsqrt-range-factor", type=float, default=10.0,
                    help="newton-const: the accuracy range is [min / R, max R] of the recorded argument")
    ap.add_argument("--newton-seed-degree", type=int, default=3)
    ap.add_argument("--newton-steps", type=int, default=None, help="fixed steps; default: fewest meeting --newton-target")
    ap.add_argument("--newton-target", type=float, default=1e-3, help="max relative error of a 1/sqrt site")
    ap.add_argument("--newton-max-steps", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.degree is None and a.target_err is None and not a.site_degree and not a.site_target:
        ap.error("give --degree, --target-err, --site-degree or --site-target")

    R = json.load(open(a.ranges))
    client = {s for s in a.client_side.split(",") if s}
    overrides = []
    for s in a.site_degree:
        pat, d = s.rsplit("=", 1)
        overrides.append((re.compile(pat), int(d)))
    site_targets = []
    for s in a.site_target:
        pat, t = s.rsplit("=", 1)
        val, _, met = t.partition(":")
        site_targets.append((re.compile(pat), (float(val), met or a.metric)))

    out_sites = {}
    rows = []
    for name in R["meta"].get("site_order", list(R["sites"])):
        r = R["sites"][name]
        if name in client:
            out_sites[name] = {"exact": True, "reason": "client-side"}
            continue
        lo, hi = (r["min"], r["max"]) if a.range == "minmax" else (r[a.pct_lo], r[a.pct_hi])
        w = hi - lo
        if w <= 0:
            w = max(abs(lo), 1e-6) * 1e-3
        lo, hi = lo - a.margin * w, hi + a.margin * w
        if hi - lo <= 0:
            hi = lo + 1e-6
        if a.rsqrt == "newton-const" and name.endswith("rsqrt"):
            from rsqrt_const import newton_const_site
            from rwkv7_model import RSQRT_EPS
            nc = newton_const_site(r["min"], r["max"], RSQRT_EPS[site_kind(name)], a.rsqrt_range_factor,
                                   a.newton_target)
            out_sites[name] = {"kind": site_kind(name), "short": site_short(name), **nc}
            rows.append((name, nc["lo"], nc["hi"], -nc["steps"], float("nan"), nc["max_rel_err"], True))
            continue
        if a.rsqrt == "newton" and name.endswith("rsqrt"):
            nf = newton_fit(name, max(lo, 0.5 * r["min"]) if r["min"] > 0 else lo, hi, target_rel=a.newton_target,
                            steps=a.newton_steps, seed_degree=a.newton_seed_degree, max_steps=a.newton_max_steps)
            out_sites[name] = {"kind": site_kind(name), "short": site_short(name), **nf,
                               "calib_min": r["min"], "calib_max": r["max"]}
            rows.append((name, nf["lo"], nf["hi"], -nf["steps"], nf["max_abs_err"], nf["max_rel_err"], nf["target_met"]))
            continue
        deg = a.degree
        for pat, d in overrides:
            if pat.search(name):
                deg = d
        target, metric = a.target_err, a.metric
        for pat, t in site_targets:
            if pat.search(name):
                target, metric = t
                deg = None
        met = None
        if deg is None and target is not None:   # search on target error
            for d in range(1, a.max_degree + 1):
                c = cheb_fit(name, lo, hi, d)
                ea, er = errors(name, lo, hi, c)
                if (ea if metric == "abs" else er) <= target:
                    deg, met = d, True
                    break
            if deg is None:
                deg, met = a.max_degree, False
        if deg is None:
            deg = a.max_degree
        c = cheb_fit(name, lo, hi, deg)
        ea, er = errors(name, lo, hi, c)
        if met is None and target is not None:
            met = (ea if metric == "abs" else er) <= target
        out_sites[name] = {"kind": site_kind(name), "short": site_short(name), "lo": lo, "hi": hi, "degree": deg,
                           "coeffs": c.tolist(), "max_abs_err": ea, "max_rel_err": er, "target_met": met,
                           "calib_min": r["min"], "calib_max": r["max"]}
        rows.append((name, lo, hi, deg, ea, er, met))

    meta = {"ranges": a.ranges, "range": a.range, "pct": [a.pct_lo, a.pct_hi], "margin": a.margin,
            "degree": a.degree, "site_degree": a.site_degree, "target_err": a.target_err, "metric": a.metric,
            "max_degree": a.max_degree, "client_side": sorted(client),
            "model_weights_sha256": R["meta"].get("weights_sha256")}
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "sites": out_sites}, f, indent=1)

    print(f"{'site':34s} {'lo':>11s} {'hi':>11s} {'deg':>4s} {'max_abs_err':>12s} {'max_rel_err':>12s} met")
    for name, lo, hi, d, ea, er, met in rows:
        print(f"{name:34s} {lo:11.4g} {hi:11.4g} {d:4d} {ea:12.3e} {er:12.3e} {met}")
    print(f"wrote {a.out}: {len(rows)} polynomial sites, {len(client & set(out_sites))} client-side exact")


if __name__ == "__main__":
    main()
