# SPDX-License-Identifier: MIT
"""coeffs_mixed.json + ranges_ctx4096.json -> coeffs_mixed_nc.json: every 1/sqrt site (`method: newton`) becomes
`method: newton_const` (rsqrt_const.py, P6); every other site is copied unchanged.

  python convert_newton_const.py --coeffs coeffs_mixed.json --ranges ranges_ctx4096.json --out coeffs_mixed_nc.json \
      [--R 10] [--site-R 'gn_rsqrt=10'] [--tol 1e-3]

The calibrated range of a site is the recorded min/max of its argument (ranges JSON, x units, before eps), the same
numbers fit.py stored as calib_min/calib_max. Prints one row per site and a per-kind summary (steps, depth vs the
old site's depth in the port).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsqrt_const import depth_newton_const, depth_newton_old, newton_const_np, newton_const_site  # noqa: E402


def convert(coeffs: dict, ranges: dict, R: float = 10.0, site_R=(), tol: float = 1e-3) -> tuple[dict, list]:
    out = json.loads(json.dumps(coeffs))
    rs = ranges.get("sites", ranges)
    rows = []
    for name, s in coeffs["sites"].items():
        if s.get("method") != "newton":
            continue
        r = rs.get(name, {})
        cmin, cmax = float(r.get("min", s["calib_min"])), float(r.get("max", s["calib_max"]))
        Rs = R
        for pat, v in site_R:
            if re.search(pat, name):
                Rs = v
        nc = newton_const_site(cmin, cmax, float(s["eps"]), Rs, tol)
        # exactness check against 1/sqrt over the accuracy range (log grid in the argument)
        import numpy as np
        arg = np.geomspace(nc["lo"] + nc["eps"], nc["hi"] + nc["eps"], 2001)
        rel = float(np.max(np.abs(newton_const_np(arg - nc["eps"], nc) * np.sqrt(arg) - 1.0)))
        assert rel <= tol * (1 + 1e-9), (name, rel)
        out["sites"][name] = {k: s[k] for k in ("kind", "short") if k in s} | nc
        rows.append({"name": name, "short": s.get("short", name), "old_steps": int(s["steps"]),
                     "old_depth": depth_newton_old(len(s["seed_coeffs"]) - 1, int(s["steps"])),
                     "old_lo": float(s["lo"]), "old_hi": float(s["hi"]),
                     "steps": nc["steps"], "depth": depth_newton_const(nc["steps"]), "lo": nc["lo"], "hi": nc["hi"],
                     "conv_hi": nc["conv_hi"], "ratio": nc["ratio"], "rel": rel})
    out.setdefault("meta", {})["newton_const"] = {"R": R, "site_R": [list(x) for x in site_R], "tol": tol,
                                                  "converted": len(rows)}
    return out, rows


def summary(rows) -> dict:
    by = {}
    for r in rows:
        k = re.sub(r"^blocks\.\d+\.", "", r["name"])
        d = by.setdefault(k, {"n": 0, "old_depth": [], "depth": [], "steps": [], "old_steps": []})
        d["n"] += 1
        for f in ("old_depth", "depth", "steps", "old_steps"):
            d[f].append(r[f])
    return {k: {"n": d["n"], **{f: [min(d[f]), max(d[f]), sum(d[f]) / d["n"]] for f in
                                ("old_steps", "steps", "old_depth", "depth")}} for k, d in by.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--R", type=float, default=10.0)
    ap.add_argument("--site-R", action="append", default=[], help="REGEX=R, per-site override")
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args(argv)
    site_R = [(p, float(v)) for p, v in (x.rsplit("=", 1) for x in a.site_R)]
    out, rows = convert(json.load(open(a.coeffs)), json.load(open(a.ranges)), a.R, site_R, a.tol)
    json.dump(out, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['name']:28s} old [{r['old_lo']:.3g}, {r['old_hi']:.3g}] {r['old_steps']:2d} st d{r['old_depth']:2d}"
              f"  ->  [{r['lo']:.3g}, {r['hi']:.3g}] conv< {r['conv_hi']:.3g}  {r['steps']:2d} st d{r['depth']:2d}"
              f"  rel {r['rel']:.2e}")
    print(json.dumps(summary(rows), indent=1))
    print(f"wrote {a.out}: {len(rows)} sites converted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
