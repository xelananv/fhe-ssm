# SPDX-License-Identifier: MIT
"""`method: newton_const` -- 1/sqrt(x + eps) by Newton's iteration from a CONSTANT seed (P6, ROBUST_NORMS.md).

    u = (x + eps) / scale,   y_0 = 1,   y <- y (1.5 - 0.5 u y^2)  `steps` times,   1/sqrt(x + eps) = y scale^-1/2

Why a constant seed: with t = u y^2 one step maps sqrt(t) -> sqrt(t) (3 - t) / 2 <= 1, so after the first step every
iterate is below 1/sqrt(u) and rises monotonically to it; the first step stays positive iff u < 3 (y_0 = 1). There is
no input in (0, 3) for which the iteration diverges, only inputs for which `steps` is too few. The old `newton` method
starts from a Chebyshev seed fitted on [seed_lo, 1]; outside that interval the seed polynomial is wrong (it can be
negative or huge) and Newton then diverges -- the block-7 GroupNorm failure of 2026-11-04.

The pre-scale `scale` is the widened top of the argument, (calib_max R + eps), so the accuracy range is
u in [u_lo, 1] with u_lo = (calib_min / R + eps) / scale, and the divergence bound u = 3 sits at 3x the widened top.
`steps` is the least count with relative error <= tol over [u_lo, 1] (checked on a dense log grid; the error is
largest at u_lo). Same `scale` convention as `newton`: the circuit folds scale^-1/2 into the next constant.

Site JSON fields: method, eps, scale, steps, lo, hi (the accuracy range in x units), conv_lo, conv_hi (the
convergence region in x units, open), R, tol, max_rel_err, calib_min, calib_max (+ the old site's kind/short).
"""
from __future__ import annotations

import math

import numpy as np

#: the divergence bound of plain Newton from y_0 = 1: u = 3 (y_1 = 1.5 - 0.5 u = 0)
U_DIVERGE = 3.0


def newton_const_np(x, p: dict, steps: int | None = None):
    """numpy evaluation (the converter's check and the port's mirror use the same arithmetic)."""
    u = (np.asarray(x, dtype=np.float64) + p["eps"]) / p["scale"]
    y = np.ones_like(u)
    for _ in range(int(p["steps"] if steps is None else steps)):
        y = y * (1.5 - 0.5 * u * y * y)
    return y * p["scale"] ** -0.5


def rel_err_on(u_lo: float, steps: int, n: int = 4001) -> float:
    """max |sqrt(u) y_steps(u) - 1| over u in [u_lo, 1] (log grid)."""
    u = np.geomspace(u_lo, 1.0, n)
    y = np.ones_like(u)
    for _ in range(steps):
        y = y * (1.5 - 0.5 * u * y * y)
    return float(np.max(np.abs(np.sqrt(u) * y - 1.0)))


def steps_for(u_lo: float, tol: float, max_steps: int = 64) -> int:
    for s in range(1, max_steps + 1):
        if rel_err_on(u_lo, s) <= tol:
            return s
    raise ValueError(f"newton_const: u_lo={u_lo:g} needs more than {max_steps} steps for tol {tol:g}")


def newton_const_site(calib_min: float, calib_max: float, eps: float, R: float = 10.0, tol: float = 1e-3,
                      min_steps: int = 0) -> dict:
    """The site for an argument measured in [calib_min, calib_max] (x units, before eps), widened R both ways."""
    lo = max(float(calib_min), 0.0) / R
    hi = float(calib_max) * R
    scale = hi + eps
    u_lo = (lo + eps) / scale
    if u_lo <= 0.0:
        raise ValueError("newton_const: the widened bottom plus eps must be > 0 (kk_rsqrt with calib_min = 0?)")
    steps = max(steps_for(u_lo, tol), int(min_steps))
    return {"method": "newton_const", "eps": float(eps), "scale": scale, "steps": steps,
            "lo": lo, "hi": hi, "conv_lo": -float(eps), "conv_hi": U_DIVERGE * scale - eps,
            "R": float(R), "tol": float(tol), "max_rel_err": rel_err_on(u_lo, steps),
            "ratio": 1.0 / u_lo, "calib_min": float(calib_min), "calib_max": float(calib_max)}


def depth_newton_const(steps: int) -> int:
    """Multiplicative depth in the port: step 1 is affine in u (y_1 = 1.5 - 0.5 u; one scalar product), each
    later step y (1.5) + ((-u/2) y) y^2 is two levels (the d2 product order)."""
    return 1 + 2 * (int(steps) - 1)


def depth_newton_old(seed_degree: int, steps: int) -> int:
    """The old site in the port: a Chebyshev seed (BSGS, ceil(log2(d + 1)) levels) then two levels per step."""
    return int(math.ceil(math.log2(int(seed_degree) + 1))) + 2 * int(steps)
