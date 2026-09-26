#!/usr/bin/env python3
"""
fhe_circuit_sim.py — the circuit-faithful simulator for the real-model FHE
run (REAL_MODEL_PLAN.md). This is the go/no-go instrument: it runs the
EXACT circuit the GPU harness will run — unstabilized WKV ratio form,
per-channel range control, minimax polynomials in the normalized variable,
finite Newton iterations for reciprocal/rsqrt, level-tracked bootstrap-
noise injection — in float64, so approximation and noise composition are
measured on CPU before any GPU hour is spent.

Range-control design (calibration 2026-07-17 forced all of this; see
REAL_MODEL_PLAN.md §3 + the risk register):
  * k is CENTERED per channel: exp(k) = e^{mid_c} * P((k - mid_c)) with the
    poly domain the UNIFORM [-c, c] across channels (the domain edge IS the
    clamp). The per-channel factor pi_c = e^{mid_c - ln sigma_c} is a
    plaintext multiply. sigma_c bounds |A~|, B~, |E~ v| <= ~1.
  * The raw denominator spans ~9 orders of magnitude; the clamp bounds the
    per-channel post-scale range [m_c, M_c] (measured in calibration pass
    B), and division uses Newton with the equioscillating LINEAR seed
    y0 = alpha_c + beta_c x (delta0 = ((sqrt(M/m)-1)/(sqrt(M/m)+1))^2),
    iteration count chosen per layer from the worst channel ratio.
  * The residual stream reaches |H| ~ 238 by layer 11: carried as
    H~ = H / lambda_l with per-layer calibrated lambda; LN is scale-
    invariant (eps scaled by 1/lambda^2) so the folds are exact.

Three executors share ONE control-flow description of the circuit:
  BaseOps     exact float64 (== validated rwkv_numpy baseline; `check`)
  ClampedOps  clamps + scales, exact nonlinearities  (isolates clamp cost)
  CircuitOps  polys + finite Newton + level-tracked bootstrap noise (full)

Modes: check | calibrate | fit | simulate | export | all
  python ml-eval/fhe_circuit_sim.py --mode all --model 169m --tokens 512
Level model (mirrors gpu_real_model.cu EXACTLY — keep in sync): depth=29,
levelsAfter=10; before every level-consuming op, if remaining < 5 ->
bootstrap (value += N(0, sigma_bs) iid, used = depth-levelsAfter).
sigma_bs: 4.9e-4 secure (measured, results/hpc-gpu.md §7a), 1.7e-4
structural (§1c).
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifacts"
sys.path.insert(0, str(HERE))

from remez_ablation import remez  # equioscillation-verified minimax fitter

DEPTH = 29
LEVELS_AFTER = 10
BOOT_FLOOR = 5          # boot when remaining < BOOT_FLOOR (>=4-remaining rule)
MARGIN = 1.25

MODELS = {
    "169m": {"hf": "RWKV/rwkv-4-169m-pile", "weights": "weights.npz",
             "config": "config.json"},
    "430m": {"hf": "RWKV/rwkv-4-430m-pile", "weights": "weights_430m.npz",
             "config": "config_430m.json"},
}


# ---------------------------------------------------------------------------
# data + small helpers
# ---------------------------------------------------------------------------
def load_model(model_tag):
    m = MODELS[model_tag]
    w = dict(np.load(ART / m["weights"]))
    cfg = json.loads((ART / m["config"]).read_text())
    return w, cfg


def eval_tokens(n):
    from run_eval import wikitext_tokens
    return wikitext_tokens(n)


def calib_tokens(n):
    """Calibration slice DISJOINT from the eval slice: tokens [2048:2048+n+1]
    of the WikiText-103 test text (eval uses [0:n+1])."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    path = hf_hub_download(
        repo_id="Salesforce/wikitext", repo_type="dataset",
        filename="wikitext-103-raw-v1/test-00000-of-00001.parquet")
    text = "".join(pq.read_table(path, columns=["text"]).column("text").to_pylist())
    tok = AutoTokenizer.from_pretrained(MODELS["169m"]["hf"])
    ids = tok(text[:800000], return_tensors=None)["input_ids"]
    assert len(ids) >= 2048 + n + 1, "calibration slice too short"
    return ids[2048: 2048 + n + 1]


def ln_exact(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def polyval(coeffs, x):
    return np.polynomial.polynomial.polyval(x, np.asarray(coeffs))


def to_t(x, lo, hi):
    return (2.0 * x - (lo + hi)) / (hi - lo)


def fit_poly(f, lo, hi, deg):
    """Remez minimax of f on [lo,hi], fitted/evaluated in t in [-1,1]."""
    g = lambda t: f(0.5 * (lo + hi) + 0.5 * (hi - lo) * t)   # noqa: E731
    c, emax, _ = remez(g, -1.0, 1.0, deg)
    return np.asarray(c), float(emax)


def ps_depth(deg):
    return int(np.ceil(np.log2(max(deg, 2)))) + 1  # +1: the t-affine level


def linear_rsqrt_seed(m, M):
    """Linear seed y0 = a + b*v for Newton 1/sqrt(v) on [m,M]: relative
    least-squares on a log grid. Tracks 1/sqrt across wide ranges, keeping
    v*y0^2 inside the (0,3) convergence basin where a constant seed
    (basin spread limit 3x) diverges — the constant seed NaN'd the first
    simulator round at measured var spread ~6."""
    grid = np.geomspace(m, M, 64)
    tgt = 1.0 / np.sqrt(grid)
    A = np.stack([np.ones_like(grid), grid], axis=1) / tgt[:, None]
    sol, *_ = np.linalg.lstsq(A, np.ones_like(grid), rcond=None)
    return float(sol[0]), float(sol[1])


def rsqrt_iters_for(a, b, m, M, tol=1e-3, cap=14):
    """Numerically verify basin + count Newton iterations to tol over
    [0.8m, 1.25M] (modest headroom beyond calibration).

    TWO DEFAULTS CHANGED 2026-07-19, both because the originals cost levels
    the depth-29 budget does not have (worst tag needed 29 levels for RMSNorm
    alone -- see the note at the call site in train_fhe_native_ssm.py):

    * window [m/2, 2M] -> [0.8m, 1.25M]. The old 2x-per-side headroom
      quadrupled the effective spread (worst tag M/m 46 -> 186) and with it
      the iteration count, for margin the calibration does not justify.
    * tol 1e-4 -> 1e-3. Converging the Newton rsqrt an order of magnitude
      BELOW the bootstrap noise floor is wasted depth: measured bootstrap
      error is ~1.1e-3 (structural selftest) / ~4.9e-4 (secure), so error
      below ~5e-4 is not observable in the output anyway. Each saved
      iteration is ~4 levels.

    Combined measured effect: max iters 7 -> 5, worst-tag RMSNorm cost
    29 -> 21 levels. NOTE: tol is a quality-affecting parameter -- report it
    in the paper rather than leaving it as a silent default."""
    grid = np.geomspace(0.8 * m, 1.25 * M, 256)
    y = a + b * grid
    if np.any(grid * y * y >= 3.0) or np.any(y <= 0.0):
        return cap, False
    for n in range(1, cap + 1):
        y = y * (1.5 - 0.5 * grid * y * y)
        if float(np.max(np.abs(y * np.sqrt(grid) - 1.0))) < tol:
            return n, True
    return cap, False


def newton_recip_iters(ratio, tol=1e-3):
    """Iterations for Newton 1/x from the equioscillating linear seed on a
    range with M/m = ratio: delta0 = ((sqrt(r)-1)/(sqrt(r)+1))^2, error
    squares each iteration."""
    r = max(float(ratio), 1.0 + 1e-9)
    d0 = ((np.sqrt(r) - 1.0) / (np.sqrt(r) + 1.0)) ** 2
    d, n = d0, 0
    while d > tol and n < 24:
        d, n = d * d, n + 1
    return n, d0


def linear_recip_seed(m, M):
    """Equioscillating linear seed y0 = a + b*x for Newton 1/x on [m,M]:
    the error e(x) = 1 - x(a+bx) is a quadratic in x; imposing
    e(m) = E, e(sqrt(mM)) = -E, e(M) = E gives the minimax linear seed with
    |E| = ((sqrt(M/m)-1)/(sqrt(M/m)+1))^2 < 1, so Newton always converges."""
    s = np.sqrt(m * M)
    A = np.array([[m, m * m, 1.0], [s, s * s, -1.0], [M, M * M, 1.0]])
    sol = np.linalg.solve(A, np.ones(3))   # [a, b, E]
    return float(sol[0]), float(sol[1])


# ---------------------------------------------------------------------------
# calibration collector
# ---------------------------------------------------------------------------
class Calib:
    def __init__(self):
        self.scalar = {}
        self.pc = {}

    def see(self, name, arr):
        a = np.asarray(arr, dtype=np.float64).ravel()
        s = self.scalar.setdefault(name, [])
        s.append(a if a.size <= 8000 else a[:: max(1, a.size // 8000)])

    def see_pc(self, name, arr2d, kind):
        """kind: absmax | max | min | median | p001 | p999 (per channel)."""
        a = np.asarray(arr2d, dtype=np.float64)
        ax = tuple(range(a.ndim - 1))
        red = {"absmax": lambda: np.abs(a).max(axis=ax),
               "max": lambda: a.max(axis=ax),
               "min": lambda: a.min(axis=ax),
               "median": lambda: np.median(a, axis=ax),
               "p001": lambda: np.percentile(a, 0.1, axis=0),
               "p999": lambda: np.percentile(a, 99.9, axis=0),
               "p99": lambda: np.percentile(a, 99.0, axis=0),
               "p95": lambda: np.percentile(a, 95.0, axis=0)}[kind]()
        key = (name, kind)
        if key not in self.pc:
            self.pc[key] = red
        elif kind in ("min", "p001"):
            self.pc[key] = np.minimum(self.pc[key], red)
        elif kind == "median":
            self.pc[key] = 0.5 * (self.pc[key] + red)
        else:
            self.pc[key] = np.maximum(self.pc[key], red)

    def summary(self, name):
        a = np.concatenate(self.scalar[name])
        return {"min": float(a.min()), "max": float(a.max()),
                "p001": float(np.percentile(a, 0.1)),
                "p999": float(np.percentile(a, 99.9)),
                "absmax": float(np.abs(a).max()),
                "median": float(np.median(a))}


# ---------------------------------------------------------------------------
# ONE circuit description, three executors
# ---------------------------------------------------------------------------
def rwkv_forward(w, cfg, ids, ops, calib=None, k_block=8):
    """Teacher-forced forward in the unstabilized ratio form. Mirrored by
    gpu_real_model.cu — keep the control flow in sync."""
    d = cfg["d_model"]
    L = cfg["n_layer"]
    eps = cfg["layer_norm_epsilon"]
    T = len(ids)

    H0 = w["rwkv.embeddings.weight"][ids].astype(np.float64)
    H0 = ln_exact(H0, w["rwkv.blocks.0.pre_ln.weight"],
                  w["rwkv.blocks.0.pre_ln.bias"], eps)
    H = ops.enter_stream(H0)

    for i in range(L):
        p = f"rwkv.blocks.{i}."
        # ---- time-mix -----------------------------------------------------
        x = ops.ln(H, w[p + "ln1.weight"], w[p + "ln1.bias"], eps, i, "ln1")
        xs = ops.shift(x)
        mixes = {n: w[p + f"attention.time_mix_{n}"].ravel()
                 for n in ("key", "value", "receptance")}
        k = ops.matvec(ops.mix(x, xs, mixes["key"]), w[p + "attention.key.weight"])
        v = ops.matvec(ops.mix(x, xs, mixes["value"]), w[p + "attention.value.weight"])
        r_pre = ops.matvec(ops.mix(x, xs, mixes["receptance"]),
                           w[p + "attention.receptance.weight"])
        if calib is not None:
            calib.see(f"L{i}.k", ops.raw(k))
            for kind in ("p001", "p999", "p99", "p95"):
                calib.see_pc(f"L{i}.k", ops.raw(k), kind)
            calib.see(f"L{i}.r_pre", ops.raw(r_pre))
            calib.see(f"L{i}.v", ops.raw(v)); calib.see_pc(f"L{i}.v", ops.raw(v), "absmax")
        r = ops.sigmoid(r_pre, i, "sig_r")

        decay = np.exp(-np.exp(w[p + "attention.time_decay"]))
        eu = np.exp(w[p + "attention.time_first"])

        E = ops.exp_scaled(k, i)                       # E~ = e^{k}/sigma (scaled)
        Ev = ops.mul(E, v)
        A_out, B_out = ops.scan(E, Ev, decay, k_block)
        A_in, B_in = ops.shift(A_out), ops.shift(B_out)
        pi_bonus = ops.bonus_factor(eu, i)             # plaintext e^{u} (scale-aware)
        num = ops.add(A_in, ops.mul_pt(Ev, pi_bonus))
        den = ops.add(B_in, ops.mul_pt(E, pi_bonus))
        if calib is not None:
            calib.see(f"L{i}.den", ops.raw(den))
            for kind in ("min", "max", "median"):
                calib.see_pc(f"L{i}.den", ops.raw(den), kind)
            dwarm = ops.raw(den)
            if dwarm.shape[0] > 32:                      # warm regime: skip
                calib.see_pc(f"L{i}.den_warm", dwarm[32:], "min")  # cold start
            calib.see(f"L{i}.wkv_num", ops.raw(num))
        wkv = ops.divide(num, den, i)
        if calib is not None:
            calib.see(f"L{i}.wkv", ops.raw(wkv))
        att = ops.matvec(ops.mul(r, wkv), w[p + "attention.output.weight"])
        H = ops.add_residual(H, att, i, "att")

        # ---- channel-mix --------------------------------------------------
        x = ops.ln(H, w[p + "ln2.weight"], w[p + "ln2.bias"], eps, i, "ln2")
        xs = ops.shift(x)
        mkf = w[p + "feed_forward.time_mix_key"].ravel()
        mrf = w[p + "feed_forward.time_mix_receptance"].ravel()
        kf_pre = ops.matvec(ops.mix(x, xs, mkf), w[p + "feed_forward.key.weight"])
        rf_pre = ops.matvec(ops.mix(x, xs, mrf), w[p + "feed_forward.receptance.weight"])
        if calib is not None:
            calib.see(f"L{i}.kf_pre", ops.raw(kf_pre))
            calib.see(f"L{i}.rf_pre", ops.raw(rf_pre))
        kf = ops.relu2(kf_pre, i)                       # scaled by 1/kappa
        rf = ops.sigmoid(rf_pre, i, "sig_rf")
        ffn = ops.matvec_kappa(kf, w[p + "feed_forward.value.weight"], i)
        H = ops.add_residual(H, ops.mul(rf, ffn), i, "ffn")
        if calib is not None:
            calib.see(f"L{i}.H", ops.stream_raw(H, i))
            calib.see_pc(f"L{i}.H", ops.stream_raw(H, i), "absmax")
            calib.see(f"L{i}.ln1_var", ops.last_var("ln1"))
            calib.see(f"L{i}.ln2_var", ops.last_var("ln2"))

    Hf = ops.exit_stream(H, L - 1)
    Hf = ops.ln(Hf, w["rwkv.ln_out.weight"], w["rwkv.ln_out.bias"], eps, -1, "ln_out")
    if calib is not None:
        calib.see("ln_out_var", ops.last_var("ln_out"))
    logits = ops.matvec(Hf, w["head.weight"])
    return ops.raw(logits), ops.stats()


class BaseOps:
    """Exact float64. Also the pass-A calibration executor."""
    def __init__(self):
        self._var = {}
        self.boots = 0

    # stream + values
    def enter_stream(self, x): return np.asarray(x, dtype=np.float64)
    def exit_stream(self, h, i): return h
    def stream_raw(self, h, i): return h
    def raw(self, x): return x
    def stats(self): return {"boots": self.boots, "value_absmax": None}

    # arithmetic
    def add(self, a, b): return a + b
    def mul(self, a, b): return a * b
    def mul_pt(self, a, p): return a * p
    def mix(self, x, xs, m): return x * m + xs * (1.0 - m)
    def shift(self, x):
        out = np.zeros_like(x); out[1:] = x[:-1]; return out
    def matvec(self, x, W): return x @ W.T
    def matvec_kappa(self, x, W, i): return x @ W.T
    def add_residual(self, h, delta, i, which): return h + delta

    # nonlinearities (exact)
    def ln(self, x, g, b, eps, i, name):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        self._var[name] = var.ravel().copy()
        return (x - mu) / np.sqrt(var + eps) * g + b

    def last_var(self, name): return self._var[name]
    def sigmoid(self, x, i, name): return sigmoid(x)
    def relu2(self, x, i): return np.maximum(x, 0.0) ** 2
    def exp_scaled(self, k, i): return np.exp(k)      # UNSCALED in exact mode
    def bonus_factor(self, eu, i): return eu
    def divide(self, n, d, i): return n / d
    def scan(self, E, Ev, D, kb):
        T, d = np.asarray(E).shape
        A = np.zeros((T, d)); B = np.zeros((T, d))
        a = np.zeros(d); b = np.zeros(d)
        for t in range(T):
            a = D * a + Ev[t]; b = D * b + E[t]
            A[t], B[t] = a, b
        return A, B


class ClampedOps(BaseOps):
    """Range-controlled circuit with EXACT nonlinearities: isolates the cost
    of clamps + static scales from polynomial/noise effects. Also the
    pass-B calibration executor (post-clamp den/H/var ranges)."""
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.value_absmax = 0.0

    def _track(self, x):
        self.value_absmax = max(self.value_absmax, float(np.abs(x).max()))
        return x

    def stats(self): return {"boots": self.boots, "value_absmax": self.value_absmax}

    def _lam(self, i):
        return self.spec["layers"][i]["stream_scale"]

    def enter_stream(self, x):
        return self._track(x / self._lam(0))

    def exit_stream(self, h, i):
        # stays at stream scale: the final LN is scale-invariant (its lam
        # comes from spec["ln_out"]["lam"], set in pass B)
        return h

    def stream_raw(self, h, i):
        return h * self._lam(i)

    def add_residual(self, h, delta, i, which):
        lam = self._lam(i)
        prev = self._lam(i - 1) if (which == "att" and i > 0) else lam
        if which == "att":
            return self._track(h * (prev / lam) + delta / lam)
        return self._track(h + delta / lam)

    def ln(self, x, g, b, eps, i, name):
        # Per-channel stream scale: LN reconstructs the TRUE-scale values via
        # the weighted-mask dataflow (plan §3.5): mean mask lam_c/d, centered
        # scaled x~c = x~ - mu/lam_c, var mask lam_c^2/d — numerically equal
        # to plain LN on x_true, which is what we compute here.
        lam = self._lam(i) if i >= 0 else self.spec["ln_out"]["lam"]
        xt = x * lam
        mu = xt.mean(-1, keepdims=True)
        var = ((xt - mu) ** 2).mean(-1, keepdims=True)
        self._var[name] = var.ravel().copy()
        return self._track((xt - mu) / np.sqrt(var + eps) * g + b)

    def exp_scaled(self, k, i):
        sp = self.spec["layers"][i]
        top, W = np.asarray(sp["k_top"]), sp["exp_window"]
        kc = np.clip(k - top, -W, 0.0)
        pi = np.exp(top - np.asarray(sp["ln_sigma"]))
        return self._track(np.exp(kc) * pi)

    def bonus_factor(self, eu, i):
        return eu                     # sigma already folded into E~

    def divide(self, n, d, i):
        sp = self.spec["layers"][i]
        rho = np.asarray(sp["rho"])
        return (n * rho) / (d * rho + sp.get("eps_div", 0.0))

    def matvec_kappa(self, x, W, i):
        return x @ (W * self.spec["layers"][i]["kappa"]).T

    def relu2(self, x, i):
        sp = self.spec["layers"][i]
        xc = np.clip(x, sp["kf_lo"], sp["kf_hi"])
        return self._track(np.maximum(xc, 0.0) ** 2 / sp["kappa"])

    def sigmoid(self, x, i, name):
        sp = self.spec["layers"][i][name]
        return sigmoid(np.clip(x, sp["lo"], sp["hi"]))


class Ct:
    __slots__ = ("v", "used")
    def __init__(self, v, used):
        self.v = np.asarray(v, dtype=np.float64); self.used = used


class CircuitOps(ClampedOps):
    """The full FHE circuit: ClampedOps algebra + minimax polys + finite
    Newton + level-tracked bootstrap-noise injection."""
    def __init__(self, spec, sigma_bs, rng):
        super().__init__(spec)
        self.sigma = sigma_bs
        self.rng = rng
        self.level_ops = 0
        self.first_nonfinite = None
        self._op_ctx = "start"

    def ctx(self, label):
        self._op_ctx = label

    def _finck(self, v, where):
        if self.first_nonfinite is None and not np.isfinite(v).all():
            self.first_nonfinite = f"{where}@{self._op_ctx}"

    # ---- level machinery (mirror gpu_real_model.cu) ----
    def _boot(self, c):
        c.v = c.v + self.rng.normal(0.0, self.sigma, c.v.shape)
        c.used = DEPTH - LEVELS_AFTER
        self.boots += 1

    def _spend(self, c, lv=1):
        for _ in range(lv):
            if DEPTH - c.used < BOOT_FLOOR:
                self._boot(c)
            c.used += 1
            self.level_ops += 1

    def _align(self, a, b):
        u = max(a.used, b.used); a.used = b.used = u

    # ---- overrides carrying Ct ----
    def enter_stream(self, x):
        return Ct(super().enter_stream(x), DEPTH - LEVELS_AFTER)

    def exit_stream(self, h, i):
        return h                     # stays at stream scale (see ClampedOps)

    def stream_raw(self, h, i): return h.v * self._lam(i)
    def raw(self, x): return x.v if isinstance(x, Ct) else x

    def stats(self):
        return {"boots": self.boots, "level_ops": self.level_ops,
                "value_absmax": self.value_absmax,
                "first_nonfinite": self.first_nonfinite}

    def add(self, a, b):
        self._align(a, b); return Ct(a.v + b.v, a.used)

    def mul(self, a, b):
        self._align(a, b); r = Ct(a.v * b.v, a.used); self._spend(r); return r

    def mul_pt(self, a, p):
        r = Ct(a.v * p, a.used); self._spend(r); return r

    def mix(self, x, xs, m):
        self._align(x, xs)
        r = Ct(x.v * m + xs.v * (1.0 - m), x.used); self._spend(r); return r

    def shift(self, c):
        out = np.zeros_like(c.v); out[1:] = c.v[:-1]; return Ct(out, c.used)

    def matvec(self, x, W):
        r = Ct(x.v @ W.T, x.used); self._spend(r); return r

    def matvec_kappa(self, x, W, i):
        r = Ct(x.v @ (W * self.spec["layers"][i]["kappa"]).T, x.used)
        self._spend(r); return r

    def add_residual(self, h, delta, i, which):
        lam = self._lam(i)
        prev = self._lam(i - 1) if (which == "att" and i > 0) else lam
        if which == "att" and float(np.max(np.abs(prev / lam - 1.0))) > 1e-12:
            h = Ct(h.v * (prev / lam), h.used); self._spend(h)
        self._align(h, delta)
        r = Ct(h.v + delta.v / lam, h.used)   # 1/lam folds into Wo/Wfv rows
        self._track(r.v)
        return r

    def _poly(self, c, coeffs, lo, hi):
        x = np.clip(c.v, lo, hi)
        r = Ct(polyval(coeffs, to_t(x, lo, hi)), c.used)
        self._spend(r, ps_depth(len(coeffs) - 1))
        self._track(r.v)
        return r

    def ln(self, x, g, b, eps, i, name):
        # weighted-mask LN dataflow (plan §3.5): level cost 4 + 3*iters + 1;
        # numerically identical to true-scale LN. Bounded intermediates:
        # x~ (stream), x~c = x~ - mu/lam, var~; the true-scale values only
        # ever appear inside masked reduction summands (<= |x|max/d).
        sp = (self.spec["layers"][i][name] if i >= 0 else self.spec["ln_out"])
        lam = self._lam(i) if i >= 0 else self.spec["ln_out"]["lam"]
        xt = x.v * lam                                    # value-level only
        mu = xt.mean(-1, keepdims=True)                   # mask lam/d + rot-sum
        xc = Ct(x.v - mu / lam, x.used)                   # mu/lam ptmult
        self._spend(xc, 2)                                # mean mask + mu/lam
        var = Ct(((xc.v * lam) ** 2).mean(-1, keepdims=True), xc.used)
        self._spend(var, 2)                               # square + var mask
        self._var[name] = var.v.ravel().copy()
        vv = var.v + eps
        y = Ct(sp["rsqrt_a"] + sp["rsqrt_b"] * var.v, var.used)  # linear seed
        self._spend(y)
        for _ in range(sp["rsqrt_iters"]):
            self._align(y, var)
            y = Ct(y.v * (1.5 - 0.5 * vv * y.v ** 2), y.used)
            self._spend(y, 3)
        self._align(xc, y)
        r = Ct((xc.v * lam) * y.v * g + b, xc.used)       # affine (lam*g fold)
        self._spend(r)
        self._track(xc.v)
        self._track(r.v)
        return r

    def exp_scaled(self, k, i):
        sp = self.spec["layers"][i]
        self.ctx(f"L{i}.exp")
        top, W = np.asarray(sp["k_top"]), sp["exp_window"]
        kc = Ct(np.clip(k.v - top, -W, 0.0), k.used)     # per-channel top shift (pt add, 0 lv)
        rr = sp["exp"]["squarings"]
        red = Ct(kc.v / (2.0 ** rr), kc.used)
        p = Ct(polyval(sp["exp"]["coeffs"],
                       to_t(red.v, -W / 2 ** rr, 0.0)), red.used)
        self._spend(p, ps_depth(len(sp["exp"]["coeffs"]) - 1))
        for it in range(rr):
            p = Ct(p.v * p.v, p.used); self._spend(p)
            self._finck(p.v, f"exp_sq{it}")
        pi = np.exp(top - np.asarray(sp["ln_sigma"]))
        r = Ct(p.v * pi, p.used); self._spend(r)
        self._finck(r.v, "exp_out")
        self._track(r.v)
        return r

    def sigmoid(self, x, i, name):
        sp = self.spec["layers"][i][name]
        r = self._poly(x, sp["coeffs"], sp["lo"], sp["hi"])
        return r

    def relu2(self, x, i):
        sp = self.spec["layers"][i]
        r = self._poly(x, sp["relu2_coeffs"], sp["kf_lo"], sp["kf_hi"])
        return r

    def divide(self, n, d, i):
        sp = self.spec["layers"][i]
        self.ctx(f"L{i}.divide")
        rho = np.asarray(sp["rho"])
        nn = Ct(n.v * rho, n.used); self._spend(nn)
        dd = Ct(d.v * rho + sp.get("eps_div", 0.0), d.used); self._spend(dd)
        self._finck(dd.v, "den_in")
        alpha = np.asarray(sp["recip_alpha"]); beta = np.asarray(sp["recip_beta"])
        self._align(nn, dd)
        y = Ct(alpha + beta * dd.v, dd.used); self._spend(y)   # linear seed
        for it in range(sp["recip_iters"]):
            self._align(y, dd)
            y = Ct(y.v * (2.0 - dd.v * y.v), y.used)
            self._spend(y, 2)
            self._finck(y.v, f"recip_iter{it}")
        self._align(nn, y)
        r = Ct(nn.v * y.v, nn.used); self._spend(r)
        self._finck(r.v, "wkv_out")
        self._track(r.v)
        return r

    def scan(self, E, Ev, D, kb):
        T, d = E.v.shape
        A = np.zeros((T, d)); B = np.zeros((T, d))
        self._align(E, Ev)
        a = Ct(np.zeros(d), E.used); b = Ct(np.zeros(d), E.used)
        nB = (T + kb - 1) // kb
        for blk in range(nB):
            t0, t1 = blk * kb, min((blk + 1) * kb, T)
            av, bv = a.v.copy(), b.v.copy()
            for t in range(t0, t1):
                av = D * av + Ev.v[t]; bv = D * bv + E.v[t]
                A[t], B[t] = av, bv
            a = Ct(av, a.used); b = Ct(bv, b.used)
            self._spend(a); self._spend(b)                 # 1 level per block/stream
        u = max(a.used, E.used)
        return Ct(A, u), Ct(B, u)


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def mode_check(w, cfg, tokens):
    ids = eval_tokens(min(tokens, 128))[:-1]
    logits, _ = rwkv_forward(w, cfg, ids, BaseOps())
    ok = bool(np.isfinite(logits).all())
    result = {"mode": "check", "finite": ok}
    if cfg["d_model"] == 768:
        from rwkv_numpy import RwkvRunner
        r = RwkvRunner(str(ART / "weights.npz"), str(ART / "config.json"))
        ref, _ = r.forward(ids, backend="float")
        rel = float(np.abs(logits - ref).max() / np.abs(ref).max())
        result.update({"max_rel_vs_baseline": rel, "ok": bool(rel < 1e-6)})
        if rel >= 1e-6:
            print(json.dumps(result)); sys.exit("CHECK FAILED")
    print(json.dumps(result))


def _passA(w, cfg, ids):
    calib = Calib()
    rwkv_forward(w, cfg, ids, BaseOps(), calib=calib)
    return calib


def _build_spec_from_passA(w, cfg, calib, k_clamp_pad, sig_deg, relu2_deg,
                           exp_deg, exp_window, k_top_pct="p999"):
    L = cfg["n_layer"]
    spec = {"arch": "rwkv4", "d_model": cfg["d_model"], "n_layer": L,
            "vocab": cfg["vocab_size"], "eps": cfg["layer_norm_epsilon"],
            "depth": DEPTH, "levels_after": LEVELS_AFTER,
            "boot_floor": BOOT_FLOOR, "k_block": 8, "layers": [],
            "ln_out": {"rsqrt_seed": 1.0, "rsqrt_iters": 2}}
    for i in range(L):
        p = f"rwkv.blocks.{i}."
        decay = np.exp(-np.exp(w[p + "attention.time_decay"]))
        eu = np.exp(w[p + "attention.time_first"])
        # k clamp: TOP-ANCHORED per channel. Only the high end of k matters
        # (e^k at the low end is e^{-W} relative — negligible at W=14), so
        # each channel clamps to [k_top_c - W, k_top_c] with k_top_c =
        # (top percentile)_c + pad and a SHARED window W. This keeps every
        # channel's sigma tight — the symmetric/range-centered variants
        # inflated sigma by e^{20+} for typical channels via outlier-channel
        # widths and destroyed quality (simulator rounds 1-2). k_top_pct
        # trades attention-spike saturation against denominator dynamic
        # range (the mode_sweep grid measures this tradeoff).
        k_hi_pc = calib.pc[(f"L{i}.k", k_top_pct)]
        k_top = k_hi_pc + k_clamp_pad
        W = exp_window
        v_absmax = calib.pc[(f"L{i}.v", "absmax")]
        geo = 1.0 / np.maximum(1.0 - decay, 1e-4)
        E_top = np.exp(k_top)                       # per-channel max e^k after clamp
        sigma = MARGIN * E_top * np.maximum.reduce(
            [geo * np.maximum(v_absmax, 1.0), geo, eu * np.maximum(v_absmax, 1.0), eu])
        rs = calib.summary(f"L{i}.r_pre")
        rf = calib.summary(f"L{i}.rf_pre")
        kf = calib.summary(f"L{i}.kf_pre")
        sc, serr = fit_poly(sigmoid, rs["p001"], rs["p999"], sig_deg)
        sfc, sferr = fit_poly(sigmoid, rf["p001"], rf["p999"], sig_deg)
        # relu2 domain from FULL observed range + pad: clipping at p999 cut
        # real FFN spikes (bisect: 149.9 -> 77.7 ppl on 128 tokens).
        kf_pad = 0.15 * (kf["max"] - kf["min"])
        kf_lo, kf_hi = kf["min"] - kf_pad, kf["max"] + kf_pad
        kappa = max(MARGIN * max(abs(kf_lo), abs(kf_hi)) ** 2, 1.0)
        rc, rerr = fit_poly(lambda x: np.maximum(x, 0.0) ** 2 / kappa,
                            kf_lo, kf_hi, relu2_deg)
        squar = max(0, int(np.ceil(np.log2(W / 1.6))))
        ec, eerr = fit_poly(np.exp, -W / 2 ** squar, 0.0, exp_deg)
        # PER-CHANNEL stream scale (RWKV residual streams carry outlier
        # channels — measured |H| up to ~4700 in single channels by L11
        # while typical channels are O(1); a scalar scale would crush the
        # typical channels below the bootstrap noise floor).
        lam_pc = MARGIN * np.maximum(calib.pc[(f"L{i}.H", "absmax")], 1.0)
        spec["layers"].append({
            "k_top": k_top.tolist(), "exp_window": W,
            "ln_sigma": np.log(sigma).tolist(),
            "exp": {"squarings": squar, "coeffs": ec.tolist(), "fit_err": eerr},
            "sig_r": {"lo": rs["p001"], "hi": rs["p999"],
                      "coeffs": sc.tolist(), "fit_err": serr},
            "sig_rf": {"lo": rf["p001"], "hi": rf["p999"],
                       "coeffs": sfc.tolist(), "fit_err": sferr},
            "kf_lo": kf_lo, "kf_hi": kf_hi, "kappa": kappa,
            "relu2_coeffs": rc.tolist(), "relu2_fit_err": rerr,
            "stream_scale": lam_pc.tolist(),
            # rho placeholder — pass B sets it from OBSERVED den maxima
            # (the analytic bound proved e^{several} too conservative:
            # typical den*rho landed at 1e-2..1e-4 where any eps floor
            # biases most tokens — sweep round 3).
            "rho": np.ones(cfg["d_model"]).tolist(),
            # refined by pass B:
            "recip_alpha": np.ones(cfg["d_model"]).tolist(),
            "recip_beta": np.zeros(cfg["d_model"]).tolist(),
            "recip_iters": 2, "den_ratio_worst": None,
            "ln1": {"rsqrt_a": 1.0, "rsqrt_b": 0.0, "rsqrt_iters": 2},
            "ln2": {"rsqrt_a": 1.0, "rsqrt_b": 0.0, "rsqrt_iters": 2},
        })
    return spec


def _passB_refine(w, cfg, ids, spec, recip_tol, eps_sigma_mult=8.0,
                  rho_headroom=2.0, sigma_bs_design=4.9e-4):
    """Run the CLAMPED circuit (exact division), measure post-clamp den/var
    ranges, then set per-channel rho (data-based), per-channel warm eps
    floors, Newton seeds/iterations, and LN rsqrt seeds. Design rationale
    (sweep round 3): eps must sit >= ~8 sigma_bs or noisy denominators go
    negative and Newton diverges; but a global eps above typical dens
    biases most tokens. Per-channel: floor at eps_sigma_mult*sigma_bs but
    below each channel's WARM-regime minimum, so the bias concentrates on
    cold-start tokens only; rho from observed maxima (headroom x) so
    typical dens sit O(0.1-1) where the floor is irrelevant."""
    lam_out = spec["layers"][-1]["stream_scale"]
    spec["ln_out"] = {"lam": lam_out, "rsqrt_seed": 1.0, "rsqrt_iters": 2}
    for l in spec["layers"]:
        l["eps_div"] = 0.0                     # exact division for pass B
    calib = Calib()
    rwkv_forward(w, cfg, ids, ClampedOps(_spec_arrays(spec)), calib=calib)
    worst_ratio, worst_rsqrt_iters, basin_fail = 0.0, 0, []
    eps_floor = eps_sigma_mult * sigma_bs_design
    for i, l in enumerate(spec["layers"]):
        dmax = np.maximum(calib.pc[(f"L{i}.den", "max")], 1e-12)
        dmin = np.maximum(calib.pc[(f"L{i}.den", "min")], 0.0)
        key_w = (f"L{i}.den_warm", "min")
        dwarm = calib.pc[key_w] if key_w in calib.pc else dmin
        rho = 1.0 / (rho_headroom * dmax)
        eps_c = np.maximum(eps_floor, 0.3 * dwarm * rho)
        m = 0.9 * eps_c
        M = np.full_like(m, 3.0 / rho_headroom)   # 3x beyond calibrated max
        ratio = float(np.max(M / m))
        iters, _d0 = newton_recip_iters(ratio, recip_tol)
        a_l = np.empty_like(m); b_l = np.empty_like(m)
        for c in range(len(m)):
            a_l[c], b_l[c] = linear_recip_seed(float(m[c]), float(M[c]))
        l["rho"] = rho.tolist()
        l["eps_div"] = eps_c.tolist()
        l["recip_alpha"] = a_l.tolist()
        l["recip_beta"] = b_l.tolist()
        l["recip_iters"] = int(iters)
        l["den_ratio_worst"] = ratio
        worst_ratio = max(worst_ratio, ratio)
        for name in ("ln1", "ln2"):
            vs = calib.summary(f"L{i}.{name}_var")    # TRUE-scale variance
            a, b = linear_rsqrt_seed(max(vs["min"], 1e-9) / 2.0, 2.0 * vs["max"])
            n, ok = rsqrt_iters_for(a, b, max(vs["min"], 1e-9), vs["max"])
            if not ok:
                basin_fail.append(f"L{i}.{name}")
            l[name] = {"rsqrt_a": a, "rsqrt_b": b, "rsqrt_iters": n,
                       "var_range": [vs["min"], vs["max"]]}
            worst_rsqrt_iters = max(worst_rsqrt_iters, n)
    vo = calib.summary("ln_out_var")
    a, b = linear_rsqrt_seed(max(vo["min"], 1e-9) / 2.0, 2.0 * vo["max"])
    n, ok = rsqrt_iters_for(a, b, max(vo["min"], 1e-9), vo["max"])
    if not ok:
        basin_fail.append("ln_out")
    spec["ln_out"] = {"lam": lam_out, "rsqrt_a": a, "rsqrt_b": b,
                      "rsqrt_iters": n}
    spec["rsqrt_basin_fail"] = basin_fail
    spec["worst_rsqrt_iters"] = worst_rsqrt_iters
    return spec, worst_ratio


def _spec_arrays(spec):
    import copy
    s = copy.deepcopy(spec)
    for l in s["layers"]:
        for kk in ("k_top", "ln_sigma", "rho", "recip_alpha", "recip_beta",
                   "stream_scale"):
            l[kk] = np.asarray(l[kk])
        if isinstance(l.get("eps_div"), list):
            l["eps_div"] = np.asarray(l["eps_div"])
    if isinstance(s["ln_out"].get("lam"), list):
        s["ln_out"]["lam"] = np.asarray(s["ln_out"]["lam"])
    return s


def mode_calibrate_fit(w, cfg, tag, calib_n, k_clamp, sig_deg, relu2_deg,
                       exp_deg, recip_tol, eps_div, exp_window,
                       k_top_pct="p999"):
    ids = calib_tokens(calib_n)[:-1]
    calA = _passA(w, cfg, ids)
    spec = _build_spec_from_passA(w, cfg, calA, k_clamp, sig_deg, relu2_deg,
                                  exp_deg, exp_window, k_top_pct)
    spec, worst_ratio = _passB_refine(w, cfg, ids, spec, recip_tol,
                                      eps_sigma_mult=eps_div / 4.9e-4
                                      if eps_div > 0.05e-3 else 8.0)
    (ART / f"circuit_{tag}.json").write_text(json.dumps(spec))
    report = {
        "mode": "fit", "tag": tag, "k_clamp": k_clamp, "eps_div": eps_div,
        "worst_den_ratio": worst_ratio,
        "recip_iters": [l["recip_iters"] for l in spec["layers"]],
        "exp_squarings": spec["layers"][0]["exp"]["squarings"],
        "worst_sig_fit": max(l["sig_r"]["fit_err"] for l in spec["layers"]),
        "worst_relu2_fit": max(l["relu2_fit_err"] for l in spec["layers"]),
        "worst_exp_fit": max(l["exp"]["fit_err"] for l in spec["layers"]),
        "exp_window": spec["layers"][0]["exp_window"],
        "stream_scale_max": [round(float(np.max(l["stream_scale"])), 1)
                             for l in spec["layers"]],
        "worst_rsqrt_iters": spec["worst_rsqrt_iters"],
        "rsqrt_basin_fail": spec["rsqrt_basin_fail"],
        "file": f"circuit_{tag}.json",
    }
    print(json.dumps(report))
    return spec


def mode_sweep(w, cfg, tag, calib_n, sweep_tokens, sigma_bs,
               sig_deg, relu2_deg, exp_deg, recip_tol, k_clamp_pad):
    """Grid over (k_top_pct, exp_window, eps_div): the denominator dynamic
    range vs noise-floor tradeoff cannot be reasoned out — measure it.
    Metric = FULL CircuitOps ppl (noise included). Writes the winning spec
    to circuit_{tag}.json and the grid to sweep_{tag}.json."""
    calib_ids = calib_tokens(calib_n)[:-1]
    calA = _passA(w, cfg, calib_ids)
    ids = eval_tokens(sweep_tokens)
    inputs, targets = ids[:-1], np.array(ids[1:])
    base_logits, _ = rwkv_forward(w, cfg, inputs, BaseOps())
    grid = []
    best = None
    for pct, W in (("p999", 8.0), ("p999", 14.0), ("p99", 8.0)):
        spec0 = _build_spec_from_passA(w, cfg, calA, k_clamp_pad, sig_deg,
                                       relu2_deg, exp_deg, W, pct)
        for eps_mult in (8.0, 16.0):
            for headroom in (2.0, 4.0):
                import copy
                spec, _ = _passB_refine(w, cfg, calib_ids, copy.deepcopy(spec0),
                                        recip_tol, eps_mult, headroom, sigma_bs)
                sa = _spec_arrays(spec)
                cl, _ = rwkv_forward(w, cfg, inputs, ClampedOps(sa))
                ops = CircuitOps(sa, sigma_bs, np.random.default_rng(1234))
                sim, st = rwkv_forward(w, cfg, inputs, ops)
                qc = _quality(base_logits, cl, targets)
                qs = _quality(base_logits, sim, targets)
                row = {"k_top_pct": pct, "exp_window": W,
                       "eps_sigma_mult": eps_mult, "rho_headroom": headroom,
                       "clamped_ppl": qc["ppl"], "circuit_ppl": qs["ppl"],
                       "circuit_top1": qs["top1"],
                       "boots_per_token": st["boots"] / max(len(inputs), 1),
                       "recip_iters": spec["layers"][0]["recip_iters"],
                       "value_absmax": st["value_absmax"]}
                grid.append(row)
                print(json.dumps(row), flush=True)
                ppl = qs["ppl"]
                if np.isfinite(ppl) and (best is None or ppl < best[0]):
                    best = (ppl, spec)
    (ART / f"sweep_{tag}.json").write_text(json.dumps(grid))
    if best is not None:
        (ART / f"circuit_{tag}.json").write_text(json.dumps(best[1]))
        print(json.dumps({"mode": "sweep", "best_circuit_ppl": best[0],
                          "ppl_float": _quality(base_logits, base_logits,
                                                targets)["ppl"],
                          "written": f"circuit_{tag}.json"}))
    else:
        print(json.dumps({"mode": "sweep", "best": None,
                          "note": "NO combo produced finite ppl"}))
    return grid


def _quality(base_logits, logits, targets):
    from run_eval import nll_stats
    nll_b = nll_stats(base_logits, targets); nll_s = nll_stats(logits, targets)
    ppl_b, ppl_s = float(np.exp(nll_b.mean())), float(np.exp(nll_s.mean()))
    top1 = float((base_logits.argmax(-1) == logits.argmax(-1)).mean())
    m = base_logits.max(-1, keepdims=True)
    lp_b = base_logits - (m + np.log(np.exp(base_logits - m).sum(-1, keepdims=True)))
    m2 = logits.max(-1, keepdims=True)
    lp_s = logits - (m2 + np.log(np.exp(logits - m2).sum(-1, keepdims=True)))
    kl = float((np.exp(lp_b) * (lp_b - lp_s)).sum(-1).mean())
    return {"ppl_float": ppl_b, "ppl": ppl_s,
            "ppl_delta_pct": 100.0 * (ppl_s - ppl_b) / ppl_b,
            "top1": top1, "mean_kl": kl}


def mode_simulate(w, cfg, tokens, tag, sigma_bs, seed=1234):
    spec = _spec_arrays(json.loads((ART / f"circuit_{tag}.json").read_text()))
    ids = eval_tokens(tokens)
    inputs, targets = ids[:-1], np.array(ids[1:])
    base_logits, _ = rwkv_forward(w, cfg, inputs, BaseOps())
    clamped_logits, cst = rwkv_forward(w, cfg, inputs, ClampedOps(spec))
    ops = CircuitOps(spec, sigma_bs, np.random.default_rng(seed))
    sim_logits, st = rwkv_forward(w, cfg, inputs, ops)
    q_clamp = _quality(base_logits, clamped_logits, targets)
    q_sim = _quality(base_logits, sim_logits, targets)
    report = {
        "mode": "simulate", "tag": tag, "tokens": len(inputs),
        "sigma_bs": sigma_bs,
        "clamped_exact": q_clamp,                # clamp/scale cost alone
        "full_circuit": q_sim,                   # polys + Newton + noise
        "boots_total": st["boots"],
        "boots_per_token": st["boots"] / max(len(inputs), 1),
        "value_absmax": st["value_absmax"],
        "gate_G0": bool(abs(q_sim["ppl_delta_pct"]) < 5.0
                        and q_sim["top1"] > 0.97),
    }
    (ART / f"sim_report_{tag}_{sigma_bs:.0e}.json").write_text(json.dumps(report))
    print(json.dumps(report))
    return report


def mode_export(w, cfg, tag):
    spec = json.loads((ART / f"circuit_{tag}.json").read_text())
    L = cfg["n_layer"]
    tensors, blobs, off = {}, [], 0

    def put(name, arr):
        nonlocal off
        a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
        tensors[name] = {"offset": off, "shape": list(a.shape)}
        blobs.append(a.tobytes()); off += a.nbytes

    put("embeddings", w["rwkv.embeddings.weight"])
    put("pre_ln.g", w["rwkv.blocks.0.pre_ln.weight"])
    put("pre_ln.b", w["rwkv.blocks.0.pre_ln.bias"])
    for i in range(L):
        p = f"rwkv.blocks.{i}."
        s = spec["layers"][i]
        put(f"L{i}.ln1.g", w[p + "ln1.weight"]); put(f"L{i}.ln1.b", w[p + "ln1.bias"])
        put(f"L{i}.ln2.g", w[p + "ln2.weight"]); put(f"L{i}.ln2.b", w[p + "ln2.bias"])
        for nm, wn in (("Wk", "attention.key.weight"),
                       ("Wv", "attention.value.weight"),
                       ("Wr", "attention.receptance.weight"),
                       ("Wo", "attention.output.weight"),
                       ("Wfk", "feed_forward.key.weight"),
                       ("Wfr", "feed_forward.receptance.weight")):
            put(f"L{i}.{nm}", w[p + wn])
        put(f"L{i}.Wfv_kappa", np.asarray(w[p + "feed_forward.value.weight"]) * s["kappa"])
        for nm, wn in (("mix_k", "attention.time_mix_key"),
                       ("mix_v", "attention.time_mix_value"),
                       ("mix_r", "attention.time_mix_receptance"),
                       ("fmix_k", "feed_forward.time_mix_key"),
                       ("fmix_r", "feed_forward.time_mix_receptance")):
            put(f"L{i}.{nm}", np.asarray(w[p + wn]).ravel())
        put(f"L{i}.decay", np.exp(-np.exp(w[p + "attention.time_decay"])))
        put(f"L{i}.eu", np.exp(w[p + "attention.time_first"]))
        for nm in ("k_top", "ln_sigma", "rho", "recip_alpha", "recip_beta",
                   "stream_scale"):
            put(f"L{i}.{nm}", np.asarray(s[nm]))
        put(f"L{i}.pi", np.exp(np.asarray(s["k_top"]) - np.asarray(s["ln_sigma"])))
    put("ln_out.g", w["rwkv.ln_out.weight"]); put("ln_out.b", w["rwkv.ln_out.bias"])
    put("ln_out.lam", np.asarray(spec["ln_out"]["lam"]))
    put("head", w["head.weight"])

    # ---- flattened spec values so gpu_real_model.cu needs NO JSON parser --
    for i, s in enumerate(spec["layers"]):
        put(f"L{i}.exp.meta", [s["exp"]["squarings"], s["exp_window"]])
        put(f"L{i}.exp.coeffs", s["exp"]["coeffs"])
        for nm in ("sig_r", "sig_rf"):
            put(f"L{i}.{nm}.meta", [s[nm]["lo"], s[nm]["hi"]])
            put(f"L{i}.{nm}.coeffs", s[nm]["coeffs"])
        put(f"L{i}.relu2.meta", [s["kf_lo"], s["kf_hi"]])
        put(f"L{i}.relu2.coeffs", s["relu2_coeffs"])
        put(f"L{i}.recip.meta", [s["recip_iters"]])
        put(f"L{i}.eps_div", np.asarray(s["eps_div"]))
        for nm in ("ln1", "ln2"):
            put(f"L{i}.{nm}.rsqrt",
                [s[nm]["rsqrt_a"], s[nm]["rsqrt_b"], s[nm]["rsqrt_iters"]])
    put("ln_out.rsqrt", [spec["ln_out"]["rsqrt_a"], spec["ln_out"]["rsqrt_b"],
                         spec["ln_out"]["rsqrt_iters"]])
    put("meta", [spec["d_model"], spec["n_layer"], spec["vocab"],
                 spec["eps"], spec["depth"], spec["levels_after"],
                 spec["boot_floor"], spec["k_block"]])

    (ART / f"bundle_{tag}.bin").write_bytes(b"".join(blobs))
    manifest = {"spec": f"circuit_{tag}.json", "bin": f"bundle_{tag}.bin",
                "total_bytes": off, "tensors": tensors}
    (ART / f"bundle_{tag}.manifest.json").write_text(json.dumps(manifest))
    # plain-text index for the CUDA loader: "name offset n_elems"
    lines = [f"{k} {v['offset']} {int(np.prod(v['shape'])) if v['shape'] else 1} "
             f"{' '.join(str(x) for x in v['shape'])}"
             for k, v in tensors.items()]
    (ART / f"bundle_{tag}.index.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"mode": "export", "bytes": off, "tensors": len(tensors)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["check", "fit", "sweep", "simulate", "export",
                             "all"])
    ap.add_argument("--sweep-tokens", type=int, default=128)
    ap.add_argument("--k-top-pct", default="p999",
                    choices=["p999", "p99", "p95"])
    ap.add_argument("--model", default="169m", choices=list(MODELS))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--calib-tokens", type=int, default=512)
    ap.add_argument("--sigma-bs", type=float, default=4.9e-4)
    ap.add_argument("--k-clamp", type=float, default=0.5,
                    help="PAD added to each layer's largest per-channel "
                         "p001/p999 half-width (the clamp itself is "
                         "data-derived per channel)")
    ap.add_argument("--sig-deg", type=int, default=15)
    ap.add_argument("--relu2-deg", type=int, default=11)
    ap.add_argument("--exp-deg", type=int, default=7)
    ap.add_argument("--recip-tol", type=float, default=1e-3)
    ap.add_argument("--exp-window", type=float, default=14.0,
                    help="shared exp clamp window W: each channel keeps "
                         "[k_top_c - W, k_top_c]; low-side loss is e^{-W} "
                         "relative (negligible at 14)")
    ap.add_argument("--eps-div", type=float, default=1e-3,
                    help="plaintext floor added to rho-normalized WKV "
                         "denominators (cold-start regularization; the "
                         "simulator measures its quality cost)")
    args = ap.parse_args()

    w, cfg = load_model(args.model)
    tag = args.model
    if args.mode in ("check", "all"):
        mode_check(w, cfg, args.tokens)
    if args.mode in ("fit", "all"):
        mode_calibrate_fit(w, cfg, tag, args.calib_tokens, args.k_clamp,
                           args.sig_deg, args.relu2_deg, args.exp_deg,
                           args.recip_tol, args.eps_div, args.exp_window,
                           args.k_top_pct)
    if args.mode == "sweep":
        mode_sweep(w, cfg, tag, args.calib_tokens, args.sweep_tokens,
                   args.sigma_bs, args.sig_deg, args.relu2_deg, args.exp_deg,
                   args.recip_tol, args.k_clamp)
    if args.mode in ("simulate", "all"):
        mode_simulate(w, cfg, args.tokens, tag, args.sigma_bs)
    if args.mode in ("export", "all"):
        mode_export(w, cfg, tag)


if __name__ == "__main__":
    main()
