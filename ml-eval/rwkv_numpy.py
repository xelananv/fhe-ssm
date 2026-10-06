"""
rwkv_numpy.py — minimal float64 RWKV-4 inference (teacher-forced) with a
pluggable wkv-recurrence backend:

  backend="float"  — stabilized recurrence in numpy float64 (baseline)
  backend="fhe"    — the recurrence runs under REAL OpenFHE CKKS with
                     bootstrapping via harness/fhe_eval (level-striding,
                     per-token plaintext coefficients); everything else stays
                     float64. This isolates exactly the cryptographic noise
                     contribution of the sequence-mixing path.
  backend="noise"  — calibrated Gaussian injection at a measured FHE error
                     floor (for projections beyond what we can run today).

The wkv recurrences evaluated homomorphically are the stabilized form used by
the HF reference (aa/bb states with running-max pp bookkeeping):
    aa_t = e1'_t ⊙ aa_{t-1} + e2'_t ⊙ v_t ,   bb_t = e1'_t ⊙ bb_{t-1} + e2'_t
with e1'_t = exp(pp_{t-1} + w − pp_t), e2'_t = exp(k_t − pp_t) ∈ (0,1] — a
diagonal linear recurrence whose per-token coefficients are plaintext in this
ablation (pp depends only on the k-path). NOTE: in a private deployment those
coefficients would themselves be encrypted (selectivity — Vector A in
PROJECT_BACKLOG.md); this pipeline is a noise-fidelity ablation, not a
private-inference demo.
"""
import json
import pathlib
import subprocess
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FHE_EVAL_BIN = REPO / "harness" / "build" / "fhe_eval"


def _ln(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _shift(x):
    out = np.zeros_like(x)
    out[1:] = x[:-1]
    return out


class RwkvRunner:
    def __init__(self, weights_npz: str, config_json: str):
        self.w = dict(np.load(weights_npz))
        self.cfg = json.loads(pathlib.Path(config_json).read_text())
        self.eps = self.cfg["layer_norm_epsilon"]

    def _layer_prefix(self, i):
        return f"rwkv.blocks.{i}."

    def forward(self, token_ids, backend="float", fhe_args=None, noise_sigma=0.0,
                rng_seed=1234):
        """Teacher-forced forward pass. Returns (logits (T,vocab) float64, stats)."""
        w, eps = self.w, self.eps
        T = len(token_ids)
        d = self.cfg["d_model"]
        rng = np.random.default_rng(rng_seed)
        stats = {"backend": backend, "layers": []}

        H = w["rwkv.embeddings.weight"][token_ids].astype(np.float64)  # (T, d)
        H = _ln(H, w["rwkv.blocks.0.pre_ln.weight"], w["rwkv.blocks.0.pre_ln.bias"], eps)

        for i in range(self.cfg["n_layer"]):
            p = self._layer_prefix(i)
            # ---------------- attention (time-mix) ----------------
            x = _ln(H, w[p + "ln1.weight"], w[p + "ln1.bias"], eps)
            xs = _shift(x)
            mk, mv, mr = (w[p + f"attention.time_mix_{n}"].reshape(1, d)
                          for n in ("key", "value", "receptance"))
            k = (x * mk + xs * (1 - mk)) @ w[p + "attention.key.weight"].T
            v = (x * mv + xs * (1 - mv)) @ w[p + "attention.value.weight"].T
            r = _sigmoid((x * mr + xs * (1 - mr)) @ w[p + "attention.receptance.weight"].T)

            decay = -np.exp(w[p + "attention.time_decay"])   # (d,)
            bonus = w[p + "attention.time_first"]            # (d,)

            # plaintext pp bookkeeping (depends on k-path only)
            pp_prev = np.full((T, d), -1e38)
            pp_new = np.empty((T, d))
            dec = np.empty((T, d))   # e1'_t
            e2p = np.empty((T, d))   # e2'_t
            pp = np.full(d, -1e38)
            for t in range(T):
                pp_prev[t] = pp
                ww = pp + decay
                qq = np.maximum(ww, k[t])
                dec[t] = np.exp(ww - qq)
                e2p[t] = np.exp(k[t] - qq)
                pp = qq
                pp_new[t] = qq

            # ------- the recurrence: float | fhe | noise -------
            aa_ref, bb_ref = self._recurrence_float(dec, e2p, v)
            if backend == "fhe":
                aa, bb, lstats = self._recurrence_fhe(dec, e2p, v, fhe_args or {})
                lstats["stateMaxAbsErr_aa"] = float(np.abs(aa - aa_ref).max())
                lstats["stateMaxAbsErr_bb"] = float(np.abs(bb - bb_ref).max())
                stats["layers"].append(lstats)
            elif backend == "noise":
                aa = aa_ref + rng.normal(0.0, noise_sigma, aa_ref.shape)
                bb = bb_ref + rng.normal(0.0, noise_sigma, bb_ref.shape)
            else:
                aa, bb = aa_ref, bb_ref

            # wkv output at t uses state BEFORE t plus the u-bonus branch
            aa_in = np.vstack([np.zeros((1, d)), aa[:-1]])
            bb_in = np.vstack([np.zeros((1, d)), bb[:-1]])
            ww = bonus.reshape(1, d) + k
            qq = np.maximum(pp_prev, ww)
            e1 = np.exp(pp_prev - qq)
            e2 = np.exp(ww - qq)
            wkv = (e1 * aa_in + e2 * v) / (e1 * bb_in + e2)
            H = H + (r * wkv) @ w[p + "attention.output.weight"].T

            # ---------------- feed-forward (channel-mix) ----------------
            x = _ln(H, w[p + "ln2.weight"], w[p + "ln2.bias"], eps)
            xs = _shift(x)
            mk = w[p + "feed_forward.time_mix_key"].reshape(1, d)
            mr = w[p + "feed_forward.time_mix_receptance"].reshape(1, d)
            kf = np.maximum((x * mk + xs * (1 - mk)) @ w[p + "feed_forward.key.weight"].T, 0.0) ** 2
            rf = _sigmoid((x * mr + xs * (1 - mr)) @ w[p + "feed_forward.receptance.weight"].T)
            H = H + rf * (kf @ w[p + "feed_forward.value.weight"].T)

        H = _ln(H, w["rwkv.ln_out.weight"], w["rwkv.ln_out.bias"], eps)
        logits = H @ w["head.weight"].T
        return logits, stats

    @staticmethod
    def _recurrence_float(dec, e2p, v):
        T, d = dec.shape
        aa = np.empty((T, d))
        bb = np.empty((T, d))
        a = np.zeros(d)
        b = np.zeros(d)
        for t in range(T):
            a = dec[t] * a + e2p[t] * v[t]
            b = dec[t] * b + e2p[t]
            aa[t], bb[t] = a, b
        return aa, bb

    @staticmethod
    def _recurrence_fhe(dec, e2p, v, fhe_args):
        """Run both recurrences (aa‖bb packed) under OpenFHE via harness/fhe_eval."""
        T, d = dec.shape
        S = 2 * d  # [aa-channels | bb-channels]
        dec_p = np.concatenate([dec, dec], axis=1)
        u_p = np.concatenate([e2p * v, e2p], axis=1)
        k = int(fhe_args.get("k", 8))
        margin = int(fhe_args.get("margin", 3))
        config = fhe_args.get("config", "demo")
        with tempfile.TemporaryDirectory() as td:
            binf = pathlib.Path(td) / "job.bin"
            outf = pathlib.Path(td) / "job.out.bin"
            with open(binf, "wb") as f:
                dec_p.astype("<f8").tofile(f)
                u_p.astype("<f8").tofile(f)
            t0 = time.time()
            proc = subprocess.run(
                [str(FHE_EVAL_BIN), "--bin", str(binf), "--out", str(outf),
                 "--T", str(T), "--S", str(S), "--k", str(k),
                 "--margin", str(margin), "--config", config],
                capture_output=True, text=True, check=True)
            wall = time.time() - t0
            states = np.fromfile(outf, dtype="<f8").reshape(T, S)
        summary = {}
        for line in proc.stdout.splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tokens" in j and "bootstraps" in j:
                summary = j
        summary["wallSec"] = round(wall, 2)
        return states[:, :d], states[:, d:], summary
