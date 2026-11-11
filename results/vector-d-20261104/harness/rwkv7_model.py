# SPDX-License-Identifier: MIT
"""RWKV-7 (x070) forward in PyTorch with every non-polynomial operation routed through one hook, `site(name, x)`.

Reference: BlinkDL, `raw/rwkv_v7_demo.py` (GPT mode, the World v2.8 / v2.9 layout) and
`raw/rwkv_v7_demo_rnn.py` (RNN mode, fused decay). Line numbers below cite those files.

What is a site. Everything a CKKS circuit cannot evaluate directly: exp, sigmoid, softplus, tanh, 1/sqrt, max/relu.
Additions, products, linear layers and the time shift are not sites. The head (logits) is computed on the client after
decryption, so it is exact and not a site; `ln_out` and `ln0` are sites but are client-side by default (see
`CLIENT_SIDE`): ln0 is fused into the embedding table by the reference itself (demo_rnn.py:73), so the server never
sees it, and ln_out sits directly before the client's head.

Site kinds (the function each site applies to its input x):
  ln0.rsqrt, lnK.rsqrt, ln_out.rsqrt   1/sqrt(x + 1e-5),   x = biased variance       nn.LayerNorm, demo.py:326-328,358
  att.gn_rsqrt                         1/sqrt(x + 64e-5),  x = per-head variance     GroupNorm ln_x, demo.py:254,285
  att.kk_rsqrt                         1/max(sqrt(x),1e-12), x = per-head ||kk||^2   F.normalize, demo.py:281
  att.decay                            exp(-exp(-0.5)*sigmoid(x)), x = w0 + tanh(xw@w1)@w2
                                       = exp(-exp(-softplus(-x) - 0.5)): demo.py:270 then demo.py:179;
                                       the fused form is demo_rnn.py:138-140. One site for the whole exp/softplus/exp chain.
  att.w_tanh                           tanh(x), inside the decay LoRA                demo.py:270
  att.a_sigmoid                        sigmoid(x), in-context learning rate          demo.py:277
  att.v_sigmoid                        sigmoid(x), value-residual gate (layers >= 1) demo.py:276
  att.g_sigmoid                        sigmoid(x), inside the output-gate LoRA       demo.py:278
  ffn.relu2                            relu(x)^2                                     demo.py:313

Modes: 'exact', 'record' (exact + per-site input statistics), 'poly' (a Chebyshev polynomial per site, from a
coefficients JSON written by fit.py; sites absent from it, or marked exact, stay exact).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP_M05 = math.exp(-0.5)  # 0.6065306597...; demo_rnn.py:140 uses the rounded 0.606531

# ---------------------------------------------------------------------------------------------------------------
# Site functions. Each takes a tensor (any float dtype) and returns the exact value. Shared by the model and fit.py.
# ---------------------------------------------------------------------------------------------------------------

SITE_FUNCS = {
    "ln.rsqrt": lambda x: torch.rsqrt(x + 1e-5),
    "att.gn_rsqrt": lambda x: torch.rsqrt(x + 64e-5),
    "att.kk_rsqrt": lambda x: 1.0 / torch.clamp(torch.sqrt(torch.clamp(x, min=0.0)), min=1e-12),
    "att.decay": lambda x: torch.exp(-EXP_M05 * torch.sigmoid(x)),
    "att.w_tanh": torch.tanh,
    "att.a_sigmoid": torch.sigmoid,
    "att.v_sigmoid": torch.sigmoid,
    "att.g_sigmoid": torch.sigmoid,
    "ffn.relu2": lambda x: torch.relu(x) ** 2,
}

# Sites the server never evaluates (see module docstring). fit.py leaves them exact unless told otherwise.
CLIENT_SIDE = ("ln0.rsqrt", "ln_out.rsqrt")

# The inverse square roots, evaluated in poly mode the way the FHE circuit does it (fhe-ssm's norm: a seed, then Newton
# steps y <- y (3 - v y^2) / 2), not as one polynomial: one polynomial of 1/sqrt over a wide variance range needs a huge
# degree, while Newton's error falls quadratically once the seed is close. v = x + eps is the argument of 1/sqrt.
RSQRT_EPS = {"ln.rsqrt": 1e-5, "att.gn_rsqrt": 64e-5, "att.kk_rsqrt": 0.0}


def newton_rsqrt_eval(x: torch.Tensor, p: dict, steps: int | None = None) -> torch.Tensor:
    """1/sqrt(x + eps) by a Chebyshev seed on u = (x + eps)/scale in [seed_lo, 1] and `steps` Newton steps on u."""
    u = (x + p["eps"]) / p["scale"]
    coeffs = p["seed_coeffs"].to(device=x.device, dtype=x.dtype)
    y = cheb_eval(u, coeffs, p["seed_lo"], 1.0)
    for _ in range(p["steps"] if steps is None else steps):
        y = y * (1.5 - 0.5 * u * y * y)
    return y * (p["scale"] ** -0.5)


def newton_const_eval(x: torch.Tensor, p: dict, steps: int | None = None) -> torch.Tensor:
    """1/sqrt(x + eps) by Newton from the constant seed 1 on u = (x + eps)/scale (rsqrt_const.py, P6). In float64:
    far below the range the iterates grow ~1.5x a step and float32 is fine, but the scan compares to exact."""
    u = (x.double() + p["eps"]) / p["scale"]
    y = torch.ones_like(u)
    for _ in range(p["steps"] if steps is None else steps):
        y = y * (1.5 - 0.5 * u * y * y)
    return (y * (p["scale"] ** -0.5)).to(x.dtype)


def newton_scaled_eval(x: torch.Tensor, p: dict) -> torch.Tensor:
    """1/sqrt(x + eps) by the scaled Newton steps (tools/rwkv7/convert_newton_scaled.py): the
    constant seed, per-step (a, b): y_1 = a_0 - b_0 u, then y <- a y - b u y^3 on u = (x + eps)/scale. Float64, as
    newton_const_eval."""
    u = (x.double() + p["eps"]) / p["scale"]
    a, b = p["steps"][0]
    y = a - b * u
    for a, b in p["steps"][1:]:
        y = a * y - b * u * y * y * y
    return (y * (p["scale"] ** -0.5)).to(x.dtype)


def site_kind(name: str) -> str:
    """'blocks.3.att.a_sigmoid' -> 'att.a_sigmoid'; 'blocks.3.ln1.rsqrt' / 'ln_out.rsqrt' -> 'ln.rsqrt'."""
    base = re.sub(r"^blocks\.\d+\.", "", name)
    if re.fullmatch(r"(ln\d*|ln_out)\.rsqrt", base):
        return "ln.rsqrt"
    return base


def site_short(name: str) -> str:
    """Kind label used for ablation grouping, keeps ln1/ln2/ln0/ln_out apart: 'blocks.3.ln1.rsqrt' -> 'ln1.rsqrt'."""
    return re.sub(r"^blocks\.\d+\.", "", name)


def site_func(name: str):
    return SITE_FUNCS[site_kind(name)]


def cheb_eval(x: torch.Tensor, coeffs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Chebyshev series sum_k c_k T_k(t), t = (2x - lo - hi)/(hi - lo), by Clenshaw. Differentiable in x."""
    t = (2.0 * x - (lo + hi)) / (hi - lo)
    n = coeffs.numel()
    if n == 1:
        return torch.zeros_like(t) + coeffs[0]
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    t2 = 2.0 * t
    for k in range(n - 1, 0, -1):
        b1, b2 = t2 * b1 - b2 + coeffs[k], b1
    return t * b1 - b2 + coeffs[0]


# ---------------------------------------------------------------------------------------------------------------
# The site controller
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class SiteStats:
    n: int = 0
    vmin: float = math.inf
    vmax: float = -math.inf
    samples: list = field(default_factory=list)
    n_sampled: int = 0


class Sites:
    """Holds the mode and per-site state. One instance per model (`model.sites`)."""

    def __init__(self, mode: str = "exact", sample_per_call: int = 65536, max_samples: int = 2_000_000, seed: int = 0):
        self.mode = mode
        self.stats: dict[str, SiteStats] = {}
        self.sample_per_call = sample_per_call
        self.max_samples = max_samples
        self.gen = torch.Generator().manual_seed(seed)
        self.polys: dict[str, dict] = {}          # name -> {lo, hi, coeffs(tensor)}
        self.clamp = False
        self.only: set[str] | None = None          # restrict poly to these site names
        self.out_counts: dict[str, list] = {}      # name -> [n_total, n_outside]
        self.seen: list[str] = []                  # site names in first-call order

    # -- configuration --------------------------------------------------------------------------------------------
    def load_poly(self, cfg: dict, clamp: bool = False, only=None):
        self.polys = {}
        for name, s in cfg["sites"].items():
            if s.get("exact"):
                continue
            if s.get("method") == "newton_const":      # P6: constant seed; lo/hi = the accuracy range
                self.polys[name] = {"lo": float(s["lo"]), "hi": float(s["hi"]), "method": "newton_const",
                                    "eps": float(s["eps"]), "scale": float(s["scale"]), "steps": int(s["steps"]),
                                    "conv_lo": float(s["conv_lo"]), "conv_hi": float(s["conv_hi"])}
                continue
            if s.get("method") == "newton_scaled":     # per-step constants; the same ranges as newton_const
                self.polys[name] = {"lo": float(s["lo"]), "hi": float(s["hi"]), "method": "newton_scaled",
                                    "eps": float(s["eps"]), "scale": float(s["scale"]),
                                    "steps": [[float(a), float(b)] for a, b in s["steps"]],
                                    "conv_lo": float(s["conv_lo"]), "conv_hi": float(s["conv_hi"])}
                continue
            if s.get("method") == "newton":
                self.polys[name] = {"lo": float(s["lo"]), "hi": float(s["hi"]), "method": "newton",
                                    "eps": float(s["eps"]), "scale": float(s["scale"]), "seed_lo": float(s["seed_lo"]),
                                    "seed_coeffs": torch.tensor(s["seed_coeffs"], dtype=torch.float64),
                                    "steps": int(s["steps"])}
                continue
            self.polys[name] = {"lo": float(s["lo"]), "hi": float(s["hi"]),
                                "coeffs": torch.tensor(s["coeffs"], dtype=torch.float64)}
        self.clamp = clamp
        self.only = set(only) if only is not None else None
        self.out_counts = {}

    def reset_counts(self):
        self.out_counts = {}

    # -- the hook -------------------------------------------------------------------------------------------------
    def __call__(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.seen:
            self.seen.append(name)
        if self.mode == "record":
            self._record(name, x)
        if self.mode == "poly" and name in self.polys and (self.only is None or name in self.only):
            return self._poly(name, x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return site_func(name)(x.float())

    def _record(self, name: str, x: torch.Tensor):
        st = self.stats.setdefault(name, SiteStats())
        v = x.detach().float().reshape(-1)
        st.n += v.numel()
        st.vmin = min(st.vmin, v.min().item())
        st.vmax = max(st.vmax, v.max().item())
        if v.numel() > self.sample_per_call:
            idx = torch.randint(0, v.numel(), (self.sample_per_call,), generator=self.gen).to(v.device)
            v = v[idx]
        st.samples.append(v.cpu())
        st.n_sampled += v.numel()
        if st.n_sampled > 2 * self.max_samples:   # thin the reservoir uniformly
            allv = torch.cat(st.samples)
            keep = torch.randperm(allv.numel(), generator=self.gen)[: self.max_samples]
            st.samples = [allv[keep]]
            st.n_sampled = self.max_samples

    def _poly(self, name: str, x: torch.Tensor) -> torch.Tensor:
        p = self.polys[name]
        lo, hi = p["lo"], p["hi"]
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            with torch.no_grad():
                n_out = ((xf < lo) | (xf > hi)).sum().item()
            c = self.out_counts.setdefault(name, [0, 0])
            c[0] += xf.numel()
            c[1] += n_out
            if self.clamp:
                xf = xf.clamp(lo, hi)
            if p.get("method") == "newton":
                return newton_rsqrt_eval(xf, p)
            if p.get("method") == "newton_const":
                return newton_const_eval(xf, p)
            if p.get("method") == "newton_scaled":
                return newton_scaled_eval(xf, p)
            coeffs = p["coeffs"].to(device=xf.device, dtype=torch.float32)
            return cheb_eval(xf, coeffs, lo, hi)

    # -- reporting ------------------------------------------------------------------------------------------------
    def ranges(self, pcts=(0.1, 99.9)) -> dict:
        out = {}
        for name, st in self.stats.items():
            allv = torch.cat(st.samples).numpy().astype(np.float64)
            q = np.percentile(allv, pcts)
            out[name] = {"kind": site_kind(name), "n": st.n, "n_sampled": int(allv.size),
                         "min": st.vmin, "max": st.vmax,
                         **{f"p{p:g}": float(v) for p, v in zip(pcts, q)}}
        return out

    def outside_fractions(self) -> dict:
        return {k: (v[1] / v[0] if v[0] else 0.0) for k, v in self.out_counts.items()}


# ---------------------------------------------------------------------------------------------------------------
# The model. Parameter names and shapes are those of the reference .pth (demo.py:223-254, 300-306, 326-359).
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class RWKV7Config:
    n_layer: int
    n_embd: int
    vocab_size: int = 65536
    head_size: int = 64
    d_decay: int = 64
    d_aaa: int = 64
    d_mv: int = 32
    d_gate: int = 128
    dim_ffn: int | None = None

    @property
    def n_head(self):
        return self.n_embd // self.head_size

    @staticmethod
    def from_state_dict(sd: dict) -> "RWKV7Config":
        n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        vocab, C = sd["emb.weight"].shape
        H, N = sd["blocks.0.att.r_k"].shape
        return RWKV7Config(n_layer=n_layer, n_embd=C, vocab_size=vocab, head_size=N,
                           d_decay=sd["blocks.0.att.w1"].shape[1], d_aaa=sd["blocks.0.att.a1"].shape[1],
                           d_mv=sd.get("blocks.0.att.v1", sd.get("blocks.1.att.v1")).shape[1], d_gate=sd["blocks.0.att.g1"].shape[1],
                           dim_ffn=sd["blocks.0.ffn.key.weight"].shape[0])


def _layer_norm(sites: Sites, name: str, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)
    return xc * sites(name, var) * ln.weight + ln.bias


def wkv7_recurrence(r, w, k, v, a, b, head_size: int):
    """The x070 state recurrence, demo.py:170-203 (the non-CUDA path), fp32. All inputs (B,T,C); w is the decay
    factor itself (already through the decay site). Per head, state S (N x N), row index = value dim:
        S <- S * w[None, :] + (S a) b^T + v k^T ;   y = S r          (demo.py:189-190)
    """
    B, T, C = r.shape
    N = head_size
    H = C // N
    with torch.autocast(device_type=r.device.type, enabled=False):
        r, w, k, v, a, b = (t.float().view(B, T, H, N) for t in (r, w, k, v, a, b))
        S = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float32)
        outs = []
        for t in range(T):
            Sa = torch.einsum("bhij,bhj->bhi", S, a[:, t])
            S = S * w[:, t, :, None, :] + Sa[..., :, None] * b[:, t, :, None, :] + v[:, t, :, :, None] * k[:, t, :, None, :]
            outs.append(torch.einsum("bhij,bhj->bhi", S, r[:, t]))
        return torch.stack(outs, dim=1).reshape(B, T, C)


class TimeMix(nn.Module):
    def __init__(self, cfg: RWKV7Config, layer_id: int):
        super().__init__()
        C, H, N = cfg.n_embd, cfg.n_head, cfg.head_size
        self.layer_id, self.H, self.N = layer_id, H, N
        P = lambda *s: nn.Parameter(torch.zeros(*s))
        self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g = (P(1, 1, C) for _ in range(6))
        self.w0, self.w1, self.w2 = P(1, 1, C), P(C, cfg.d_decay), P(cfg.d_decay, C)
        self.a0, self.a1, self.a2 = P(1, 1, C), P(C, cfg.d_aaa), P(cfg.d_aaa, C)
        self.v0, self.v1, self.v2 = P(1, 1, C), P(C, cfg.d_mv), P(cfg.d_mv, C)   # unused in layer 0 (demo.py:383)
        self.g1, self.g2 = P(C, cfg.d_gate), P(cfg.d_gate, C)
        self.k_k, self.k_a, self.r_k = P(1, 1, C), P(1, 1, C), P(H, N)
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)   # demo.py:254; eps lives in the gn_rsqrt site

    def forward(self, sites: Sites, pfx: str, x, v_first):
        B, T, C = x.shape
        H, N = self.H, self.N
        xx = F.pad(x, (0, 0, 1, -1)) - x                       # time shift, demo.py:249,260
        xr, xw, xk, xv, xa, xg = (x + xx * m for m in (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g))

        r = self.receptance(xr)
        z = self.w0 + sites(pfx + "w_tanh", xw @ self.w1) @ self.w2            # demo.py:270 (tanh)
        w = sites(pfx + "decay", z)                                          # demo.py:270 + 179, fused (rnn:140)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * sites(pfx + "v_sigmoid", self.v0 + (xv @ self.v1) @ self.v2)   # demo.py:276
        a = sites(pfx + "a_sigmoid", self.a0 + (xa @ self.a1) @ self.a2)                       # demo.py:277
        g = sites(pfx + "g_sigmoid", xg @ self.g1) @ self.g2                                   # demo.py:278

        kk = (k * self.k_k).view(B, T, H, N)
        kk = (kk * sites(pfx + "kk_rsqrt", (kk * kk).sum(-1, keepdim=True))).view(B, T, C)   # demo.py:281
        k = k * (1 + (a - 1) * self.k_a)                                                       # demo.py:282

        y = wkv7_recurrence(r, w, k, v, -kk, kk * a, N)                                        # demo.py:284
        y = y.view(B, T, H, N)                                                                 # GroupNorm, demo.py:285
        yc = y - y.mean(-1, keepdim=True)
        var = (yc * yc).mean(-1, keepdim=True)
        y = (yc * sites(pfx + "gn_rsqrt", var)).view(B, T, C) * self.ln_x.weight + self.ln_x.bias
        y = y + ((r.view(B, T, H, N) * k.view(B, T, H, N) * self.r_k).sum(-1, keepdim=True)
                 * v.view(B, T, H, N)).view(B, T, C)                                           # demo.py:287
        return self.output(y * g), v_first                                                     # demo.py:288


class ChannelMix(nn.Module):
    def __init__(self, cfg: RWKV7Config):
        super().__init__()
        C = cfg.n_embd
        F_ = cfg.dim_ffn or 4 * C
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, F_, bias=False)
        self.value = nn.Linear(F_, C, bias=False)

    def forward(self, sites: Sites, pfx: str, x):
        xx = F.pad(x, (0, 0, 1, -1)) - x                       # demo.py:300,310
        k = x + xx * self.x_k
        return self.value(sites(pfx + "relu2", self.key(k)))   # demo.py:313


class Block(nn.Module):
    def __init__(self, cfg: RWKV7Config, layer_id: int):
        super().__init__()
        C = cfg.n_embd
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(C)
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.att = TimeMix(cfg, layer_id)
        self.ffn = ChannelMix(cfg)

    def forward(self, sites: Sites, x, v_first):
        p = f"blocks.{self.layer_id}."
        if self.layer_id == 0:
            x = _layer_norm(sites, "ln0.rsqrt", x, self.ln0)                       # demo.py:336-337
        dx, v_first = self.att(sites, p + "att.", _layer_norm(sites, p + "ln1.rsqrt", x, self.ln1), v_first)
        x = x + dx
        x = x + self.ffn(sites, p + "ffn.", _layer_norm(sites, p + "ln2.rsqrt", x, self.ln2))
        return x, v_first


class RWKV7(nn.Module):
    def __init__(self, cfg: RWKV7Config, sites: Sites | None = None):
        super().__init__()
        self.cfg = cfg
        self.sites = sites or Sites("exact")
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.grad_ckpt = False

    def forward(self, idx, return_hidden: bool = False):
        x = self.emb(idx)
        v_first = torch.zeros_like(x)
        hidden = []
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x, v_first = torch.utils.checkpoint.checkpoint(blk, self.sites, x, v_first, use_reentrant=False)
            else:
                x, v_first = blk(self.sites, x, v_first)
            if return_hidden:
                hidden.append(x)
        x = _layer_norm(self.sites, "ln_out.rsqrt", x, self.ln_out)
        logits = self.head(x)   # client-side, exact
        return (logits, hidden) if return_hidden else logits


def drop_unused_ln0(sd: dict) -> dict:
    """The reference builds an ln0 in every block but uses only block 0's (demo.py:326, :336-337): drop the rest."""
    return {k: v for k, v in sd.items() if not re.fullmatch(r"blocks\.([1-9]\d*)\.ln0\.(weight|bias)", k)}


def load_rwkv7(path: str, device="cpu", dtype=torch.float32, sites: Sites | None = None) -> RWKV7:
    sd = drop_unused_ln0(torch.load(path, map_location="cpu", weights_only=True))
    cfg = RWKV7Config.from_state_dict(sd)
    model = RWKV7(cfg, sites)
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    # The reference loads strict=False too (demo.py:383). Report anything beyond the known layer-0 v0/v1/v2 gap.
    bad = [m for m in missing if not re.fullmatch(r"blocks\.0\.att\.v[012]", m)]
    if bad or unexpected:
        raise RuntimeError(f"state-dict mismatch: missing={bad} unexpected={list(unexpected)}")
    return model.to(device=device, dtype=dtype).eval()


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
