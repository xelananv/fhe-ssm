#!/usr/bin/env python3
"""Exact incremental (recurrent) evaluation of the native plaintext model, batched over lanes: one token per lane per
step at O(1) cost in the context length, carrying per layer the shift-mix input of the previous token and the scan state --
the same two quantities the FHE server carries. It is the plaintext reference for a FREE-RUNNING (autoregressive) session,
where the inputs are not known in advance and a full-prefix forward per tick would grow with the context.

Mirrors ml-eval/train_fhe_native_ssm.py (TimeMix.branch / ChannelMix.branch / NativeLM.forward) statement by statement for
the sequential block with shift_mix and the polynomial gates (CFG nonlin "poly", gates "expanded", block "sequential": the
demo model pbd430a). `selftest()` compares it with the model's own parallel forward on random token sequences.

usage (self-test):  plain_recurrent.py --art-dir ~/Documents/fhe-ssm-backup/mac_art [--tag pbd430a] [--tokens 48] [--lanes 3]
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
import numpy as np, torch                      # noqa: E402


class RecurrentRef:
    """state: per block (u_prev [B,d], S [B,d]); step(ids [B]) -> logits [B,V] (float32 torch tensor)."""

    def __init__(self, model, cfg, lanes, device="cpu", eps=1e-5):
        self.m = model; self.cfg = cfg; self.B = lanes; self.dev = device; self.eps = eps
        for k, want in (("nonlin", "poly"), ("shift_mix", True)):
            if cfg.get(k) != want: raise SystemExit(f"plain_recurrent: CFG[{k!r}] = {cfg.get(k)!r}, this mirror is written for {want!r}")
        if cfg.get("block", "sequential") == "parallel": raise SystemExit("plain_recurrent: parallel block not mirrored")
        if cfg.get("gates", "expanded") not in ("expanded", "horner"): raise SystemExit("plain_recurrent: unknown gates")
        d = cfg["d_model"]
        self.u_prev = [torch.zeros(lanes, d, device=device) for _ in model.blocks]
        self.S = [torch.zeros(lanes, d, device=device) for _ in model.blocks]
        self.alpha = float(cfg.get("alpha_res", 1.0)); self.t = 0

    def _norm(self, x, g): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * g

    @torch.no_grad()
    def step(self, ids, site_hook=None):
        """site_hook(name, x [B,d]) is called with every norm INPUT (L{i}.tm, L{i}.cm, ln_out), for range probes."""
        m = self.m; h = m.emb(torch.as_tensor(ids, device=self.dev, dtype=torch.long))
        for li, b in enumerate(m.blocks):
            tm, cm = b.tm, b.cm
            if site_hook: site_hook(f"L{li}.tm", h)
            u = self._norm(h, tm.g1)
            um = u * tm.mix + self.u_prev[li] * (1.0 - tm.mix)          # the shift uses the PRE-mix u of the previous token
            self.u_prev[li] = u
            x = tm.win(um); a = tm.decay().float()
            S = a * self.S[li] + (1.0 - a) * x.float(); self.S[li] = S
            z = tm.c * S.to(u.dtype) + tm.dd * x
            g = z * (tm.p0 + tm.p1 * z + tm.p2 * z * z)
            h = h + self.alpha * tm.wout(g)
            if site_hook: site_hook(f"L{li}.cm", h)
            u2 = self._norm(h, cm.g2); k, r = cm.wk(u2), cm.wr(u2)
            act = cm.q0 + cm.q1 * k + cm.q2 * k * k + cm.q3 * k * k * k
            gate = cm.r0 + cm.r1 * r + cm.r2 * r * r
            h = h + cm.wv(act * gate)
        if site_hook: site_hook("ln_out", h)
        self.t += 1
        return m.head(self._norm(h, m.g_out)).float()


def load(tag, art_dir, device="cpu"):
    from _native_loader_copy import load_native
    cfg = json.loads(open(os.path.join(art_dir, f"{tag}_config.json")).read())
    return load_native(tag, art_dir, device), cfg


def selftest(tag, art_dir, tokens, lanes, device="cpu"):
    model, cfg = load(tag, art_dir, device); rng = np.random.default_rng(7)
    ids = rng.integers(0, cfg["vocab"], size=(lanes, tokens))
    with torch.no_grad():
        o = model(torch.tensor(ids, device=device)); o = o if torch.is_tensor(o) else o[0]
        ref = o.float()                                                    # [B,T,V]
    rr = RecurrentRef(model, cfg, lanes, device); worst = 0.0; agree = 0
    for t in range(tokens):
        lg = rr.step(ids[:, t]); den = ref[:, t].pow(2).mean(-1).sqrt()
        rel = ((lg - ref[:, t]).pow(2).mean(-1).sqrt() / den).max().item(); worst = max(worst, rel)
        agree += int((lg.argmax(-1) == ref[:, t].argmax(-1)).sum())
    print(json.dumps({"plainRecurrentSelftest": True, "tag": tag, "lanes": lanes, "tokens": tokens, "worstRelErrRmsVsParallelForward": worst,
                      "argmaxAgree": f"{agree}/{lanes * tokens}"}))
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="pbd430a")
    ap.add_argument("--art-dir", default=os.path.join(os.path.dirname(HERE), "ml-eval", "artifacts"))
    ap.add_argument("--tokens", type=int, default=48); ap.add_argument("--lanes", type=int, default=3)
    a = ap.parse_args(); w = selftest(a.tag, os.path.expanduser(a.art_dir), a.tokens, a.lanes)
    sys.exit(0 if w < 1e-3 else 1)
