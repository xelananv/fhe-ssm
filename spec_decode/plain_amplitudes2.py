#!/usr/bin/env python3
"""Plaintext probe of the FHE circuit's other bootstrapped intermediates, per tick, block max over 64 lanes:
channel-mix k=W_k u2, r=W_r u2, act (cubic in k), gate (quadratic in r), act*gate, the time-mix gate input z and its
polynomial output g; and the RMSNorm mean-squares per site vs the bundle's calibrated ms_range (pbd430a_seeds.json)."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
from _native_loader_copy import load_native   # noqa: E402
import fhe_client as FC                        # noqa: E402
import numpy as np, torch                      # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--expect", required=True)
ap.add_argument("--tag", default="pbd430a"); ap.add_argument("--art-dir", default=os.path.join(os.path.dirname(HERE), "ml-eval", "artifacts"))
ap.add_argument("--seeds", default=os.path.join(os.path.dirname(HERE), "ml-eval", "artifacts", "pbd430a_seeds.json"))
ap.add_argument("--positions", type=int, default=16); ap.add_argument("--label", default="")
a = ap.parse_args()
tok = FC.get_tokenizer(); model = load_native(a.tag, a.art_dir, "cpu"); seeds = json.load(open(a.seeds))
exp = {json.loads(l)["prompt"]: json.loads(l) for l in open(a.expect)}
prompts = [x.rstrip("\n") for x in open(a.prompts) if x.strip()]
L = len(model.blocks); P = a.positions
Q = {k: np.zeros(P) for k in ("k", "r", "act", "gate", "actgate", "z", "g", "S")}
msHi = np.zeros(P); msLo = np.full(P, np.inf)          # worst ms/hi (>1 = above the calibrated range) and ms/lo (<1 = below)
def upd(key, arr):  # arr [T, ...] -> per-position max abs
    m = np.abs(arr).reshape(arr.shape[0], -1).max(axis=1); n = min(P, len(m)); Q[key][:n] = np.maximum(Q[key][:n], m[:n])
def ms_check(site, h):   # h [T, d]
    ms = (h.astype(np.float64) ** 2).mean(axis=1); rng = seeds.get(site, {}).get("ms_range")
    if not rng: return
    n = min(P, len(ms)); msHi[:n] = np.maximum(msHi[:n], ms[:n] / rng[1]); msLo[:n] = np.minimum(msLo[:n], ms[:n] / rng[0])
def block_hooks(li, b):
    def tm_pre(mod, inp): ms_check(f"L{li}.tm", inp[0][0].detach().float().numpy())
    def cm_pre(mod, inp): ms_check(f"L{li}.cm", inp[0][0].detach().float().numpy())
    def wk_hook(mod, inp, out):
        k = out[0].detach().float().numpy(); cm = b.cm
        q0, q1, q2, q3 = (p.detach().float().numpy() for p in (cm.q0, cm.q1, cm.q2, cm.q3))
        act = q0 + q1 * k + q2 * k * k + q3 * k ** 3; upd("k", k); upd("act", act); b._act = act
    def wr_hook(mod, inp, out):
        r = out[0].detach().float().numpy(); cm = b.cm
        r0, r1, r2 = (p.detach().float().numpy() for p in (cm.r0, cm.r1, cm.r2))
        gate = r0 + r1 * r + r2 * r * r; upd("r", r); upd("gate", gate)
        if hasattr(b, "_act"): upd("actgate", b._act * gate)
    def win_hook(mod, inp, out):
        x = out[0].detach().float().numpy(); tm = b.tm; aa = tm.decay().detach().float().numpy()
        S = np.zeros_like(x[0]); Ss = []
        for t in range(x.shape[0]): S = aa * S + (1.0 - aa) * x[t]; Ss.append(S.copy())
        Ss = np.array(Ss); c, dd = tm.c.detach().float().numpy(), tm.dd.detach().float().numpy()
        z = c * Ss + dd * x; p0, p1, p2 = (p.detach().float().numpy() for p in (tm.p0, tm.p1, tm.p2))
        g = z * (p0 + p1 * z + p2 * z * z); upd("S", Ss); upd("z", z); upd("g", g)
    b.tm.register_forward_pre_hook(tm_pre); b.cm.register_forward_pre_hook(cm_pre)
    b.cm.wk.register_forward_hook(wk_hook); b.cm.wr.register_forward_hook(wr_hook); b.tm.win.register_forward_hook(win_hook)
for li, b in enumerate(model.blocks): block_hooks(li, b)
with torch.no_grad():
    for p in prompts:
        ids = (tok.encode(p) + (tok.encode(exp[p]["text"]) if p in exp else []))[:P]
        model(torch.tensor(ids).unsqueeze(0))
print(f"[{a.label}] lanes {len(prompts)}, block max over lanes/channels/layers per tick; window ~0.3..20")
print("tick      : " + " ".join(f"{t:6d}" for t in range(P)))
for k in ("S", "z", "g", "k", "r", "act", "gate", "actgate"): print(f"max|{k:7s}|: " + " ".join(f"{v:6.1f}" for v in Q[k]))
print("ms/hi max : " + " ".join(f"{v:6.2f}" for v in msHi)); print("ms/lo min : " + " ".join(f"{v:6.2f}" for v in msLo))
