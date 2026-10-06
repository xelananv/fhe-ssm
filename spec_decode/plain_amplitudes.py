#!/usr/bin/env python3
"""Plaintext amplitude probe for the FHE circuit's two unbounded bootstrap inputs: the per-layer scan state S (the
context accumulator, carried across ticks) and the residual stream h at each block entry. For a prompt set, run the
model teacher-forced on prompt + its plaintext-greedy continuation and report, per position (= tick), the block
maximum over lanes and channels of |S| and |h| per layer, and the max over layers. The harness's own note puts the
usable bootstrap amplitude window at ~0.3..20 (gpu_real_model.cu, exp7_amp.jsonl)."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
from _native_loader_copy import load_native   # noqa: E402
import fhe_client as FC                        # noqa: E402
import numpy as np, torch                      # noqa: E402
import train_fhe_native_ssm as tr              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--expect", required=True, help="plain_greedy expectations jsonl (prompt -> text)")
ap.add_argument("--tag", default="pbd430a"); ap.add_argument("--art-dir", default=os.path.join(os.path.dirname(HERE), "ml-eval", "artifacts"))
ap.add_argument("--positions", type=int, default=16); ap.add_argument("--label", default="")
a = ap.parse_args()
tok = FC.get_tokenizer(); model = load_native(a.tag, a.art_dir, "cpu")
exp = {json.loads(l)["prompt"]: json.loads(l) for l in open(a.expect)}
prompts = [x.rstrip("\n") for x in open(a.prompts) if x.strip()]
L = len(model.blocks); P = a.positions
Smax = np.zeros((L, P)); Hmax = np.zeros((L + 1, P)); Schan = np.zeros((L, model.blocks[0].tm.c.numel()))
def win_hook(li):
    def f(mod, inp, out):                                   # out = x = W_in u, [1, T, d]; S_t = a S_{t-1} + (1-a) x_t
        x = out[0].detach().float().numpy(); aa = model.blocks[li].tm.decay().detach().float().numpy()
        S = np.zeros_like(x[0]); n = min(P, x.shape[0])
        for t in range(x.shape[0]):
            S = aa * S + (1.0 - aa) * x[t]
            if t < P: Smax[li, t] = max(Smax[li, t], np.abs(S).max()); Schan[li] = np.maximum(Schan[li], np.abs(S))
    return f
for li, b in enumerate(model.blocks): b.tm.win.register_forward_hook(win_hook(li))
def pre_hook(li):
    def f(mod, inp):
        h = inp[0][0].abs().float().numpy(); n = min(P, h.shape[0]); Hmax[li, :n] = np.maximum(Hmax[li, :n], h[:n].max(axis=1))
    return f
for li, b in enumerate(model.blocks): b.register_forward_pre_hook(pre_hook(li))
with torch.no_grad():
    for p in prompts:
        ids = tok.encode(p) + (tok.encode(exp[p]["text"]) if p in exp else [])
        ids = ids[:P]
        model(torch.tensor(ids).unsqueeze(0))
print(f"[{a.label}] lanes {len(prompts)}, positions {P}, window ~0.3..20")
print("tick :  " + " ".join(f"{t:6d}" for t in range(P)))
print("max|S|: " + " ".join(f"{v:6.1f}" for v in Smax.max(axis=0)))
print("max|h|: " + " ".join(f"{v:6.1f}" for v in Hmax.max(axis=0)))
worst = np.argsort(-Smax.max(axis=1))[:3]
for l in worst: print(f"  layer {l:2d} max|S| per tick: " + " ".join(f"{v:6.1f}" for v in Smax[l]))
print("per-layer max|S| over all ticks: " + " ".join(f"{v:.0f}" for v in Smax.max(axis=1)))
print("channels with max|S|>20 per layer: " + " ".join(str(int((Schan[l] > 20).sum())) for l in range(L)))
