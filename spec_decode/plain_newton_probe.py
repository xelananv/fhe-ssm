#!/usr/bin/env python3
"""Emulate the FHE circuit's seeded Newton rsqrt per norm site, lane and tick in plaintext: y0 = a + b*ms (the bundle's
affine seed, calibrated on a training ms_range), then `iters` steps of y <- y*(1.5 - 0.5*ms*y^2) (the harness's loop;
the ms-norm frame is a change of units that leaves the iteration invariant). Reports, per tick, the worst relative error
|y*sqrt(ms)-1| over lanes and sites, the number of divergent lane-sites (err > 0.5 or non-finite), and the first site."""
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
ap.add_argument("--positions", type=int, default=12); ap.add_argument("--label", default=""); ap.add_argument("--eps", type=float, default=1e-5)
ap.add_argument("--robust-frac", type=float, default=0.0, help="if > 0: constant seed y0 = frac * (a + b*ms_mid) with ms_mid the calibrated range's geometric middle, b = 0")
ap.add_argument("--robust-iters", type=int, default=0)
a = ap.parse_args()
tok = FC.get_tokenizer(); model = load_native(a.tag, a.art_dir, "cpu"); seeds = json.load(open(a.seeds))
exp = {json.loads(l)["prompt"]: json.loads(l) for l in open(a.expect)}
prompts = [x.rstrip("\n") for x in open(a.prompts) if x.strip()]
P = a.positions; worst = np.zeros(P); ndiv = np.zeros(P, dtype=int); first = {}
def newton(site, h, lane):   # h [T, d]
    sd = seeds.get(site)
    if not sd: return
    ms = (h.astype(np.float64) ** 2).mean(axis=1) + a.eps
    iters = int(sd["iters"])
    if a.robust_frac > 0:
        mid = float(np.sqrt(sd["ms_range"][0] * sd["ms_range"][1])); y = np.full_like(ms, a.robust_frac * (sd["a"] + sd["b"] * mid)); iters = max(iters, a.robust_iters)
    else: y = sd["a"] + sd["b"] * ms
    with np.errstate(all="ignore"):
        for _ in range(iters): y = y * (1.5 - 0.5 * ms * y * y)
        err = np.abs(y * np.sqrt(ms) - 1.0)
    n = min(P, len(err)); e = np.where(np.isfinite(err[:n]), err[:n], np.inf)
    worst[:n] = np.maximum(worst[:n], e)
    for t in np.nonzero(e > 0.5)[0]:
        ndiv[t] += 1; first.setdefault(int(t), (site, lane, float(ms[t]), sd["ms_range"], float(sd["a"] + sd["b"] * ms[t])))
state = {"lane": 0}
def hooks(li, b):
    b.tm.register_forward_pre_hook(lambda m, i: newton(f"L{li}.tm", i[0][0].detach().float().numpy(), state["lane"]))
    b.cm.register_forward_pre_hook(lambda m, i: newton(f"L{li}.cm", i[0][0].detach().float().numpy(), state["lane"]))
for li, b in enumerate(model.blocks): hooks(li, b)
# 2026-09-18: the FINAL norm (NativeLM.forward: h = rmsnorm(h, g_out) after the last block; seeds key "ln_out") has no module
# of its own, so its input is the last block's output.
model.blocks[-1].register_forward_hook(lambda m, i, o: newton("ln_out", (o[0] if isinstance(o, (tuple, list)) else o)[0].detach().float().numpy(), state["lane"]))
with torch.no_grad():
    for k, p in enumerate(prompts):
        state["lane"] = k; ids = (tok.encode(p) + (tok.encode(exp[p]["text"]) if p in exp else []))[:P]
        model(torch.tensor(ids).unsqueeze(0))
print(f"[{a.label}] lanes {len(prompts)}: seeded-Newton rsqrt emulation, per tick")
print("tick        : " + " ".join(f"{t:7d}" for t in range(P)))
print("worst relErr: " + " ".join(f"{min(v, 9e9):7.2g}" for v in worst))
print("divergent   : " + " ".join(f"{v:7d}" for v in ndiv))
for t in sorted(first)[:4]: s, lane, ms, rng, y0 = first[t]; print(f"  first divergence at tick {t}: site {s}, lane {lane}, ms {ms:.3g} (range {rng[0]:.3g}..{rng[1]:.3g}, ms/hi {ms/rng[1]:.2f}), seed y0 {y0:.3g}")
