#!/usr/bin/env python3
"""Before paying for a long AUTOREGRESSIVE FHE run: does free-running greedy text keep the circuit's only approximated op
(the seeded Newton rsqrt of every RMSNorm site) and the carried magnitudes inside their working ranges for N tokens?

Plaintext only. For each prompt (one lane each) the model generates N tokens greedily with the exact recurrent mirror
(plain_recurrent.RecurrentRef); at every tick and every norm site (L{i}.tm, L{i}.cm, ln_out) the lane's mean-square is pushed
through the harness's Newton iteration y <- y(1.5 - 0.5 ms y^2) from the seed the server would use:
  --robust-frac F --robust-iters K : constant seed F*(a + b*ms_mid), max(iters, K) steps  (the demo's --newton-robust 0.3 8)
  (default)                         : the bundle's affine seed a + b*ms
and the relative error |y*sqrt(ms) - 1| is recorded. Also recorded per tick: max|scan state| over lanes/layers/channels (the
bootstrap window is ~0.3..20), max|residual|, max ms/hi and min ms/lo against the calibrated ms_range, and how many lanes
are in a repetition loop (<= 3 distinct tokens in the last 24).
--flip-prob P --flip-margin M : PERTURBED trajectories -- when the plaintext top-1 margin is below M logits, take the
  runner-up with probability P (an FHE near-tie flip sends the session down a different path; the ranges must hold there too).
Writes one JSON line per lane (prompt, generated text, first-divergence info) to --out and a per-tick table to stdout."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
import numpy as np, torch                      # noqa: E402
import fhe_client as FC                        # noqa: E402
from plain_recurrent import RecurrentRef, load  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--tag", default="pbd430a"); ap.add_argument("--art-dir", default=os.path.expanduser("~/Documents/fhe-ssm-backup/mac_art"))
ap.add_argument("--seeds", default=None); ap.add_argument("--eps", type=float, default=1e-5)
ap.add_argument("--robust-frac", type=float, default=0.0); ap.add_argument("--robust-iters", type=int, default=0)
ap.add_argument("--flip-prob", type=float, default=0.0); ap.add_argument("--flip-margin", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--label", default="")
ap.add_argument("--candidates", default="", help="extra seed settings evaluated on the SAME trajectory, e.g. '0.3:8,0.2:10,0.15:11,0.1:12' (frac:iters)")
a = ap.parse_args()
CAND = [(float(c.split(":")[0]), int(c.split(":")[1])) for c in a.candidates.split(",") if c.strip()]
cand_worst = {c: 0.0 for c in CAND}; cand_div = {c: 0 for c in CAND}; cand_lanes = {c: set() for c in CAND}
lane_hi = {}; lane_lo = {}   # per lane: (max ms/hi, site, tick), (min ms/lo, site, tick)
tok = FC.get_tokenizer(); model, cfg = load(a.tag, a.art_dir)
seeds = json.load(open(a.seeds or os.path.join(a.art_dir, f"{a.tag}_seeds.json")))
prompts = [l.rstrip("\n") for l in open(a.prompts) if l.strip() and not l.startswith("# type:")]
ids = [tok.encode(p) for p in prompts]; B = len(ids); P = [len(x) for x in ids]; T = max(P) + a.tokens
rr = RecurrentRef(model, cfg, B); rng = np.random.default_rng(a.seed)
fed = [list(x[:1]) for x in ids]; gen = [[] for _ in range(B)]
worst = np.zeros(T); ndiv = np.zeros(T, dtype=int); maxS = np.zeros(T); maxH = np.zeros(T); hi_r = np.zeros(T); lo_r = np.full(T, np.inf)
loops = np.zeros(T, dtype=int); flips = 0; first = {}
tick = {"t": 0}
def site(name, x):
    sd = seeds.get(name)
    if not sd: return
    t = tick["t"]; ms = (x.double() ** 2).mean(-1).numpy() + a.eps; iters = int(sd["iters"])
    if a.robust_frac > 0:
        mid = float(np.sqrt(sd["ms_range"][0] * sd["ms_range"][1])); y = np.full_like(ms, a.robust_frac * (sd["a"] + sd["b"] * mid)); iters = max(iters, a.robust_iters)
    else: y = sd["a"] + sd["b"] * ms
    with np.errstate(all="ignore"):
        for _ in range(iters): y = y * (1.5 - 0.5 * ms * y * y)
        err = np.abs(y * np.sqrt(ms) - 1.0)
    e = np.where(np.isfinite(err), err, np.inf); worst[t] = max(worst[t], e.max()); bad = np.nonzero(e > 0.5)[0]; ndiv[t] += len(bad)
    for l in bad: first.setdefault(int(l), {"tick": t, "site": name, "ms": float(ms[l]), "ms_range": sd["ms_range"]})
    hi_r[t] = max(hi_r[t], float((ms / sd["ms_range"][1]).max())); lo_r[t] = min(lo_r[t], float((ms / sd["ms_range"][0]).min()))
    rh = ms / sd["ms_range"][1]; rl = ms / sd["ms_range"][0]
    for l in range(len(ms)):
        if l not in lane_hi or rh[l] > lane_hi[l][0]: lane_hi[l] = (float(rh[l]), name, t)
        if l not in lane_lo or rl[l] < lane_lo[l][0]: lane_lo[l] = (float(rl[l]), name, t)
    mid = float(np.sqrt(sd["ms_range"][0] * sd["ms_range"][1]))
    for (fc, kc) in CAND:
        yc = np.full_like(ms, fc * (sd["a"] + sd["b"] * mid))
        with np.errstate(all="ignore"):
            for _ in range(max(int(sd["iters"]), kc)): yc = yc * (1.5 - 0.5 * ms * yc * yc)
            ec = np.abs(yc * np.sqrt(ms) - 1.0)
        ec = np.where(np.isfinite(ec), ec, np.inf); cand_worst[(fc, kc)] = max(cand_worst[(fc, kc)], float(ec.max()))
        bad_c = np.nonzero(ec > 1e-3)[0]; cand_div[(fc, kc)] += len(bad_c); cand_lanes[(fc, kc)].update(int(x) for x in bad_c)
    maxH[t] = max(maxH[t], float(x.abs().max()))
for t in range(T):
    tick["t"] = t
    lg = rr.step([f[-1] for f in fed], site_hook=site)
    maxS[t] = max(float(s.abs().max()) for s in rr.S)
    top2 = torch.topk(lg, 2, dim=-1)
    for l in range(B):
        if t + 1 < P[l]: nxt = ids[l][t + 1]                                   # still feeding the prompt
        else:
            nxt = int(top2.indices[l, 0])
            if a.flip_prob > 0 and float(top2.values[l, 0] - top2.values[l, 1]) < a.flip_margin and rng.random() < a.flip_prob:
                nxt = int(top2.indices[l, 1]); flips += 1
            gen[l].append(nxt)
        fed[l].append(nxt)
        if len(gen[l]) >= 24 and len(set(gen[l][-24:])) <= 3: loops[t] += 1
with open(a.out, "w") as f:
    for l in range(B):
        f.write(json.dumps({"lane": l, "prompt": prompts[l], "promptTokens": P[l], "genTokens": len(gen[l]), "text": tok.decode(gen[l]), "gen_ids": gen[l],
                            "firstDivergence": first.get(l)}) + "\n")
print(f"[{a.label}] lanes {B}, ticks {T} (prompts {min(P)}-{max(P)} tokens + {a.tokens} generated), seed mode: " + (f"robust {a.robust_frac}/{a.robust_iters}" if a.robust_frac > 0 else "bundle affine") + (f", perturbed: flip prob {a.flip_prob} below margin {a.flip_margin} -> {flips} flips" if a.flip_prob > 0 else ""))
print(f"Newton rsqrt over all sites/lanes/ticks: worst relErr {worst.max():.3g} at tick {int(worst.argmax())}; divergent lane-site events {int(ndiv.sum())}; lanes that ever diverge {len(first)}")
print(f"max|scan state| {maxS.max():.2f} at tick {int(maxS.argmax())}; max|norm input| {maxH.max():.2f}; max ms/hi {hi_r.max():.2f} at tick {int(hi_r.argmax())}; min ms/lo {lo_r.min():.3f} at tick {int(lo_r.argmin())}")
print(f"lanes in a repetition loop (<= 3 distinct tokens in the last 24): at tick {T // 4}: {loops[T // 4]}, {T // 2}: {loops[T // 2]}, {T - 1}: {loops[T - 1]} of {B}")
print("tick | worst Newton relErr | divergent | max|S| | max ms/hi | min ms/lo | lanes looping")
for t in list(range(0, T, max(T // 25, 1))) + [T - 1]:
    print(f"{t:4d} | {worst[t]:.2e} | {ndiv[t]} | {maxS[t]:.2f} | {hi_r[t]:.2f} | {lo_r[t]:.3f} | {loops[t]}")
for l in sorted(first)[:8]: print("  first divergence lane", l, first[l], "prompt:", prompts[l][:60])
print("lanes with the largest ms/hi (site, tick):"); 
for l in sorted(lane_hi, key=lambda q: -lane_hi[q][0])[:6]: print(f"  lane {l}: ms/hi {lane_hi[l][0]:.2f} at {lane_hi[l][1]} tick {lane_hi[l][2]} | {prompts[l][:50]!r} -> {tok.decode(gen[l][:12])!r}...")
print("lanes with the smallest ms/lo (site, tick):")
for l in sorted(lane_lo, key=lambda q: lane_lo[q][0])[:4]: print(f"  lane {l}: ms/lo {lane_lo[l][0]:.3f} at {lane_lo[l][1]} tick {lane_lo[l][2]}")
if CAND:
    print("candidate seed settings on this trajectory (event = relErr > 1e-3 at a lane-site-tick):")
    for c in CAND: print(f"  frac {c[0]} iters {c[1]}: worst relErr {cand_worst[c]:.3g}, events {cand_div[c]}, lanes {sorted(cand_lanes[c])}")
