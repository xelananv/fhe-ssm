#!/usr/bin/env python3
"""Does the served stateful session (A11, teacher-forced) really carry CONTEXT? Records-only check on the pod's
fidelity rows: for every decoded (tick, lane) compare the FHE argmax with
  (F) the plaintext model on the FULL prefix ids[:t+1]   (what a server that carries the state must reproduce), and
  (N) the plaintext model on the LAST TOKEN ALONE [ids[t]] (what a server with NO carried context would produce).
If the state were not carried, the FHE argmax would track (N). It tracks (F).
usage: a11_context_check.py --fidelity <fidelity.json> --prompts <prompts file> [--art-dir ~/Documents/fhe-ssm-backup/mac_art]"""
import argparse, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
from _native_loader_copy import load_native   # noqa: E402
import fhe_client as FC                        # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--fidelity", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--tag", default="pbd430a"); ap.add_argument("--art-dir", default=os.path.expanduser("~/Documents/fhe-ssm-backup/mac_art")); ap.add_argument("--device", default="cpu")
ap.add_argument("--max-tick", type=int, default=10**9)
a = ap.parse_args()
tok = FC.get_tokenizer(); model = load_native(a.tag, a.art_dir, a.device)
prompts = [l.rstrip("\n") for l in open(a.prompts) if l.strip()]; ids = [tok.encode(p) for p in prompts]
d = json.load(open(a.fidelity)); rows = d.get("rows", d) if isinstance(d, dict) else d
rows = [r for r in rows if not r.get("decFailed") and r["tick"] <= a.max_tick]
@torch.no_grad()
def argmax_of(seq):
    o = model(torch.tensor(seq, device=a.device).unsqueeze(0))
    if not torch.is_tensor(o): o = o[0]
    lg = o.float()[0] if o.dim() == 3 else o.float()
    return int(torch.argmax(lg[-1]))
cache = {}
def nocontext(tokid):
    if tokid not in cache: cache[tokid] = argmax_of([tokid])
    return cache[tokid]
per_tick = {}; nF = nN = nBoth = nDiffRef = nFheIsFullWhereDiffer = nFheIsNoneWhereDiffer = 0; refMismatch = 0
full_cache = {}
for r in rows:
    t, l = r["tick"], r["lane"]
    N = nocontext(ids[l][t])
    F = r["refArgmax"]                      # the pod's own full-prefix reference, recorded per row
    if t <= 3 and (l, t) not in full_cache:  # spot-check that the recorded reference IS the full-prefix argmax (first 4 ticks)
        full_cache[(l, t)] = argmax_of(ids[l][:t + 1]); refMismatch += (full_cache[(l, t)] != F)
    e = r["encArgmax"]; nF += (e == F); nN += (e == N)
    pt = per_tick.setdefault(t, [0, 0, 0, 0]); pt[0] += 1; pt[1] += (e == F); pt[2] += (e == N)
    if F != N:
        nDiffRef += 1; pt[3] += 1; nFheIsFullWhereDiffer += (e == F); nFheIsNoneWhereDiffer += (e == N)
n = len(rows)
print(f"rows (decoded lane-ticks): {n}; recorded reference re-derived as the full-prefix argmax on ticks 0-3: {len(full_cache) - refMismatch}/{len(full_cache)} equal")
print(f"FHE argmax == plaintext FULL-prefix argmax: {nF}/{n} ({nF / n * 100:.2f} %)")
print(f"FHE argmax == plaintext NO-context argmax (last token alone): {nN}/{n} ({nN / n * 100:.2f} %)")
print(f"lane-ticks where the two plaintext predictions DIFFER (context matters): {nDiffRef}; there the FHE output equals the full-prefix one in {nFheIsFullWhereDiffer} and the no-context one in {nFheIsNoneWhereDiffer}")
print("tick | context tokens | lane-ticks | FHE==full | FHE==no-context | full!=no-context")
for t in sorted(per_tick):
    c = per_tick[t]; print(f"{t:4d} | {t + 1:3d} | {c[0]:3d} | {c[1]:3d} | {c[2]:3d} | {c[3]:3d}")
