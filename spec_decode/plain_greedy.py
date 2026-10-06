#!/usr/bin/env python3
"""Plaintext greedy continuations of candidate demo prompts on the demo model (the reference the FHE
lanes track under top-1 agreement). Mirrors the tick engine's retirement rule: WORDS = generated TOKENS
(fhe_client.apply_lane_predictions), stop on a newline token. Prints prompt token count (= the tick at
which the lane starts generating) and the continuation, one JSON line per prompt, written as it goes."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
from _native_loader_copy import load_native   # noqa: E402
import fhe_client as FC                        # noqa: E402
import numpy as np, torch                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="pbd430a"); ap.add_argument("--art-dir", default=os.path.join(os.path.dirname(HERE), "ml-eval", "artifacts"))
ap.add_argument("--tokens", type=int, default=8); ap.add_argument("--device", default="cpu")
a = ap.parse_args()
tok = FC.get_tokenizer()
stop_ids = {tok.encode(c)[0] for c in ("\n", "\n\n") if len(tok.encode(c)) == 1}
model = load_native(a.tag, a.art_dir, a.device)
typ = "?"; out = open(a.out, "a")
for line in open(a.prompts):
    line = line.rstrip("\n")
    if not line.strip(): continue
    if line.startswith("# type:"): typ = line.split(":", 1)[1].strip(); continue
    ids = tok.encode(line); gen = []; t0 = time.time()
    with torch.no_grad():
        for _ in range(a.tokens):
            o = model(torch.tensor(ids + gen, device=a.device).unsqueeze(0))
            if not torch.is_tensor(o): o = o[0]
            lg = o.float()[0] if o.dim() == 3 else o.float()
            nxt = int(torch.argmax(lg[-1]).item()); gen.append(nxt)
            if nxt in stop_ids: break
    rec = {"type": typ, "prompt": line, "promptTokens": len(ids), "genTokens": len(gen),
           "text": tok.decode(gen), "ticksToFinish": len(ids) + len(gen), "sec": round(time.time() - t0, 1)}
    out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
