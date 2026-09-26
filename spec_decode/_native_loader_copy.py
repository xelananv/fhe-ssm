#!/usr/bin/env python3
"""
_native_loader_copy.py — load_native(tag, art_dir, device): the trained model's weights (<tag>_weights.npz + <tag>_config.json)
into the trainer's module, for the plaintext references (plain_recurrent.py, plain_greedy.py, prompt selection).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_native(tag, art_dir, device):
    import torch as T
    import train_fhe_native_ssm as tr
    cfg = json.loads(open(os.path.join(art_dir, f"{tag}_config.json")).read())
    # CFG is module-global; snapshot+restore so two models with different
    # shapes (dense vs band16 share shape, but stay safe) build correctly.
    saved = dict(tr.CFG)
    tr.CFG.update(cfg)
    tr.CFG["dropout"] = 0.0
    model = tr.build_model(T)
    w = np.load(os.path.join(art_dir, f"{tag}_weights.npz"))
    sd = {k: T.from_numpy(np.asarray(w[k])) for k in w.files}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"{tag}: missing weights {sorted(missing)[:5]} — refuse")
    tr.CFG.clear(); tr.CFG.update(saved)
    return model.float().eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--art-dir", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from run_eval import wikitext_tokens
    torch.set_grad_enabled(False)

    ids = np.asarray(wikitext_tokens(a.tokens + 1), dtype=np.int64)
    drafter = load_native(a.drafter, a.art_dir, a.device)
    verifier = load_native(a.verifier, a.art_dir, a.device)

    def greedy(model, window):
        x = torch.from_numpy(window).to(a.device).unsqueeze(0)
        out = model(x)
        if not torch.is_tensor(out):
            out = out[0]
        lg = out.float()[0] if out.dim() == 3 else out.float()
        return lg.argmax(dim=-1).cpu().numpy()          # greedy for each position

    # Teacher-forced windows with stride=ctx: score every position once.
    matches, total = 0, 0
    match_by_chunk = []
    for lo in range(0, len(ids) - 1, a.ctx):
        window = ids[lo:lo + a.ctx]
        if len(window) < 32:
            break
        gd = greedy(drafter, window)
        gv = greedy(verifier, window)
        m = (gd == gv)
        matches += int(m.sum()); total += len(m)
        match_by_chunk.append(round(float(m.mean()), 4))

    alpha = matches / max(total, 1)
    # expected ACCEPTED drafted tokens per verify pass with k drafts
    exp_accept = sum(alpha ** i for i in range(1, a.k + 1))
    rec = {
        "alphaPair": True, "drafter": a.drafter, "verifier": a.verifier,
        "positions": total, "alpha": round(alpha, 4), "k": a.k,
        "expectedAcceptedPerPass": round(exp_accept, 3),
        "tokensPerTickIncludingVerifier": round(exp_accept + 1.0, 3),
        "matchByChunk": match_by_chunk,
        "note": ("teacher-forced greedy match on WikiText held-out; replaces "
                 "the L7-era alpha=0.665 which was a DIFFERENT model pair"),
    }
    line = json.dumps(rec)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
