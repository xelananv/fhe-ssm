# SPDX-License-Identifier: MIT
"""Run a text file through the model in `record` mode; write per-site input ranges (min, max, percentiles) as JSON.

Percentiles come from a uniform subsample of each site's inputs (up to --sample-per-call values per forward call,
thinned to about --max-samples per site); min and max are exact over every input.

  python calibrate.py --model M.pth --text wiki.valid.raw --tokens 200000 --ctx 1024 --out ranges.json
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from common import batches, sha256_file, tokenize_file, windows
from rwkv7_model import Sites, load_rwkv7, pick_device
from rwkv_tokenizer import RWKVTokenizer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, default=200_000, help="max tokens of the text to run")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--pcts", default="0.1,99.9")
    ap.add_argument("--sample-per-call", type=int, default=65536)
    ap.add_argument("--max-samples", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    dev = pick_device(a.device)
    tok = RWKVTokenizer(a.vocab) if a.vocab else RWKVTokenizer()
    sites = Sites("record", sample_per_call=a.sample_per_call, max_samples=a.max_samples, seed=a.seed)
    model = load_rwkv7(a.model, device=dev, sites=sites)
    ids = tokenize_file(a.text, tok, a.tokens)
    wins = windows(ids, a.ctx)
    # Padded positions would be recorded as site inputs: keep full-length windows only (the partial tail is dropped
    # unless it is the only window), so no batch is ever padded.
    full = [w for w in wins if len(w[0]) == a.ctx]
    wins = full or wins[:1]
    t0 = time.time()
    with torch.no_grad():
        for x, _ in batches(wins, a.batch):
            model(x.to(dev))
    pcts = tuple(float(p) for p in a.pcts.split(","))
    out = {"meta": {"model": a.model, "weights_sha256": sha256_file(a.model), "text": a.text,
                    "tokens": int(len(ids)), "ctx": a.ctx, "windows": len(wins), "pcts": list(pcts),
                    "seconds": round(time.time() - t0, 1), "device": str(dev), "site_order": sites.seen},
           "sites": sites.ranges(pcts)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}: {len(out['sites'])} sites, {len(ids)} tokens, {out['meta']['seconds']} s")


if __name__ == "__main__":
    main()
