#!/usr/bin/env python3
"""
run_eval.py — teacher-forced WikiText-103 evaluation of RWKV-4-169m with the
wkv recurrence computed (a) in float64 [baseline], (b) under REAL OpenFHE
CKKS with bootstrapping [fhe], and optionally (c) with calibrated Gaussian
noise at measured FHE floors [noise].

Saves per-mode logits + stats to ml-eval/artifacts/run_<mode>.npz; then run
accuracy_delta.py to produce the report.

Usage:
  .venv/bin/python ml-eval/run_eval.py --tokens 512 --modes float,fhe
  .venv/bin/python ml-eval/run_eval.py --modes noise --noise-sigma 4.4e-4
"""
import argparse
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifacts"
MODEL_ID = "RWKV/rwkv-4-169m-pile"


def wikitext_tokens(n_tokens: int):
    """First contiguous n_tokens+1 tokens of the WikiText-103 test split.
    (The wikitext-103-raw test split is the canonical small eval slice.)"""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    path = hf_hub_download(
        repo_id="Salesforce/wikitext", repo_type="dataset",
        filename="wikitext-103-raw-v1/test-00000-of-00001.parquet")
    text = "".join(pq.read_table(path, columns=["text"]).column("text").to_pylist())
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(text[:200000], return_tensors=None)["input_ids"][: n_tokens + 1]
    assert len(ids) == n_tokens + 1, "slice too short"
    return ids


def nll_stats(logits: np.ndarray, targets: np.ndarray):
    """Per-position negative log-likelihood, float64, via logsumexp."""
    m = logits.max(-1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.exp(logits - m).sum(-1))
    tgt_logit = np.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
    return lse - tgt_logit  # (T,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--modes", default="float,fhe")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--config", default="demo")
    ap.add_argument("--noise-sigma", type=float, default=4.4e-4,
                    help="calibrated sigma for mode 'noise' (default: measured "
                         "128-bit lean-budget bootstrap floor)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE))
    from rwkv_numpy import RwkvRunner

    ids = wikitext_tokens(args.tokens)
    inputs, targets = ids[:-1], np.array(ids[1:])
    runner = RwkvRunner(str(ART / "weights.npz"), str(ART / "config.json"))

    for mode in args.modes.split(","):
        mode = mode.strip()
        t0 = time.time()
        logits, stats = runner.forward(
            inputs, backend=mode,
            fhe_args={"k": args.k, "margin": args.margin, "config": args.config},
            noise_sigma=args.noise_sigma)
        wall = time.time() - t0
        nll = nll_stats(logits.astype(np.float64), targets)
        ppl = float(np.exp(nll.mean()))
        stats.update({"mode": mode, "tokens": len(inputs), "wallSec": round(wall, 1),
                      "ppl": ppl, "k": args.k, "config": args.config,
                      "noise_sigma": args.noise_sigma if mode == "noise" else None})
        suffix = f"_{args.noise_sigma:.0e}" if mode == "noise" else ""
        np.savez_compressed(
            ART / f"run_{mode}{suffix}.npz",
            logits=logits.astype(np.float32), nll=nll, targets=targets,
            stats=json.dumps(stats))
        print(json.dumps({k: v for k, v in stats.items() if k != "layers"}))
        for l in stats.get("layers", []):
            print("  layer:", json.dumps(l))


if __name__ == "__main__":
    main()
