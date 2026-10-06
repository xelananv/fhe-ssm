#!/usr/bin/env python3
"""Compute ppl / top-1 / KL between the ENCRYPTED forward pass and the
float64 plaintext reference, from a --dump-logits file produced by
gpu_real_model plus the bundle's logits_ref.

Usage:
  compute_encrypted_metrics.py --bundle-dir DIR --tag TAG --dump FILE [--out FILE]

The dump holds float64 rows appended in decode order; FILE.tokens lists the
token index of each row (dead blocks leave gaps -- rows are matched to the
reference BY INDEX, never positionally). Targets are the same WT103 slice the
bundle was exported with (inputs = ids[:-1], targets = ids[1:]).
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def read_bundle_tensor(bdir, tag, name):
    idx = pathlib.Path(bdir) / f"bundle_{tag}.index.txt"
    blob = pathlib.Path(bdir) / f"bundle_{tag}.bin"
    for line in idx.read_text().splitlines():
        parts = line.split()
        if parts[0] == name:
            off, count = int(parts[1]), int(parts[2])
            shape = [int(x) for x in parts[3:]] or [count]
            a = np.fromfile(blob, dtype="<f8", count=count, offset=off)
            return a.reshape(shape)
    raise KeyError(name)


def nll_rows(logits, targets):
    m = logits.max(-1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.exp(logits - m).sum(-1))
    return lse - logits[np.arange(len(targets)), targets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = read_bundle_tensor(args.bundle_dir, args.tag, "logits_ref")  # (T, V)
    T, V = ref.shape

    toks = [int(x) for x in open(args.dump + ".tokens").read().split()]
    enc = np.fromfile(args.dump, dtype="<f8").reshape(len(toks), V)

    from run_eval import wikitext_tokens
    ids = wikitext_tokens(T + 1)
    targets = np.asarray(ids[1:])[:T]

    sel = np.asarray(toks)
    sel = sel[sel < T]
    enc = enc[: len(sel)]

    nll_ref = nll_rows(ref[sel], targets[sel])
    nll_enc = nll_rows(enc, targets[sel])

    # KL(ref || enc) per token, fp64
    def logsm(x):
        m = x.max(-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))

    lr, le = logsm(ref[sel]), logsm(enc)
    kl = (np.exp(lr) * (lr - le)).sum(-1)

    rep = {
        "metrics": "encrypted_vs_ref",
        "tag": args.tag,
        "dump": args.dump,
        "tokensDecoded": int(len(sel)),
        "tokensTotal": int(T),
        "ppl_ref": float(np.exp(nll_ref.mean())),
        "ppl_encrypted": float(np.exp(nll_enc.mean())),
        "ppl_delta_pct": float(100 * (np.exp(nll_enc.mean()) - np.exp(nll_ref.mean())) / np.exp(nll_ref.mean())),
        "top1_agree": float((ref[sel].argmax(-1) == enc.argmax(-1)).mean()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "finite": bool(np.isfinite(enc).all()),
    }
    line = json.dumps(rep)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
