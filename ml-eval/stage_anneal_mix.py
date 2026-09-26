#!/usr/bin/env python3
"""
stage_anneal_mix.py — build the pbd430a quality-anneal corpus (S3.1, 2026-08-31).

Mix (chunk-interleaved, 1M-token chunks, weighted round-robin):
  - WikiText-103 TRAIN split, all of it (~110M NeoX tokens) — the WT103 OOD gap
  - Gutenberg English fiction (sedthh/gutenberg_english) to --fiction-tokens
    (~150M) — the LAMBADA (narrative) gap
  - FineWeb-Edu replay from the existing staged corpus, first --replay-tokens
    (~150M) — forgetting guard

Output: <out>/tokens.uint16.bin + provenance.json (same format the trainer's
staged: reader consumes; vocab 50277 < 2^16 so uint16 is lossless).
Writes <out>/READY on success — the anneal cell asserts it before training.
"""
import argparse
import json
import os
import sys

import numpy as np

CHUNK = 1_000_000


def wt103_train_tokens(tok):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    parts = []
    for i in range(2):
        p = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                            filename=f"wikitext-103-raw-v1/train-0000{i}-of-00002.parquet")
        parts.append(pq.read_table(p).column("text").to_pylist())
    text = "\n".join(t for part in parts for t in part)
    ids = []
    B = 1 << 20
    for i in range(0, len(text), B):
        ids.extend(tok(text[i:i + B], add_special_tokens=False)["input_ids"])
    return np.asarray(ids, dtype=np.uint16)


def gutenberg_tokens(tok, want, skip=0):
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    files = sorted(f for f in list_repo_files("sedthh/gutenberg_english",
                                              repo_type="dataset")
                   if f.endswith(".parquet"))
    out = np.empty(want, dtype=np.uint16)
    n = 0
    skipped = 0
    for f in files:
        p = hf_hub_download(repo_id="sedthh/gutenberg_english",
                            repo_type="dataset", filename=f)
        t = pq.read_table(p)
        col = "TEXT" if "TEXT" in t.column_names else t.column_names[0]
        for row in t.column(col).to_pylist():
            if not row:
                continue
            ids = tok(row, add_special_tokens=False)["input_ids"]
            if skipped < skip:
                d = min(len(ids), skip - skipped)
                skipped += d
                ids = ids[d:]
                if not ids:
                    continue
            take = min(len(ids), want - n)
            out[n:n + take] = np.asarray(ids[:take], dtype=np.uint16)
            n += take
            if n >= want:
                return out
        print(json.dumps({"gutenberg_file_done": f, "tokens": n}), flush=True)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/anneal_staged")
    ap.add_argument("--staged-src", default="/root/src/ml-eval/artifacts/staged/tokens.uint16.bin")
    ap.add_argument("--fiction-tokens", type=int, default=150_000_000)
    ap.add_argument("--replay-tokens", type=int, default=150_000_000)
    ap.add_argument("--fiction-skip", type=int, default=0)
    ap.add_argument("--wt103-cap", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

    print(json.dumps({"stage": "wt103"}), flush=True)
    wt = wt103_train_tokens(tok)
    if a.wt103_cap:
        wt = wt[:a.wt103_cap]
    print(json.dumps({"wt103_tokens": int(len(wt))}), flush=True)
    print(json.dumps({"stage": "gutenberg"}), flush=True)
    gut = gutenberg_tokens(tok, a.fiction_tokens, skip=a.fiction_skip)
    print(json.dumps({"gutenberg_tokens": int(len(gut))}), flush=True)
    rep = np.fromfile(a.staged_src, dtype=np.uint16, count=a.replay_tokens)
    print(json.dumps({"replay_tokens": int(len(rep))}), flush=True)

    # weighted round-robin in 1M chunks so the stream mixes at anneal scale
    srcs = [("wt103", wt), ("gutenberg", gut), ("fineweb_replay", rep)]
    idx = {n: 0 for n, _ in srcs}
    order = []
    total = sum(len(x) for _, x in srcs)
    live = True
    with open(os.path.join(a.out, "tokens.uint16.bin"), "wb") as f:
        while live:
            live = False
            for name, arr in srcs:
                i = idx[name]
                if i >= len(arr):
                    continue
                arr[i:i + CHUNK].tofile(f)
                idx[name] = i + CHUNK
                order.append(name)
                live = True
    import os as _os
    size=_os.path.getsize(_os.path.join(a.out,"tokens.uint16.bin"))
    json.dump({"mix": {n: int(len(x)) for n, x in srcs}, "total": int(total),
               "dtype": "uint16", "train_tokens": size // 2, "holdout_tokens": 0,
               "eos_id": 0, "vocab": 50277, "bin_bytes": size,
               "chunk": CHUNK, "interleave": "round-robin-1M",
               "tokenizer": "EleutherAI/pythia-410m", "order_head": order[:12],
               "purpose": "pbd430a quality anneal (DEMO_DECISIONS + verdict 2026-08-31)"},
              open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
    open(os.path.join(a.out, "READY"), "w").write("ok\n")
    print(json.dumps({"staged": a.out, "total_tokens": int(total)}), flush=True)


if __name__ == "__main__":
    main()
