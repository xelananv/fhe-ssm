#!/usr/bin/env python3
"""Stage FineWeb-Edu tokens to disk as uint16 + provenance (S3 T2 / S2.75 A-0).

Industrializes the RAM-safe v2 pattern of kaggle_datamix_pilot.py:210-262:
sorted shard listing, ParquetFile.iter_batches(256), per-doc-batch tokenize,
EOS id between docs, first-doc sha16 fingerprint. Deltas for the full run:
  - uint16 on disk (vocab 50,277 < 65,536; the pilot used int32->int64 in
    RAM — uint16 halves the 26 GB staging footprint), appended incrementally
    so liveness (file_growth) sees progress and a killed run resumes by
    truncating to the last whole token.
  - reserves a HOLDOUT slice (--holdout-tokens, default 2M) AFTER the
    training budget, written as holdout.npy — the T4 instruments'
    non-annealed FineWeb axis.
  - provenance JSON: repo/config/shards consumed/column/first-doc sha16/
    eos_id/counts (the pilot's pattern, extended).

  python ml-eval/stage_fineweb_tokens.py --out DIR --tokens N \
      --provenance FILE [--holdout-tokens 2000000] [--repo HuggingFaceTB/smollm-corpus] \
      [--config fineweb-edu-dedup]

NOTE: the reader for this output is `staged_token_reader.py` (AUTHORED+
VERIFIED 2026-08-18) — `install(base)` then `--data staged:<dir>`. Whether
Phase B trains from streaming (--data fineweb) or from this file is still
the phase gate's decision; Phase A arms deliberately all stream so they
stay data-matched to each other.

**Pass `--provenance <out>/provenance.json`.** The reader treats provenance
as mandatory and hard-errors without it: `train_tokens` is the holdout
barrier, and it cannot be recovered from the directory alone because the
loop below stops on a whole document, leaving a ragged tail past the
holdout (`total_staged >= need`). A reader that walked this file to EOF
would train on the holdout it reserved.
"""
import argparse
import hashlib
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--holdout-tokens", type=int, default=2_000_000)
    ap.add_argument("--repo", default="HuggingFaceTB/smollm-corpus")
    ap.add_argument("--config", default="fineweb-edu-dedup")
    args = ap.parse_args()

    import numpy as np
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")
    vocab = 50277
    assert vocab < 65536, "uint16 staging requires vocab < 65536"
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    os.makedirs(args.out, exist_ok=True)
    bin_path = os.path.join(args.out, "tokens.uint16.bin")
    need = args.tokens + args.holdout_tokens

    # resume: truncate to whole tokens, continue from the recorded shard set
    done = 0
    if os.path.exists(bin_path):
        size = os.path.getsize(bin_path) - (os.path.getsize(bin_path) % 2)
        with open(bin_path, "r+b") as f:
            f.truncate(size)
        done = size // 2
        print(json.dumps({"event": "resume", "tokens_on_disk": done}), flush=True)
    if done >= need:
        print(json.dumps({"event": "already_complete", "tokens": done}), flush=True)
        _finalize(args, np, bin_path, eos_id, [], None, None, done)
        return 0

    files = sorted(f for f in list_repo_files(args.repo, repo_type="dataset")
                   if f.startswith(args.config + "/") and f.endswith(".parquet"))
    used, col, fp = [], None, None
    total = done
    last_print = total
    out = open(bin_path, "ab")
    skip = done   # deterministic re-walk: skip already-staged tokens
    for fn in files:
        p = hf_hub_download(args.repo, repo_type="dataset", filename=fn)
        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names
        col = "text" if "text" in names else next(
            n for n in names if "string" in str(pf.schema_arrow.field(n).type))
        used.append(fn)
        for rb in pf.iter_batches(batch_size=256, columns=[col]):
            docs = [d for d in rb.column(0).to_pylist() if d]
            del rb
            if not docs:
                continue
            if fp is None:
                fp = hashlib.sha256(docs[0][:1024].encode()).hexdigest()[:16]
            for ids in tok(docs, return_tensors=None)["input_ids"]:
                arr = np.asarray(ids + [eos_id], dtype=np.uint16)
                n = len(arr)
                if skip >= n:            # already on disk from a prior run
                    skip -= n
                    continue
                if skip:
                    arr = arr[skip:]
                    n = len(arr)
                    skip = 0
                out.write(arr.tobytes())
                total += n
                if total >= need:
                    break
            if total - last_print >= 2_000_000:
                print(json.dumps({"event": "progress", "tokens": total,
                                  "of": need, "shard": fn}), flush=True)
                last_print = total
            if total >= need:
                break
        out.flush()
        print(json.dumps({"event": "shard_done", "shard": fn, "tokens": total}), flush=True)
        if total >= need:
            break
    out.close()
    assert total >= need, f"only {total:,} of {need:,} tokens available"
    _finalize(args, np, bin_path, eos_id, used, col, fp, total)
    return 0


def _finalize(args, np, bin_path, eos_id, used, col, fp, total):
    # holdout = the slice AFTER the training budget
    hold_path = os.path.join(args.out, "holdout.npy")
    with open(bin_path, "rb") as f:
        f.seek(args.tokens * 2)
        hold = np.frombuffer(f.read(args.holdout_tokens * 2), dtype=np.uint16)
    np.save(hold_path, hold)
    prov = {"repo": args.repo, "cfg": args.config, "files": used, "col": col,
            "first_doc_sha16": fp, "eos_id": int(eos_id),
            "train_tokens": args.tokens, "holdout_tokens": int(len(hold)),
            "total_staged": int(total), "dtype": "uint16",
            "bin_bytes": os.path.getsize(bin_path)}
    os.makedirs(os.path.dirname(os.path.abspath(args.provenance)), exist_ok=True)
    with open(args.provenance, "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps({"event": "done", "tokens": int(total),
                      "holdout": int(len(hold)), "provenance": args.provenance}),
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
