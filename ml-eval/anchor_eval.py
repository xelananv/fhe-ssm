#!/usr/bin/env python3
"""anchor_eval.py — S3.1: the externally-legible evaluation, run IDENTICALLY on
the public anchors and (post-training) on our models.

WHY THIS EXISTS (S3.0_DECISION_PACKAGE.md §3). `native_test_ppl` is 2,048
contiguous WT103 tokens, one context — not a published-comparable number. The
quality plan pre-registers bands against Pythia-410M/160M `step6000` (the
12.58 B-token checkpoints, +3.6% vs our 12.14 B) and Mamba-370M, all sharing
the GPT-NeoX tokenizer, and requires the anchor evaluations to run and be
COMMITTED BEFORE training starts, so the bands are set on numbers not guesses.
NEW SCRIPT by design ("not an edit of run_eval.py").

The three axes, identical for every model:
  1. wt103   — WikiText-103 FULL test split, token-level ppl, NeoX tokenizer,
               ctx 1024, fixed non-overlapping stride 1024, token count
               reported; word-level ppl derived alongside (exp(nll_total /
               n_words), n_words = whitespace words of the raw split) for the
               literature.
  2. lambada — LAMBADA-OpenAI, FULL 5,153 items (not the 300-row instrument
               slice): last-word accuracy (teacher-forced greedy argmax over
               every token of the target word — the EleutherAI-harness
               definition) + last-word ppl.
  3. holdout — the staged FineWeb-Edu holdout (uint16 NeoX ids), token-level
               ppl, ctx 1024, disjoint windows.

Harness validation (run once): Pythia-410M FINAL must land near the published
LAMBADA 51.4% / 10.84 (Mamba paper table 3) within noise; that pins the
LAMBADA implementation before any band is set.

Backends:
  --backend hf     : any HF causal LM sharing the NeoX tokenizer
                     (EleutherAI/pythia-410m [--revision step6000],
                      EleutherAI/pythia-160m, state-spaces/mamba-370m-hf).
  --backend native : our checkpoint via the trainer module — wired
                     post-training (the pre-training gate needs only anchors).

Records: one JSON line per (model, axis) to stdout AND --out JSONL. R5: every
number in a summary must be transcribed from those lines.

Determinism: float32 weights (405 M fits any card here), no sampling anywhere,
fixed window boundaries, dataset order as shipped. Model+revision+sha of this
file stamped into every line.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

CTX = 1024


def log(obj):
    line = json.dumps(obj)
    print(line, flush=True)
    if ARGS.out:
        with open(ARGS.out, "a") as f:
            f.write(line + "\n")


def self_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


# --------------------------------------------------------------- backends
class HFModel:
    def __init__(self, name, revision):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.revision = revision
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=torch.float32)
        self.model.eval().to(ARGS.device)

    @torch.no_grad()
    def token_nll(self, ids):
        """ids: 1-D LongTensor on device, len >= 2. Returns (sum_nll_nats,
        n_scored) scoring positions 1..len-1 (teacher-forced)."""
        x = ids.unsqueeze(0)
        out = self.model(x).logits.float()          # [1, T, V]
        lsm = torch.log_softmax(out[0, :-1], dim=-1)
        tgt = ids[1:]
        nll = -lsm.gather(1, tgt.unsqueeze(1)).squeeze(1)
        return float(nll.sum().item()), int(tgt.numel())

    @torch.no_grad()
    def greedy_ok_and_nll(self, ctx_ids, tgt_ids):
        """Teacher-forced: feed ctx+tgt, check argmax at each tgt position,
        accumulate tgt nll. Returns (all_argmax_match, sum_nll, n_tgt)."""
        ids = torch.cat([ctx_ids, tgt_ids])
        x = ids.unsqueeze(0)
        out = self.model(x).logits.float()[0]
        # position predicting tgt_ids[j] is len(ctx)+j-1
        start = ctx_ids.numel() - 1
        sl = out[start:start + tgt_ids.numel()]
        lsm = torch.log_softmax(sl, dim=-1)
        nll = -lsm.gather(1, tgt_ids.unsqueeze(1)).squeeze(1)
        ok = bool((sl.argmax(dim=-1) == tgt_ids).all().item())
        return ok, float(nll.sum().item()), int(tgt_ids.numel())


class NativeModel:
    """Our checkpoint, scored on the identical axes. Loads the trainer's own
    CFG json + state_dict npz (exactly what _save() writes), builds the model
    with the trainer's build_model, and tokenizes text with the same NeoX
    tokenizer the corpus was staged with (vocab 50,277 — shared with Pythia,
    which is the entire point of the anchor design). Works for BOTH arms:
    masked training zeroes weights in place, so the banded arm's saved
    weights run bit-identically under the stock model class (the same
    property that lets banded bundles run in the plain FHE harness)."""

    def __init__(self, tag, art_dir):
        import torch as T
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import train_fhe_native_ssm as tr
        cfg = json.loads(open(os.path.join(art_dir, f"{tag}_config.json")).read())
        tr.CFG.update(cfg)
        tr.CFG["dropout"] = 0.0
        self.name = f"native:{tag}"
        self.revision = None
        model = tr.build_model(T)
        w = np.load(os.path.join(art_dir, f"{tag}_weights.npz"))
        sd = {k: T.from_numpy(np.asarray(w[k])) for k in w.files}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            log({"anchorEval": "nativeLoadWarn", "tag": tag,
                 "missing": list(missing), "unexpected": list(unexpected)})
            if missing:
                raise SystemExit("native backend: missing weights — refuse to score")
        self.model = model.float().eval().to(ARGS.device)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        self.T = T

    @torch.no_grad()
    def _logits(self, ids):
        out = self.model(ids.unsqueeze(0))
        if not torch.is_tensor(out):                 # (logits, ...) tuples
            out = out[0]
        return out.float()[0] if out.dim() == 3 else out.float()

    @torch.no_grad()
    def token_nll(self, ids):
        lsm = torch.log_softmax(self._logits(ids)[:-1], dim=-1)
        tgt = ids[1:]
        nll = -lsm.gather(1, tgt.unsqueeze(1)).squeeze(1)
        return float(nll.sum().item()), int(tgt.numel())

    @torch.no_grad()
    def greedy_ok_and_nll(self, ctx_ids, tgt_ids):
        ids = torch.cat([ctx_ids, tgt_ids])
        out = self._logits(ids)
        start = ctx_ids.numel() - 1
        sl = out[start:start + tgt_ids.numel()]
        lsm = torch.log_softmax(sl, dim=-1)
        nll = -lsm.gather(1, tgt_ids.unsqueeze(1)).squeeze(1)
        ok = bool((sl.argmax(dim=-1) == tgt_ids).all().item())
        return ok, float(nll.sum().item()), int(tgt_ids.numel())


def make_model():
    if ARGS.backend == "hf":
        return HFModel(ARGS.model, ARGS.revision)
    return NativeModel(ARGS.model, ARGS.art_dir)   # --model is the TAG here


# --------------------------------------------------------------- datasets
def wt103_text():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    return "".join(r["text"] for r in ds)


def lambada_rows():
    # same source as campaign/scripts/stage_lambada.py, but the FULL set
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("EleutherAI/lambada_openai", repo_type="dataset",
                        filename="data/lambada_test_en.jsonl")
    return [json.loads(l)["text"] for l in open(p) if l.strip()]


def holdout_ids(path):
    return np.fromfile(path, dtype=np.uint16).astype(np.int64)


# --------------------------------------------------------------- axes
def eval_token_ppl(m, ids_np, axis, meta):
    dev = ARGS.device
    total_nll, total_scored = 0.0, 0
    n_win = 0
    t0 = time.time()
    for s in range(0, len(ids_np) - 1, CTX):
        w = ids_np[s:s + CTX + 1]                  # +1: last token is target-only
        if len(w) < 2:
            break
        ids = torch.from_numpy(np.asarray(w, dtype=np.int64)).to(dev)
        nll, n = m.token_nll(ids)
        total_nll += nll
        total_scored += n
        n_win += 1
        if ARGS.max_windows and n_win >= ARGS.max_windows:
            meta = dict(meta, truncated=True)
            break
    rec = {"anchorEval": axis, "model": m.name, "revision": m.revision,
           "ctx": CTX, "stride": CTX, "windows": n_win,
           "tokensScored": total_scored,
           "tokenPpl": math.exp(total_nll / max(1, total_scored)),
           "nllTotalNats": total_nll,
           "wallSec": round(time.time() - t0, 1), "script": self_sha()}
    rec.update(meta)
    log(rec)


def eval_wt103(m):
    text = wt103_text()
    n_words = len(text.split())
    enc = m.tok(text, return_tensors="np")["input_ids"][0]
    total_nll, total_scored, n_win = 0.0, 0, 0
    t0 = time.time()
    for s in range(0, len(enc) - 1, CTX):
        w = enc[s:s + CTX + 1]
        if len(w) < 2:
            break
        ids = torch.from_numpy(np.asarray(w, dtype=np.int64)).to(ARGS.device)
        nll, n = m.token_nll(ids)
        total_nll += nll
        total_scored += n
        n_win += 1
        if ARGS.max_windows and n_win >= ARGS.max_windows:
            break
    log({"anchorEval": "wt103", "model": m.name, "revision": m.revision,
         "ctx": CTX, "stride": CTX, "windows": n_win,
         "tokensScored": total_scored, "tokensTotal": int(len(enc)),
         "wordsTotal": n_words,
         "tokenPpl": math.exp(total_nll / max(1, total_scored)),
         # word ppl derived over the words the SCORED tokens cover; at full
         # split scored ~= total so use the standard derivation and say so
         "wordPplDerived": math.exp(total_nll / max(1, n_words)),
         "wordPplBasisNote": "exp(nll_total/n_words), n_words = whitespace "
                             "words of the raw test split; valid at full "
                             "split only (truncated runs: token ppl only)",
         "truncated": bool(ARGS.max_windows and n_win >= ARGS.max_windows),
         "nllTotalNats": total_nll,
         "wallSec": round(time.time() - t0, 1), "script": self_sha()})


def eval_lambada(m):
    rows = lambada_rows()
    n_ok, total_nll, total_tgt = 0, 0.0, 0
    t0 = time.time()
    n = 0
    for text in rows:
        # target = last whitespace word; context = everything before it
        idx = text.rstrip().rfind(" ")
        ctx_text, tgt_text = text[:idx], text[idx:]      # tgt keeps the space
        ctx_ids = m.tok(ctx_text, return_tensors="np")["input_ids"][0]
        tgt_ids = m.tok(tgt_text, return_tensors="np")["input_ids"][0]
        if len(ctx_ids) == 0 or len(tgt_ids) == 0:
            continue
        ctx_t = torch.from_numpy(np.asarray(ctx_ids[-(CTX - len(tgt_ids)):],
                                            dtype=np.int64)).to(ARGS.device)
        tgt_t = torch.from_numpy(np.asarray(tgt_ids, dtype=np.int64)).to(ARGS.device)
        ok, nll, ntgt = m.greedy_ok_and_nll(ctx_t, tgt_t)
        n_ok += int(ok)
        total_nll += nll
        total_tgt += ntgt
        n += 1
        if ARGS.max_items and n >= ARGS.max_items:
            break
    log({"anchorEval": "lambada", "model": m.name, "revision": m.revision,
         "items": n, "acc": n_ok / max(1, n),
         "lastWordPpl": math.exp(total_nll / max(1, total_tgt)),
         # the PUBLISHED-comparable number: perplexity per WORD (nll summed
         # over the word tokens, mean over examples) -- validated 2026-08-28:
         # Pythia-410M final 10.78 vs published 10.84 (acc 51.6 vs 51.4)
         "lambadaPplWord": math.exp(total_nll / max(1, n)),
         "tgtTokens": total_tgt,
         "defNote": "acc = teacher-forced greedy argmax matches EVERY token "
                    "of the last whitespace word (leading space attached); "
                    "ppl = exp(mean nll over last-word tokens)",
         "published": ({"lambadaPpl": 10.84, "lambadaAcc": 51.4}
                       if "410m" in m.name and m.revision in (None, "main") else None),
         "wallSec": round(time.time() - t0, 1), "script": self_sha()})


def eval_holdout(m):
    ids = holdout_ids(ARGS.holdout)
    eval_token_ppl(m, ids, "holdout",
                   {"holdoutPath": ARGS.holdout, "holdoutTokens": int(len(ids))})


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "native"], default="hf")
    ap.add_argument("--model", required=True,
                    help="HF id, e.g. EleutherAI/pythia-410m")
    ap.add_argument("--revision", default=None,
                    help="HF revision, e.g. step6000 (the 12.58B-token ckpt)")
    ap.add_argument("--axes", default="wt103,lambada,holdout")
    ap.add_argument("--holdout", default="ml-eval/artifacts/staged/holdout.bin")
    ap.add_argument("--art-dir", default="ml-eval/artifacts",
                    help="native backend: artifacts dir holding {tag}_config.json + {tag}_weights.npz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="debug: cap ppl windows (marks record truncated)")
    ap.add_argument("--max-items", type=int, default=0,
                    help="debug: cap lambada items")
    ARGS = ap.parse_args()
    torch.set_grad_enabled(False)
    m = make_model()
    log({"anchorEval": "config", "model": ARGS.model, "revision": ARGS.revision,
         "backend": ARGS.backend, "device": ARGS.device,
         "torch": torch.__version__, "script": self_sha()})
    for axis in ARGS.axes.split(","):
        {"wt103": eval_wt103, "lambada": eval_lambada,
         "holdout": eval_holdout}[axis.strip()](m)
    log({"anchorEvalDone": True, "model": ARGS.model, "revision": ARGS.revision})


if __name__ == "__main__":
    main()
