# SPDX-License-Identifier: MIT
"""Shared helpers: text -> token windows, file hashing, JSONL output."""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch

from rwkv_tokenizer import RWKVTokenizer

BOS = 0  # the reference prefixes LAMBADA contexts with token 0 (demo.py:419)


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tokenize_file(path: str, tok: RWKVTokenizer, max_tokens: int | None = None, cache: bool = True) -> np.ndarray:
    """Tokenize a whole raw text file. Cached as <path>.rwkvtok.npy beside the text (keyed on size+mtime)."""
    st = os.stat(path)
    cpath = f"{path}.rwkvtok.{st.st_size}.{int(st.st_mtime)}.npy"
    if cache and os.path.exists(cpath):
        ids = np.load(cpath)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = np.asarray(tok.encode(text), dtype=np.int64)
        if cache:
            try:
                np.save(cpath, ids)
            except OSError:
                pass
    return ids[:max_tokens] if max_tokens else ids


def windows(ids: np.ndarray, ctx: int):
    """Non-overlapping windows of `ctx` targets. Input = [BOS] + chunk[:-1], target = chunk: every token is scored
    exactly once, each window starts from a zero state. The last partial window is kept."""
    out = []
    for s in range(0, len(ids), ctx):
        chunk = ids[s: s + ctx]
        if len(chunk) < 2:
            continue
        inp = np.concatenate([[BOS], chunk[:-1]])
        out.append((inp, chunk))
    return out


def batches(wins, batch: int):
    """Right-pad to the batch's longest window (the model is causal, so padding never touches scored positions)."""
    for i in range(0, len(wins), batch):
        grp = wins[i: i + batch]
        L = max(len(w[0]) for w in grp)
        x = torch.zeros(len(grp), L, dtype=torch.long)
        y = torch.full((len(grp), L), -100, dtype=torch.long)
        for j, (inp, tgt) in enumerate(grp):
            x[j, : len(inp)] = torch.as_tensor(inp)
            y[j, : len(tgt)] = torch.as_tensor(tgt)
        yield x, y


def append_jsonl(path: str, rec: dict):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
