# SPDX-License-Identifier: MIT
"""Evaluate quality (and fidelity, R4) of the exact and polynomial models. One JSONL line per configuration.

Quality:
  - token perplexity on a raw text file (WikiText-103 test): non-overlapping windows of --ctx targets, each window
    starting from a zero state with input [0] + chunk[:-1], so every token is scored exactly once;
  - LAMBADA (OpenAI jsonl, field 'text'): context = text up to the last space, target = ' ' + last word,
    input [0] + enc(context) (demo.py:410-436); correct iff greedy argmax matches every target token. lambada_ppl is the
    reference's definition: exp(-mean over docs of the summed target log-prob).
Fidelity (poly configs, unless --no-fidelity): the same model in exact mode on the same windows, then
  top1_agree (argmax equality per scored token), kl_mean = mean KL(exact || poly) per token, and the exact ppl.

Configurations:
  --mode exact                    one line
  --mode poly --coeffs C.json     one line (every polynomial site in C.json)
  --ablate --coeffs C.json        one line per group, only that group polynomial, the rest exact
                                  (--ablate-by kind: a group = one site kind across all layers, e.g. 'ffn.relu2',
                                   'ln1.rsqrt'; --ablate-by site: a group = one site)

  python eval.py --model M.pth --wikitext wiki.test.raw --lambada lambada_test.jsonl --mode exact --out ev.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from common import BOS, append_jsonl, batches, sha256_file, tokenize_file, windows
from rwkv7_model import Sites, load_rwkv7, pick_device, site_short
from rwkv_tokenizer import RWKVTokenizer


def _set(sites: Sites, mode: str, only):
    sites.mode = mode
    sites.only = only


@torch.no_grad()
def eval_text(model, wins, batch, dev, only, fidelity: bool):
    s = model.sites
    nll = 0.0
    n = 0
    nll_ex = 0.0
    agree = 0
    kl = 0.0
    for x, y in batches(wins, batch):
        x, y = x.to(dev), y.to(dev)
        m = y != -100
        if fidelity:
            _set(s, "exact", None)
            lp_ex = F.log_softmax(model(x).float(), -1)
        _set(s, "poly" if only is not None else "exact", only)
        lp = F.log_softmax(model(x).float(), -1)
        yt = y.clamp(min=0)
        nll -= lp.gather(-1, yt[..., None])[..., 0][m].sum().item()
        n += int(m.sum())
        if fidelity:
            nll_ex -= lp_ex.gather(-1, yt[..., None])[..., 0][m].sum().item()
            agree += int((lp.argmax(-1) == lp_ex.argmax(-1))[m].sum())
            kl += (lp_ex.exp() * (lp_ex - lp)).sum(-1)[m].sum().item()
            del lp_ex
    out = {"wt_tokens": n, "wt_nll": nll / n, "wt_ppl": math.exp(nll / n)}
    if fidelity:
        out.update({"wt_ppl_exact_same_run": math.exp(nll_ex / n), "top1_agree": agree / n, "kl_mean": kl / n})
    return out


def load_lambada(path, tok, limit=None):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)["text"]
            ctx, last = t.rsplit(" ", 1)
            docs.append(([BOS] + tok.encode(ctx), tok.encode(" " + last)))
            if limit and len(docs) >= limit:
                break
    return docs


@torch.no_grad()
def eval_lambada(model, docs, batch, dev, only):
    _set(model.sites, "poly" if only is not None else "exact", only)
    order = sorted(range(len(docs)), key=lambda i: len(docs[i][0]) + len(docs[i][1]))
    correct = 0
    lsum = 0.0
    for b in range(0, len(order), batch):
        grp = [docs[i] for i in order[b: b + batch]]
        L = max(len(s) + len(d) for s, d in grp)
        x = torch.zeros(len(grp), L, dtype=torch.long)
        for j, (s, d) in enumerate(grp):
            x[j, : len(s) + len(d)] = torch.tensor(s + d)
        logits = model(x.to(dev))
        for j, (s, d) in enumerate(grp):
            pos = torch.arange(len(s) - 1, len(s) - 1 + len(d), device=logits.device)
            tgt = torch.tensor(d, device=logits.device)
            row = F.log_softmax(logits[j, pos].float(), -1)   # only the target positions
            lsum += row.gather(-1, tgt[:, None]).sum().item()
            correct += int(bool((row.argmax(-1) == tgt).all()))
    return {"lambada_n": len(docs), "lambada_acc": correct / len(docs), "lambada_ppl": math.exp(-lsum / len(docs))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wikitext", default=None)
    ap.add_argument("--lambada", default=None)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lambada-batch", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-lambada", type=int, default=None)
    ap.add_argument("--mode", choices=["exact", "poly"], default="exact")
    ap.add_argument("--coeffs", default=None)
    ap.add_argument("--clamp", action="store_true", help="clamp site inputs to [lo, hi] before the polynomial")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--ablate-by", choices=["kind", "site"], default="kind")
    ap.add_argument("--also-full", action="store_true", help="with --ablate, also run exact and all-poly lines")
    ap.add_argument("--no-fidelity", action="store_true")
    ap.add_argument("--weights", default=None, help="optional healed state_dict (.pt) loaded over --model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if (a.mode == "poly" or a.ablate) and not a.coeffs:
        ap.error("--coeffs is required for poly / ablate")

    dev = pick_device(a.device)
    tok = RWKVTokenizer(a.vocab) if a.vocab else RWKVTokenizer()
    sites = Sites("exact")
    model = load_rwkv7(a.model, device=dev, sites=sites)
    wsha = sha256_file(a.model)
    if a.weights:
        ck = torch.load(a.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        model.to(dev)
        wsha = sha256_file(a.weights)
    cfg = None
    if a.coeffs:
        cfg = json.load(open(a.coeffs))
        sites.load_poly(cfg, clamp=a.clamp)
    poly_names = list(sites.polys)

    configs = OrderedDict()
    if a.ablate:
        if a.also_full:
            configs["exact"] = None
            configs["poly:all"] = set(poly_names)
        groups = OrderedDict()
        for nm in poly_names:
            groups.setdefault(site_short(nm) if a.ablate_by == "kind" else nm, set()).add(nm)
        for g, names in groups.items():
            configs[f"ablate:{g}"] = names
    elif a.mode == "exact":
        configs["exact"] = None
    else:
        configs["poly:all"] = set(poly_names)

    wins = windows(tokenize_file(a.wikitext, tok, a.max_tokens), a.ctx) if a.wikitext else None
    docs = load_lambada(a.lambada, tok, a.max_lambada) if a.lambada else None

    for label, only in configs.items():
        t0 = time.time()
        sites.reset_counts()
        rec = {"tag": a.tag, "config": label, "model": a.model, "weights_sha256": wsha, "weights": a.weights,
               "coeffs": a.coeffs if only is not None else None, "clamp": a.clamp if only is not None else None,
               "n_poly_sites": 0 if only is None else len(only), "ctx": a.ctx, "device": str(dev)}
        if wins:
            rec.update(eval_text(model, wins, a.batch, dev, only, fidelity=(only is not None and not a.no_fidelity)))
        if docs:
            rec.update(eval_lambada(model, docs, a.lambada_batch, dev, only))
        if only is not None:
            fr = sites.outside_fractions()
            rec["outside_frac_max"] = max(fr.values()) if fr else 0.0
            rec["outside_frac_top"] = dict(sorted(fr.items(), key=lambda kv: -kv[1])[:8])
        rec["seconds"] = round(time.time() - t0, 1)
        append_jsonl(a.out, rec)
        print(json.dumps({k: rec[k] for k in rec if k in
                          ("config", "wt_ppl", "top1_agree", "kl_mean", "lambada_acc", "lambada_ppl",
                           "outside_frac_max", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
