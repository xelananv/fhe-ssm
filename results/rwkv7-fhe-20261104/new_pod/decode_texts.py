#!/usr/bin/env python3
"""Decode the prompt and generated ids of the final-code records into text with the RWKV World tokenizer
(results/vector-d-20261104/harness/rwkv_tokenizer.py + raw/rwkv_vocab_v20230424.txt), beside the exact model's greedy
continuation (../old_pod_mirror/out_refs/refs_exact.jsonl) where one exists. Writes TEXTS_20261105.md."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vector-d-20261104/harness"))
from rwkv_tokenizer import RWKVTokenizer
T = RWKVTokenizer()
dec = lambda ids: T.decode(list(ids))
def load(f): return [json.loads(l) for l in open(f) if l.strip()]
refs = {(r["tag"], r["model"].split("/")[-1][:22]): r for r in load("../old_pod_mirror/out_refs/refs_exact.jsonl")}
out = ["# Decoded texts of the final-code runs (0.1B, pod 3, build 071cfa2c, client keys), 2026-11-05", "",
       "Every text is decoded from the record's `prompt_ids` / `generated_ids` with the RWKV World tokenizer; the exact",
       "model's greedy continuation (`refs_exact.jsonl`, float32, no polynomials) is quoted where it exists. Greedy decoding;",
       "a divergence after a near-tie is expected and is not an error (the fidelity measure is top-1 against the",
       "polynomial model, in RESULTS.md).", ""]
def show(title, prompt_ids, gen_ids, ref=None, note=""):
    out.append(f"## {title}"); out.append("")
    out.append(f"**Prompt:** {dec(prompt_ids)!r}"); out.append("")
    out.append(f"**FHE, greedy ({len(gen_ids)} tokens):** {dec(gen_ids)!r}"); out.append("")
    if ref is not None:
        same = next((i for i, (a, b) in enumerate(zip(gen_ids, ref["generated_ids"])) if a != b), None)
        out.append(f"**Exact model, greedy ({len(ref['generated_ids'])} tokens):** {ref['generated_text']!r}")
        out.append(f"  (identical for {len(gen_ids) if same is None else same} of {len(gen_ids)} tokens{' , then diverges' if same is not None else ''})"); out.append("")
    if note: out.append(note); out.append("")
for p in ["P1", "P2", "P3"]:
    s = load(f"f2_01b/run_best_{p}.jsonl")[-1]
    show(f"Decode {p}, current layout (`f2_01b/run_best_{p}.jsonl`, 9.74–9.76 s/token)", s["prompt_ids"], s["generated_ids"], refs.get((p, "RWKV-x070-World-0.1B-v")))
s = load("f2_01b/run_filled_best_P1.jsonl")[-1]
show("Decode P1, FILLED layout (`f2_01b/run_filled_best_P1.jsonl`; the error drifts, RESULTS.md)", s["prompt_ids"], s["generated_ids"], refs.get(("P1", "RWKV-x070-World-0.1B-v")))
for p in ["L1", "L5"]:
    s = load(f"f2_01b/run_pf32_best_{p}.jsonl")[-1]
    show(f"Prefill {p}, T = 32 (`f2_01b/run_pf32_best_{p}.jsonl`; the prompt read by prefill passes, then 32 decoded)", s["prompt_ids"], s["generated_ids"])
for tag, f in [("current layout, `b2` (27.99 s/pass)", "f2_01b/batch_k32_best_pob2.jsonl"), ("FILLED layout, `b2` (19.36 s/pass)", "f2_01b/batch_k32_filled_best_pob2.jsonl")]:
    s = load(f)[-1]
    for c in [0, 1, 2, 5, 12, 20]:
        ref = refs.get((f"P{c+1}", "RWKV-x070-World-0.1B-v")) if c < 3 else None
        show(f"Batched K = 32, conversation {c}, {tag} (`{f}`)", s["prompt_ids"][c], s["generated_ids"][c], ref,
             f"top-1 vs its own polynomial model: {s['top1_agree_per_conv'][c]} / {s['tokens_per_conv'][c]}")
open("TEXTS_20261105.md", "w").write("\n".join(out) + "\n"); print("written", len(out), "lines")
