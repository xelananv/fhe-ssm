#!/usr/bin/env python3
"""One line per prefill record (run_pf*.jsonl): passes, s/pass, s per prompt token, top-1, errors, refreshes, pairs.
Usage: python3 tab_prefill.py FILE...  (the same extraction as tab_controls.py's prefill block, any file names)."""
import json, statistics as st, sys, re
def load(f): return [json.loads(l) for l in open(f) if l.strip()]
def med(x): return st.median(x) if x else float('nan')
def pairs(summary):
    m = re.search(r"'pairs': \(?(\d+)", str(summary.get("planned_stats") or ""))
    return m.group(1) if m else "?"
for f in sys.argv[1:]:
    rows = load(f); h = rows[0]; s = rows[-1]
    if s.get("kind") != "summary": print(f"{f}: no summary (partial record, {len(rows)} rows)"); continue
    pf = [r for r in rows if r["kind"] == "prefill"]
    tok = [r for r in rows if r["kind"] == "token"]
    tpf = [r for r in tok if r["mode"] == "prefill"]; tdc = [r for r in tok if r["mode"] == "decode"]
    oor = [r for r in tok if r["poly_sites_out_of_range"]]
    nb = sum(v["n_bootstraps"] for k, v in h["programs"].items() if k.startswith("pf_block"))
    nbd = sum(v["n_bootstraps"] for k, v in h["programs"].items() if not k.startswith("pf_block"))
    print(f"{f}: keys={h.get('keys')} has_secret={h.get('has_secret')} prompt {len(h['tokens'])} tokens | passes {len(pf)} ({s['prefilled_tokens']} prefilled) | s/pass {' / '.join('%.2f' % r['seconds'] for r in pf)} | s per prompt token {s['prefill_seconds_per_token']:.3f} (sum of passes / prefilled {sum(r['seconds'] for r in pf)/s['prefilled_tokens']:.3f}) | top1 pf {s['top1_agree_prefill']}/{len(tpf)} dec {s['top1_agree']-s['top1_agree_prefill']}/{len(tdc)} | relErr med pf {med([r['logits_relErrRms'] for r in tpf]):.1e} dec {med([r['logits_relErrRms'] for r in tdc]):.1e} max {s['logits_relErrRms_max']:.1e} | mirror max {s['hidden_relErrRms_vs_mirror_max']:.3f} | dec s/tok med {s['seconds_per_token_median']:.3f} | setup {s['setup_s']} wall {s['wall_s']} | oor {len(oor)} | pf refreshes {nb} dec {nbd} | pairs {pairs(s)} | warmup {[r['seconds'] for r in rows if r['kind']=='warmup']}")
