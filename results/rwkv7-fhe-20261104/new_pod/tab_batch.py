#!/usr/bin/env python3
"""One block per batched record (batch*.jsonl): passes, s/pass, aggregate tok/s, refreshes, pairs, fidelity.
Usage: python3 tab_batch.py FILE...  (the same extraction as tab_controls.py's batched block, any file names)."""
import json, statistics as st, sys, re
def load(f): return [json.loads(l) for l in open(f) if l.strip()]
def med(x): return st.median(x) if x else float('nan')
def pairs(summary):
    m = re.search(r"'pairs': \(?(\d+)", str(summary.get("planned_stats") or ""))
    return m.group(1) if m else "?"
for f in sys.argv[1:]:
    rows = load(f); h = rows[0]; s = rows[-1]
    if s.get("kind") != "summary": print(f"{f}: no summary (partial record, {len(rows)} rows)"); continue
    ps = [r for r in rows if r["kind"] == "pass"]; tok = [r for r in rows if r["kind"] == "token"]
    sec = [r["seconds"] for r in ps]; nb = {k: v["n_bootstraps"] for k, v in h["programs"].items()}
    oor = [r for r in tok if r["poly_sites_out_of_range"]]
    perpass = {}
    for r in tok: perpass.setdefault(r["pass"], []).append(r["logits_relErrRms"])
    pm = [med(perpass[k]) for k in sorted(perpass)]
    print(f"{f}: keys={h.get('keys')} has_secret={h.get('has_secret')} layout={h.get('layout')} plan={str(h.get('plan'))[-40:]} | passes {len(ps)} | s/pass med {med(sec):.2f} after-first {med(sec[1:]):.2f} min {min(sec):.2f} max {max(sec):.2f} first {sec[0]:.2f} | agg tok/s at 32 active {32/med(sec):.3f} | summary agg {s.get('aggregate_tokens_per_s')} full {s.get('aggregate_tokens_per_s_full')} | compiled refreshes {sum(nb.values())} {nb} | top1 {s['top1_agree']}/{s['tokens']} | relErr med {s['logits_relErrRms_median']:.2e} max {s['logits_relErrRms_max']:.2e} | per-pass med first {pm[0]:.1e} last {pm[-1]:.1e} | mirror max {s['hidden_relErrRms_vs_mirror_max']:.4f} | oor {len(oor)} | setup {s['setup_s']} wall {s['wall_s']} | pairs {pairs(s)} | pool {h.get('pool')} stc {h.get('stc_cache')}")
    print("   active per pass:", sorted(set(r['active'] for r in ps)), "| top1 per conv diff vs tokens:", [(i, a, b) for i, (a, b) in enumerate(zip(s['top1_agree_per_conv'], s['tokens_per_conv'])) if a != b])
    print("   shed_twin:", h.get("shed_twin"))
