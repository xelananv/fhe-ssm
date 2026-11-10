import json, statistics as st, sys, re
def load(f): return [json.loads(l) for l in open(f) if l.strip()]
def med(x): return st.median(x) if x else float('nan')
def pairs(summary):
    ps = summary.get("planned_stats") or ""
    m = re.search(r"'pairs': \(?(\d+)", str(ps))
    return m.group(1) if m else "?"
print("=== PREFILL ===")
for tag in ["best_", ""]:
    for p in ["L1","L5","L9","L25"]:
        f=f"r01b_final/run_pf32_{tag}{p}.jsonl"
        rows=load(f); h=rows[0]; s=rows[-1]
        pf=[r for r in rows if r["kind"]=="prefill"]
        tok=[r for r in rows if r["kind"]=="token"]
        tpf=[r for r in tok if r["mode"]=="prefill"]; tdc=[r for r in tok if r["mode"]=="decode"]
        oor=[r for r in tok if r["poly_sites_out_of_range"]]
        nb=sum(v["n_bootstraps"] for k,v in h["programs"].items() if k.startswith("pf_block"))
        nbd=sum(v["n_bootstraps"] for k,v in h["programs"].items() if not k.startswith("pf_block"))
        print(f"{tag or 'off_'}{p} ({len(h['tokens'])}) passes {len(pf)} ({s['prefilled_tokens']}) | s/pass {' / '.join('%.2f' % r['seconds'] for r in pf)} | s/prompt tok {s['prefill_seconds_per_token']:.3f} (from passes: {sum(r['seconds'] for r in pf)/s['prefilled_tokens']:.3f}) | top1 pf {s['top1_agree_prefill']}/{len(tpf)} dec {s['top1_agree']-s['top1_agree_prefill']}/{len(tdc)} | relErr med pf {med([r['logits_relErrRms'] for r in tpf]):.1e} dec {med([r['logits_relErrRms'] for r in tdc]):.1e} max {s['logits_relErrRms_max']:.1e} | mirror max {s['hidden_relErrRms_vs_mirror_max']:.3f} | dec s/tok med {s['seconds_per_token_median']:.3f} | setup {s['setup_s']} wall {s['wall_s']} | oor {len(oor)} | pf refreshes {nb} dec {nbd} | pairs {pairs(s)} | warmup {[r['seconds'] for r in rows if r['kind']=='warmup']}")
print("=== BATCHED ===")
for tag in ["_best",""]:
    f=f"r01b_final/batch32{tag}.jsonl"; rows=load(f); h=rows[0]; s=rows[-1]
    ps=[r for r in rows if r["kind"]=="pass"]; tok=[r for r in rows if r["kind"]=="token"]
    sec=[r["seconds"] for r in ps]
    nb={k:v["n_bootstraps"] for k,v in h["programs"].items()}
    oor=[r for r in tok if r["poly_sites_out_of_range"]]
    perpass={}
    for r in tok: perpass.setdefault(r["pass"],[]).append(r["logits_relErrRms"])
    pm=[med(perpass[k]) for k in sorted(perpass)]
    print(f"batch32{tag or '_off'}: passes {len(ps)} | s/pass med {med(sec):.2f} after-first {med(sec[1:]):.2f} min {min(sec):.2f} max {max(sec):.2f} first {sec[0]:.2f} | agg tok/s at 32 active {32/med(sec):.3f} | summary agg {s.get('aggregate_tokens_per_s')} full {s.get('aggregate_tokens_per_s_full')} | compiled refreshes {sum(nb.values())} {nb} | top1 {s['top1_agree']}/{s['tokens']} | relErr med {s['logits_relErrRms_median']:.2e} max {s['logits_relErrRms_max']:.2e} | per-pass med first {pm[0]:.1e} last {pm[-1]:.1e} | mirror max {s['hidden_relErrRms_vs_mirror_max']:.4f} | oor {len(oor)} | setup {s['setup_s']} wall {s['wall_s']} | pairs {pairs(s)} | pool {h.get('pool')} stc {h.get('stc_cache')}")
    print("   active per pass:", sorted(set(r['active'] for r in ps)), "| top1 per conv diff vs tokens:", [ (i,a,b) for i,(a,b) in enumerate(zip(s['top1_agree_per_conv'], s['tokens_per_conv'])) if a!=b])
    print("   planned_stats:", str(s.get("planned_stats"))[:300])
