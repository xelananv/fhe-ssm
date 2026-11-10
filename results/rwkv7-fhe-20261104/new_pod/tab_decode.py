"""tab_decode.py: one line per decode record (run_*P*.jsonl): positions, top-1, logits relErrRms median/max, s/token
median and after-first, setup, keys mode, compiled refreshes, pairs. Usage: python3 tab_decode.py FILE..."""
import json, re, statistics as st, sys
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f) if l.strip()]
    h=rows[0]; s=rows[-1]; tok=[r for r in rows if r["kind"]=="token"]
    if s["kind"]!="summary": print(f, "INCOMPLETE:", len(tok), "tokens so far"); continue
    sec=[r["seconds"] for r in tok if "seconds" in r]
    flips=[(r["pos"], r.get("mode")) for r in tok if not r["top1_agree"]]
    nb=sum(v["n_bootstraps"] for k,v in h["programs"].items())
    pairs=re.search(r"'pairs': \(?(\d+)", str(s.get("planned_stats"))); pairs=pairs.group(1) if pairs else "?"
    oor=[r for r in tok if r["poly_sites_out_of_range"]]
    print(f"{f}: keys={h.get('keys')} has_secret={h.get('has_secret', s.get('has_secret'))} positions={len(tok)} top1={s['top1_agree']}/{s['tokens']} flips={flips} relErr med={s['logits_relErrRms_median']:.2e} max={s['logits_relErrRms_max']:.2e} s/tok med={s['seconds_per_token_median']:.3f} after-first={s['seconds_per_token_after_first']:.3f} (min {min(sec):.3f} max {max(sec):.3f}) setup={s['setup_s']} wall={s['wall_s']} refreshes={nb} pairs={pairs} oor={len(oor)} warmup={[r['seconds'] for r in rows if r['kind']=='warmup']}")
