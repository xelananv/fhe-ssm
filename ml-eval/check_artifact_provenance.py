"""check_artifact_provenance.py -- catch stale evals BEFORE they become claims.

Run this before citing any *_eval.json / *_export.json number, and after every
training session that syncs artifacts off a pod.

WHY
---
2026-08-02: an earlier analysis concluded "the d=256 model was never trained" from
`agnd256_train_log.jsonl` (1 row, step 0) and `agnd256_eval.json` (ppl 50,162).
Both were real files. Both were STALE: the eval ran at 07:44 and the real
weights landed at 10:32, ~2h48m later. The model had in fact trained for 29,800
steps and evaluates at ppl 508.85. The wrong conclusion propagated into four
documents before anyone caught it.

The generalisable failure: **an eval JSON does not carry the identity of the
weights it measured.** There is no step field, no hash, no provenance of any
kind. The only signal on disk is the modification time.

TWO CHECKS
----------
1. TIMESTAMP: does any eval/export/bundle predate the weights it claims to
   describe? (cheap, catches the whole class)
2. RE-EVAL: does the recorded number actually reproduce from the weights on
   disk? (definitive; `--reeval` runs it)

The re-eval is exact when the artifacts do correspond -- verified 2026-08-02:
  agnd768q2  recorded 234.4360727760061  ->  re-measured 234.43607256621758
  agnd768w   recorded  51.47387508932797 ->  re-measured  51.473875030144555
so any disagreement beyond ~1e-6 relative is a real checkpoint mismatch, not
numerical drift.

USAGE
    python ml-eval/check_artifact_provenance.py                # timestamp check
    python ml-eval/check_artifact_provenance.py --reeval TAG   # + re-evaluate
"""
import os, sys, glob, json, datetime, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART_DIRS = [os.path.join(ROOT, "results/a100-secure-20260730/artifacts"),
            os.path.join(HERE, "artifacts")]
EVAL_DIRS = [os.path.join(ROOT, "results/a100-secure-20260730"), HERE]


def mtime(p):
    if not p or os.path.islink(p) or not os.path.exists(p):
        return None
    return os.path.getmtime(p)


def find(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p) and not os.path.islink(p):
            return p
    return None


def tags():
    out = set()
    for d in ART_DIRS:
        for p in glob.glob(os.path.join(d, "*_weights.npz")):
            b = os.path.basename(p)[:-len("_weights.npz")]
            if "_s" not in b and "_v1" not in b:
                out.add(b)
    for d in EVAL_DIRS:
        for p in glob.glob(os.path.join(d, "*_eval.json")):
            out.add(os.path.basename(p)[:-len("_eval.json")])
    return sorted(out)


def recorded_ppl(tag):
    """(ppl, tokens, path) from *_export.json (512 tok) or *_eval.json (2048)."""
    for suffix in ("_export.json", "_eval.json"):
        p = find(EVAL_DIRS, tag + suffix)
        if not p:
            continue
        try:
            txt = open(p, errors="ignore").read()
        except OSError:
            continue
        ppl = tok = None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("{") and "native_test_ppl" in line:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ppl, tok = d.get("native_test_ppl"), d.get("tokens")
        if ppl is not None:
            return ppl, tok, p
    return None, None, None


def main():
    fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S") if t else "        -"
    print(f"{'model':<11} {'weights.npz':<17} {'eval':<17} {'export':<17} {'bundle':<17} verdict")
    stale = []
    for t in tags():
        w = find(ART_DIRS, f"{t}_weights.npz")
        wt = mtime(w) if w else None
        et = mtime(find(EVAL_DIRS, f"{t}_eval.json"))
        xt = mtime(find(EVAL_DIRS, f"{t}_export.json"))
        bt = mtime(find(ART_DIRS, f"bundle_{t}.bin"))
        flags = []
        if wt is None:
            flags.append("NO FINAL WEIGHTS ON DISK")
        else:
            for lbl, tt in (("eval", et), ("export", xt), ("BUNDLE", bt)):
                if tt and tt < wt - 60:
                    flags.append(f"{lbl} stale {(wt-tt)/3600:.1f}h")
        if flags:
            stale.append(t)
        print(f"{t:<11} {fmt(wt):<17} {fmt(et):<17} {fmt(xt):<17} {fmt(bt):<17} "
              f"{'; '.join(flags) if flags else 'ok'}")

    print()
    if stale:
        print("STALE / UNRESOLVED:", ", ".join(stale))
        print("  A stale BUNDLE is the dangerous one: an FHE run against it measures")
        print("  a different model than the ppl you quote beside it.")
        print(f"  Confirm with:  python {os.path.relpath(__file__, ROOT)} --reeval <tag>")
    else:
        print("All artifact groups consistent by timestamp.")

    if "--reeval" in sys.argv:
        tag = sys.argv[sys.argv.index("--reeval") + 1]
        ppl, tok, src = recorded_ppl(tag)
        print(f"\nre-evaluating {tag} against the weights currently on disk...")
        print(f"  recorded: {ppl} @ {tok} tokens   ({os.path.relpath(src, ROOT) if src else '-'})")
        if tok is None:
            print("  no recorded ppl to compare against"); return
        r = subprocess.run([sys.executable, os.path.join(HERE, "train_fhe_native_ssm.py"),
                            "eval", "--tag", tag, "--tokens", str(tok)],
                           cwd=HERE, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if "native_test_ppl" in l]
        if not line:
            print("  re-eval failed:\n", r.stdout[-400:], r.stderr[-400:]); return
        got = json.loads(line[-1])["native_test_ppl"]
        rel = abs(got - ppl) / ppl
        print(f"  measured: {got}")
        print(f"  relative difference: {rel:.2e}  ->  "
              + ("MATCH (artifacts correspond)" if rel < 1e-6 else
                 "*** MISMATCH: the weights on disk are NOT the checkpoint this number describes ***"))


if __name__ == "__main__":
    main()
