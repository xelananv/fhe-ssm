#!/usr/bin/env python3
"""Prompt selection for the long AUTOREGRESSIVE run (author, 2026-09-19): a mix of reading lengths instead of 64 short openers.

Plaintext only (the exact recurrent mirror, spec_decode/plain_recurrent.py). Two steps:
  build   : ~640 candidates in four classes, one prompt per line (single-line, no tabs):
              A long reading   (100-130 tokens)  \\  passages of the locally cached FineWeb-Edu shard (quality int_score >= 4),
              B medium reading ( 40- 60 tokens)  /   cut at a sentence end; the model's own training distribution
              C short reading  ( 10- 16 tokens): the opening words of such passages, cut at a word boundary
              D standard short (  4- 10 tokens): the demo's prompt files (results/dense-demo-s31/sessions/*.txt)
  select  : every candidate is rolled out greedily to --ticks fed tokens (prompt + generation), exactly as fidelity_tick.py
            --free-run would feed it if the FHE argmax equals the plaintext argmax, and scored on what decides a clean long run:
              HARD  newton : the seeded Newton rsqrt of every norm site with the run's seed (frac:iters) converges (relErr < 1e-6)
                             at EVERY tick, and max ms/hi stays below --hi-max (the window's reach is ~80x; default 30)
              HARD  loop   : no repetition loop (<= 3 distinct tokens in any 24-token window of the generated part)
              SOFT  nearTie: number of ticks whose plaintext top-1 margin is below --margin logits (each is a coin flip for the
                             encrypted argmax: fewer = a cleaner token-exact record); then distinct-2 of the generation (higher first)
            and the best --per-class of each class are written, interleaved A,B,C,D so that every region of the block mixes classes.
Outputs: <out>.candidates.tsv (class, tokens, prompt), <out>.scores.jsonl (one line per candidate), <out>.txt (the selected
prompts, `# type:` comment lines allowed by the drivers), <out>.report.md.
"""
import argparse, glob, json, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-eval"))
import numpy as np, torch                      # noqa: E402
import fhe_client as FC                        # noqa: E402
from plain_recurrent import RecurrentRef, load  # noqa: E402
import decode_policy as DP                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--ticks", type=int, default=172)
ap.add_argument("--tag", default="pbd430a"); ap.add_argument("--art-dir", default=os.path.expanduser("~/Documents/fhe-ssm-backup/mac_art"))
ap.add_argument("--parquet", default=os.path.expanduser("~/.cache/huggingface/hub/datasets--HuggingFaceTB--smollm-corpus/snapshots/3ba9d605774198c5868892d7a8deda78031a781f/fineweb-edu-dedup/train-00000-of-00234.parquet"))
ap.add_argument("--per-class-candidates", type=int, default=160); ap.add_argument("--per-class", type=int, default=16)
ap.add_argument("--frac", type=float, default=0.1); ap.add_argument("--iters", type=int, default=12); ap.add_argument("--eps", type=float, default=1e-5)
ap.add_argument("--hi-max", type=float, default=30.0); ap.add_argument("--margin", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--seed", type=int, default=20260919)
ap.add_argument("--decode-policy", choices=("greedy", "norepeat"), default="greedy"); ap.add_argument("--rep-penalty", type=float, default=1.3)
ap.add_argument("--rep-window", type=int, default=64); ap.add_argument("--no-repeat-ngram", type=int, default=4)
ap.add_argument("--extra-candidates", default="", help="a file of SELF-WRITTEN candidates with '# type:A|B|C' section headers: scored INSTEAD of the corpus (A/B cut at a sentence end inside the band)")
ap.add_argument("--a-band", default="100:130", help="class A prompt length band in tokens (lo:hi)"); ap.add_argument("--b-band", default="40:60")
a = ap.parse_args()
ALO, AHI = [int(x) for x in a.a_band.split(":")]; BLO, BHI = [int(x) for x in a.b_band.split(":")]
tok = FC.get_tokenizer(); rng = np.random.default_rng(a.seed)
BAN = [int(tok.eos_token_id)] if getattr(tok, "eos_token_id", None) is not None else []
REPO = os.path.dirname(HERE)


def clean(s):
    return re.sub(r"\s+", " ", s.replace("\t", " ")).strip()


def prose(p):
    if len(p) < 250 or not p[0].isupper(): return False
    if re.search(r"http|www\.|@|\||\{|\}|<|>|©|\bclick\b|\bcookie", p, re.I): return False
    letters = sum(c.isalpha() or c == " " for c in p) / len(p)
    return letters > 0.93 and p.count(".") >= 3


def cut_sentence(ids, lo, hi):
    """longest prefix of ids with lo <= len <= hi that ends at a sentence end; None if there is none"""
    for n in range(min(hi, len(ids)), lo - 1, -1):
        t = tok.decode(ids[:n])
        if t.rstrip().endswith((".", "?", "!")) and not t.rstrip().endswith(("Mr.", "Dr.", "e.g.", "i.e.")): return ids[:n]
    return None


def build_extra():
    cands, cls, drop = [], None, 0
    for l in open(a.extra_candidates):
        l = clean(l)
        if not l: continue
        if l.startswith("# type:"): cls = l.split(":", 1)[1].strip(); continue
        ids = tok.encode(l)
        if cls == "A": c = cut_sentence(ids, ALO, AHI)
        elif cls == "B": c = cut_sentence(ids, BLO, BHI)
        else: c = ids if 8 <= len(ids) <= 16 else None
        if c is None: drop += 1; continue
        cands.append((cls, tok.decode(c)))
    with open(a.out + ".candidates.tsv", "w") as fo:
        for c, x in cands: fo.write(f"{c}\t{len(tok.encode(x))}\t{x}\n")
    print({k: sum(1 for c, _ in cands if c == k) for k in "ABC"}, "self-written candidates kept,", drop, "outside their band", flush=True)
    return cands


def build():
    if a.extra_candidates: return build_extra()
    import pyarrow.parquet as pq
    f = pq.ParquetFile(a.parquet); A, B, C = [], [], []; rgs = rng.permutation(f.num_row_groups)[:12]
    for rg in rgs:
        tb = f.read_row_group(int(rg), columns=["text", "metadata"]); texts = tb.column("text").to_pylist(); sc = [m.get("int_score") if m else None for m in tb.column("metadata").to_pylist()]
        for txt, s in zip(texts, sc):
            if s is None or s < 4: continue
            paras = [clean(p) for p in txt.split("\n")]; paras = [p for p in paras if prose(p)]
            if not paras: continue
            p = paras[0]; ids = tok.encode(p)
            if len(ids) >= ALO and len(A) < a.per_class_candidates:
                c = cut_sentence(ids, ALO, AHI)
                if c: A.append(tok.decode(c)); continue
            if len(ids) >= BLO and len(B) < a.per_class_candidates:
                c = cut_sentence(ids, BLO, BHI)
                if c: B.append(tok.decode(c)); continue
            if len(C) < a.per_class_candidates and len(ids) >= 20:
                n = int(rng.integers(10, 17)); t = tok.decode(ids[:n]); t = t[: t.rfind(" ")] if " " in t else t
                if 8 <= len(tok.encode(t)) <= 16 and not t.rstrip().endswith((".", ",")): C.append(t)
        if min(len(A), len(B), len(C)) >= a.per_class_candidates: break
    D = []; seen = set()
    for fn in sorted(glob.glob(os.path.join(REPO, "results/dense-demo-s31/sessions/demo_prompts_64.txt")) + glob.glob(os.path.join(REPO, "results/dense-demo-s31/sessions/prompt_candidates*_20260918.txt"))):
        for l in open(fn):
            l = clean(l)
            if not l or l.startswith("# type:") or l in seen: continue
            n = len(tok.encode(l))
            if 3 <= n <= 12: seen.add(l); D.append(l)
    D = D[: a.per_class_candidates]
    cands = [("A", x) for x in A] + [("B", x) for x in B] + [("C", x) for x in C] + [("D", x) for x in D]
    with open(a.out + ".candidates.tsv", "w") as fo:
        for c, x in cands: fo.write(f"{c}\t{len(tok.encode(x))}\t{x}\n")
    print({k: sum(1 for c, _ in cands if c == k) for k in "ABCD"}, "candidates ->", a.out + ".candidates.tsv", flush=True)
    return cands


def rollout(cands, model, cfg, seeds):
    out = []
    for b0 in range(0, len(cands), a.batch):
        t0 = time.time(); chunk = cands[b0:b0 + a.batch]; Bn = len(chunk)
        ids = [tok.encode(x) for _, x in chunk]; P = [len(x) for x in ids]; T = a.ticks
        rr = RecurrentRef(model, cfg, Bn); fed = [[x[0]] for x in ids]; gen = [[] for _ in range(Bn)]
        worst = np.zeros(Bn); hi = np.zeros(Bn); lo = np.full(Bn, np.inf); near = np.zeros(Bn, dtype=int); minmargin = np.full(Bn, np.inf)
        maxS = np.zeros(Bn); hisite = [""] * Bn

        def site(name, x):
            sd = seeds.get(name)
            if not sd: return
            ms = (x.double() ** 2).mean(-1).numpy() + a.eps; mid = float(np.sqrt(sd["ms_range"][0] * sd["ms_range"][1]))
            y = np.full_like(ms, a.frac * (sd["a"] + sd["b"] * mid))
            with np.errstate(all="ignore"):
                for _ in range(max(int(sd["iters"]), a.iters)): y = y * (1.5 - 0.5 * ms * y * y)
                e = np.abs(y * np.sqrt(ms) - 1.0)
            e = np.where(np.isfinite(e), e, np.inf); np.maximum(worst, e, out=worst)
            r = ms / sd["ms_range"][1]
            for l in np.nonzero(r > hi)[0]: hisite[l] = name
            np.maximum(hi, r, out=hi); np.minimum(lo, ms / sd["ms_range"][0], out=lo)
        for t in range(T):
            lg = rr.step([f[-1] for f in fed], site_hook=site)
            maxS = np.maximum(maxS, np.max(np.stack([s.abs().max(-1).values.numpy() for s in rr.S]), axis=0))
            if a.decode_policy == "greedy":
                top2 = torch.topk(lg, 2, dim=-1); mg = (top2.values[:, 0] - top2.values[:, 1]).numpy(); picks = top2.indices[:, 0].numpy()
            else:                                   # the margin that matters is the one of the logits the policy takes its argmax over
                lgn = lg.double().numpy(); mg = np.zeros(Bn); picks = np.zeros(Bn, dtype=np.int64)
                for l in range(Bn):
                    z = DP.modified(lgn[l], fed[l], a.decode_policy, a.rep_penalty, a.rep_window, a.no_repeat_ngram, BAN)
                    i2 = np.argpartition(z, -2)[-2:]; i2 = i2[np.argsort(z[i2])]; picks[l] = i2[1]; mg[l] = z[i2[1]] - z[i2[0]]
            near += (mg < a.margin); minmargin = np.minimum(minmargin, mg)
            for l in range(Bn):
                if t + 1 < P[l]: nxt = ids[l][t + 1]
                else: nxt = int(picks[l]); gen[l].append(nxt)
                fed[l].append(nxt)
        for l in range(Bn):
            g = gen[l]; loop = any(len(set(g[i:i + 24])) <= 3 for i in range(0, max(len(g) - 23, 0)))
            d2 = len(set(zip(g, g[1:]))) / max(len(g) - 1, 1)
            seen8, rep8 = set(), None                # first position at which an 8-token sequence of the generation repeats
            for i in range(max(len(g) - 7, 0)):
                k8 = tuple(g[i:i + 8])
                if k8 in seen8: rep8 = i; break
                seen8.add(k8)
            out.append({"class": chunk[l][0], "prompt": chunk[l][1], "promptTokens": P[l], "genTokens": len(g), "newtonWorst": float(worst[l]), "msHiMax": float(hi[l]),
                        "msHiSite": hisite[l], "msLoMin": float(lo[l]), "maxAbsState": float(maxS[l]), "nearTies": int(near[l]), "minMargin": float(minmargin[l]),
                        "loop": bool(loop), "distinct2": round(d2, 4), "firstRepeat8": rep8, "text": tok.decode(g)})
        print(f"batch {b0 // a.batch + 1}/{(len(cands) + a.batch - 1) // a.batch}: {Bn} lanes x {T} ticks in {time.time() - t0:.1f} s", flush=True)
    return out


def main():
    cands = build(); model, cfg = load(a.tag, a.art_dir)
    seeds = json.load(open(os.path.join(a.art_dir, f"{a.tag}_seeds.json")))
    sc = rollout(cands, model, cfg, seeds)
    with open(a.out + ".scores.jsonl", "w") as fo:
        for r in sc: fo.write(json.dumps(r) + "\n")
    sel = {}; rep = ["# Prompt selection for the long autoregressive run (plaintext emulation; spec_decode/prompt_select_longrun.py)", "",
                     f"ticks {a.ticks}, Newton seed {a.frac}/{a.iters}, hard filters: newtonWorst < 1e-6, msHiMax < {a.hi_max}, no repetition loop; soft: nearTies (margin < {a.margin}), distinct-2", "",
                     "| class | candidates | pass newton | pass ms/hi | pass loop | pass all | selected | nearTies of the selected (min..max) | msHiMax of the selected (max) |", "|---|---|---|---|---|---|---|---|---|"]
    for k in "ABCD":
        rows = [r for r in sc if r["class"] == k]; ok_n = [r for r in rows if r["newtonWorst"] < 1e-6]; ok_h = [r for r in rows if r["msHiMax"] < a.hi_max]; ok_l = [r for r in rows if not r["loop"]]
        ok = [r for r in rows if r["newtonWorst"] < 1e-6 and r["msHiMax"] < a.hi_max and not r["loop"]]
        ok.sort(key=lambda r: (r["nearTies"], -r["distinct2"], r["msHiMax"])); sel[k] = ok[: a.per_class]
        s = sel[k]; rep.append(f"| {k} | {len(rows)} | {len(ok_n)} | {len(ok_h)} | {len(ok_l)} | {len(ok)} | {len(s)} | {min([r['nearTies'] for r in s], default='-')}..{max([r['nearTies'] for r in s], default='-')} | {max([r['msHiMax'] for r in s], default=0):.2f} |")
    order = []
    for i in range(a.per_class):
        for k in "ABCD":
            if i < len(sel[k]): order.append(sel[k][i])
    with open(a.out + ".txt", "w") as fo:
        for r in order: fo.write(r["prompt"] + "\n")
    rep += ["", f"selected {len(order)} prompts -> `{os.path.basename(a.out)}.txt` (interleaved A,B,C,D)", "", "| lane | class | prompt tokens | generated | nearTies | minMargin | msHiMax (site) | max abs state | distinct-2 | prompt (start) | generation (start) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(order):
        rep.append(f"| {i} | {r['class']} | {r['promptTokens']} | {r['genTokens']} | {r['nearTies']} | {r['minMargin']:.4f} | {r['msHiMax']:.2f} ({r['msHiSite']}) | {r['maxAbsState']:.2f} | {r['distinct2']} | {r['prompt'][:60]!r} | {r['text'][:80]!r} |")
    open(a.out + ".report.md", "w").write("\n".join(rep) + "\n"); print("\n".join(rep[:12])); print("selected", len(order))


if __name__ == "__main__":
    main()
