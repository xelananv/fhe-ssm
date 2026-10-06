#!/usr/bin/env python3
"""Tables for PROBE_B_REPORT.md, computed ONLY from the JSONL records beside this script (records-only: every number
printed here is a function of lines in those files; the model column re-evaluates the library's own constants from
vendor/openfhe-development/src/pke/include/scheme/ckksrns/ckksrns-fhe.h exactly as ../boot_overflow_estimate.py does).

usage: analyze_probe_b.py            -> writes probe_b_tables.md next to itself
"""
import json, math, os, re, statistics as st, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
HDR = os.path.join(REPO, "vendor/openfhe-development/src/pke/include/scheme/ckksrns/ckksrns-fhe.h")
src = open(HDR).read()
coeffs = [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", re.search(r"g_coefficientsUniform\{(.*?)\};", src, re.S).group(1))]
K = int(re.search(r"K_UNIFORM\s*=\s*(\d+)", src).group(1)); R = int(re.search(r"R_UNIFORM\s*=\s*(\d+)", src).group(1))


def g(t):   # Chebyshev series (c0/2 + sum c_k T_k) then R double-angle steps y <- 2y^2 - (2pi)^(-2^i), i = 1-R..0
    Tkm, Tk = 1.0, t; s = coeffs[0] / 2 + coeffs[1] * t
    for c in coeffs[2:]:
        Tkm, Tk = Tk, 2 * t * Tk - Tkm; s += c * Tk
    y = s
    for i in range(1 - R, 1):
        y = 2 * y * y - (2 * math.pi) ** (-(2.0 ** i))
    return y


def model_eps(I):   # signed error of the library's approximation at an integer overflow (the exact value there is sin(2 pi I) = 0)
    try:
        return g(I / K)
    except OverflowError:
        return float("inf")


def Q(x): return 0.5 * math.erfc(x / math.sqrt(2))


def load(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            try: recs.append(json.loads(line))
            except json.JSONDecodeError: pass   # a line cut by a kill is not a record
    return recs


def fmt(x, p=3):
    if x is None: return "-"
    if isinstance(x, str): return x
    if x == 0: return "0"
    if math.isinf(x): return "inf"
    return f"{x:.{p}e}"


out = []
def P(s=""): out.append(s)


# ---------------------------------------------------------------- 1. soundness of the hand-built key set
P("## T1. Hand-built key set vs the library's KeyGen (a = 1, no planting)\n")
P("| file | mode | N | F | sum s_i^2 | sigma_I pred | sigma_I measured (mean over trials) | trials | decode fails | adjust==library | relErrRms median | relErrRms max | coefErrRms median | bootRestored |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(HERE, "sound_*.jsonl"))):
    recs = load(f); h = [r for r in recs if r.get("rec") == "header"][0]
    for ks in [r for r in recs if r.get("rec") == "keyset"]:
        tr = [r for r in recs if r.get("rec") == "trial" and r["a"] == ks["a"]]
        ok = [r for r in tr if not r.get("decodeFail") and "relErrRms" in r]
        P(f"| {os.path.basename(f)} | {h['mode']} | {h['ringDim']} | {h['correctionFactor']} | {ks['sumS2']} | {ks['sigmaIpred']:.3f} | "
          f"{st.mean(r['sigmaIhat'] for r in tr):.3f} | {len(tr)} | {sum(1 for r in tr if r.get('decodeFail'))} | "
          f"{sum(1 for r in tr if r['adjustMatchesLibrary'])}/{len(tr)} | {fmt(st.median(r['relErrRms'] for r in ok))} | {fmt(max(r['relErrRms'] for r in ok))} | "
          f"{fmt(st.median(r['coefErrRms'] for r in ok))} | {sum(1 for r in tr if r.get('bootRestored'))}/{len(tr)} |")
P()

# ---------------------------------------------------------------- 2. planted sweeps
P("## T2. Planted single-coefficient overflow (library KeyGen, uniform ternary secret)\n")
P("One coefficient j* carries the overflow I*; every other coefficient stays far inside |I| <= K (column `max other |I|`). "
  "`share` = (e_j*^2 + e_(j*+N/2)^2) / sum_j e_j^2 over the decode-free coefficient errors (1.0 = all the damage sits on the planted "
  "coefficient and the other half of its slot). `coef err at j*` and `at j*+N/2` are the decode-free coefficient errors (message units) at the planted coefficient and at the other half of "
  "the same slot; `floor` is the median |coefficient error| over all N coefficients of the same trial. `model` = |g(I/K)| * 2^F, the library's "
  "approximation re-evaluated in float64 outside its fitted range, scaled by the 2^F the bootstrap multiplies back in.\n")
for f in sorted(glob.glob(os.path.join(HERE, "planted_*.jsonl"))):
    recs = load(f)
    hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; F = h["correctionFactor"]; N = h["ringDim"]
    tr = [r for r in recs if r.get("rec") == "trial"]
    if not tr: continue
    nver = sum(1 for r in tr if r.get("plantVerified")); nadj = sum(1 for r in tr if r["adjustMatchesLibrary"]); nhit = sum(1 for r in tr if r["plantAchieved"] == r["plantTarget"])
    P(f"### {os.path.basename(f)} -- N = {N}, F = {F}, plant index {h['plantIndex']}, {len(tr)} bootstraps; plantVerified {nver}/{len(tr)}, "
      f"achieved == target {nhit}/{len(tr)}, adjust==library {nadj}/{len(tr)}\n")
    P("| I* | n | decode exceptions | relErrRms (min / median / max over non-throwing) | share of squared coef error on {j*, j*+N/2} (median) | |coef err at j*| (min / median / max) | |coef err at j*+N/2| median | floor | model |g|*2^F | max other |I| |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for T in sorted({r["plantTarget"] for r in tr}, key=lambda v: abs(v)):
        g_ = [r for r in tr if r["plantTarget"] == T]
        ok = [r for r in g_ if not r.get("decodeFail") and r.get("relErrRms") is not None]
        ce = [abs(r["coefErrAtArgmaxI"]) for r in g_ if r.get("coefErrAtArgmaxI") is not None and r["argmaxI"] == h["plantIndex"]]
        pe = [abs(r["coefErrAtPartner"]) for r in g_ if r.get("coefErrAtPartner") is not None and r["argmaxI"] == h["plantIndex"]]
        fl = [r["coefErrMedAbs"] for r in g_ if r.get("coefErrMedAbs") is not None]
        oth = max(max(abs(v) for v in r["top5I"][1:]) for r in g_)
        rel = f"{fmt(min(r['relErrRms'] for r in ok))} / {fmt(st.median(r['relErrRms'] for r in ok))} / {fmt(max(r['relErrRms'] for r in ok))}" if ok else "-"
        cej = f"{fmt(min(ce))} / {fmt(st.median(ce))} / {fmt(max(ce))}" if ce else "-"
        mm = [(r["coefErrAtArgmaxI"] ** 2 + r["coefErrAtPartner"] ** 2) / (N * r["coefErrRms"] ** 2) for r in g_
              if r.get("coefErrRms") and r.get("coefErrAtArgmaxI") is not None and r["argmaxI"] == h["plantIndex"] and math.isfinite(r["coefErrRms"] ** 2)]
        mms = f"{st.median(mm):.4f}" if mm else "-"
        P(f"| {T} | {len(g_)} | {sum(1 for r in g_ if r.get('decodeFail'))} | {rel} | {mms} | {cej} | {fmt(st.median(pe)) if pe else '-'} | {fmt(st.median(fl)) if fl else '-'} | "
          f"{fmt(abs(model_eps(abs(T))) * 2 ** F)} | {oth} |")
    P()

# ---------------------------------------------------------------- 2s. thresholds read off the planted sweeps
P("## T2s. Thresholds read off the planted sweeps (each entry is the smallest |I*| of that sweep at which the condition first holds)\n")
P("| sweep | N | F | I* grid | baseline relErrRms (median over |I*| <= 510) | median relErrRms > 2x baseline | median >= 1e-3 | median >= 1e-2 | median >= 1e-1 | first decode exception | all trials throw from | whole ciphertext destroyed (median coefficient error > 1) | measured / model, median over 518 <= |I*| <= 540 (min..max) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(HERE, "planted_*.jsonl"))):
    recs = load(f); hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; F = h["correctionFactor"]; N = h["ringDim"]
    tr = [r for r in recs if r.get("rec") == "trial"]
    Ts = sorted({abs(r["plantTarget"]) for r in tr})
    def grp(T): return [r for r in tr if abs(r["plantTarget"]) == T]
    def medrel(T):
        ok = [r["relErrRms"] for r in grp(T) if not r.get("decodeFail") and r.get("relErrRms") is not None]
        return st.median(ok) if len(ok) == len(grp(T)) else (float("inf") if not ok else max(st.median(ok), float("inf") if len(ok) < len(grp(T)) / 2 else st.median(ok)))
    base_pool = [r["relErrRms"] for r in tr if abs(r["plantTarget"]) <= 510 and r.get("relErrRms") is not None]
    base = st.median(base_pool) if base_pool else None
    def first(cond):
        for T in Ts:
            if cond(T): return T
        return None
    thr2 = first(lambda T: base is not None and medrel(T) > 2 * base)
    t3 = first(lambda T: medrel(T) >= 1e-3); t2 = first(lambda T: medrel(T) >= 1e-2); t1 = first(lambda T: medrel(T) >= 1e-1)
    fe = first(lambda T: any(r.get("decodeFail") for r in grp(T)))
    # "all trials throw from": smallest T such that every trial at every grid point >= T throws
    ae = None
    for T in reversed(Ts):
        if all(r.get("decodeFail") for r in grp(T)): ae = T
        else: break
    wd = first(lambda T: st.median([r["coefErrMedAbs"] for r in grp(T) if r.get("coefErrMedAbs") is not None] or [0]) > 1)
    ratios = [abs(r["coefErrAtArgmaxI"]) / (abs(model_eps(abs(r["plantTarget"]))) * 2 ** F) for r in tr
              if 518 <= abs(r["plantTarget"]) <= 540 and r.get("coefErrAtArgmaxI") is not None and r["argmaxI"] == h["plantIndex"]]
    rr = f"{st.median(ratios):.3f} ({min(ratios):.3f}..{max(ratios):.3f}), n = {len(ratios)}" if ratios else "-"
    P(f"| {os.path.basename(f)} | {N} | {F} | {min(Ts)}..{max(Ts)} step {Ts[1] - Ts[0] if len(Ts) > 1 else '-'} | {fmt(base)} | {thr2} | {t3} | {t2} | {t1} | {fe} | {ae} | {wd} | {rr} |")
P()

# ---------------------------------------------------------------- 3. dose-response with hand-built wide secrets
P("## T3. Dose-response: secrets with coefficients uniform in {-a..a}, random fresh encryptions, exact I per trial\n")
BOUNDS = (512, 521, 523, 528, 532, 534, 538)
for f in sorted(glob.glob(os.path.join(HERE, "dose_*.jsonl"))):
    recs = load(f)
    hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; N = h["ringDim"]; F = h["correctionFactor"]
    P(f"### {os.path.basename(f)} -- N = {N}, F = {F}, levelBudget {h['levelBudget']}, depth {h['depth']}\n")
    P("Outcome classes per trial (absolute thresholds, the same for every width): `exception` = the library's decode threw; `garbage` = relErrRms >= 0.1; "
      "`degraded` = 1e-3 <= relErrRms < 0.1; `normal` = relErrRms < 1e-3. `baseline` = median relErrRms of that width's trials with max|I| <= 500 "
      "(`-` when there is no such trial).\n")
    P("The three `predicted` columns use the thresholds the PLANTED sweep of the same ring and F measured (T2s, planted_r13_F10_pos: median relErrRms >= 1e-3 "
      "from |I*| = 522, >= 0.1 from 529, first decode exception at 535), i.e. P(max|I| > 521), P(max|I| > 528), P(max|I| > 534) under the Gaussian model.\n")
    P("| a | sum s_i^2 | sigma_I pred | sigma_I measured | K/sigma | trials | adjust==library | baseline relErrRms | normal | degraded | garbage | exception | observed P(not normal) | predicted P(max|I|>521) | observed P(garbage or exception) | predicted P(max|I|>528) | observed P(exception) | predicted P(max|I|>534) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for ks in sorted([r for r in recs if r.get("rec") == "keyset"], key=lambda r: r["a"]):
        a = ks["a"]; tr = [r for r in recs if r.get("rec") == "trial" and r["a"] == a]
        if not tr: continue
        base_pool = [r["relErrRms"] for r in tr if not r.get("decodeFail") and r.get("relErrRms") is not None and r["maxAbsI"] <= 500]
        base = st.median(base_pool) if base_pool else None
        cls = {"normal": 0, "degraded": 0, "garbage": 0, "exception": 0}
        for r in tr:
            if r.get("decodeFail") or r.get("bootThrew"): c = "exception"
            elif r.get("relErrRms") is None or r["relErrRms"] >= 0.1: c = "garbage"
            elif r["relErrRms"] >= 1e-3: c = "degraded"
            else: c = "normal"
            r["_cls"] = c; cls[c] += 1
        n = len(tr); rows[a] = (ks, tr, base)
        P(f"| {a} | {ks['sumS2']} | {ks['sigmaIpred']:.2f} | {st.mean(r['sigmaIhat'] for r in tr):.2f} | {ks['KoverSigma']:.3f} | {n} | "
          f"{sum(1 for r in tr if r['adjustMatchesLibrary'])}/{n} | {fmt(base)} | {cls['normal']} | {cls['degraded']} | {cls['garbage']} | {cls['exception']} | "
          f"{(n - cls['normal']) / n:.4f} | {1 - (1 - 2 * Q(521.5 / ks['sigmaIpred'])) ** N:.4g} | {(cls['garbage'] + cls['exception']) / n:.4f} | "
          f"{1 - (1 - 2 * Q(528.5 / ks['sigmaIpred'])) ** N:.4g} | {cls['exception'] / n:.4f} | {1 - (1 - 2 * Q(534.5 / ks['sigmaIpred'])) ** N:.4g} |")
    P()
    P("Predicted P(max_j |I_j| > B) = 1 - (1 - 2Q((B + 0.5)/sigma_I))^N with sigma_I = sqrt((sum s_i^2 + 1)/12), next to the OBSERVED fraction of "
      "trials whose exact max|I| exceeded B (this checks the Gaussian model of I itself, independently of any error threshold):\n")
    P("| a | trials | " + " | ".join(f"B={B}: predicted P | expected count | observed count" for B in BOUNDS) + " |")
    P("|---|---|" + "---|---|---|" * len(BOUNDS))
    tot = {B: [0.0, 0, 0.0] for B in BOUNDS}
    for a, (ks, tr, base) in sorted(rows.items()):
        sig = ks["sigmaIpred"]; n = len(tr); cells = []
        for B in BOUNDS:
            p1 = 2 * Q((B + 0.5) / sig); pb = 1 - (1 - p1) ** N
            k = sum(1 for r in tr if r["maxAbsI"] > B)
            tot[B][0] += n * pb; tot[B][1] += k; tot[B][2] += n * pb * (1 - pb)
            cells.append(f"{pb:.4g} | {n * pb:.2f} | {k}")
        P(f"| {a} | {n} | " + " | ".join(cells) + " |")
    P("| all | " + str(sum(len(tr) for ks, tr, base in rows.values())) + " | " + " | ".join(f"- | {tot[B][0]:.2f} (binomial sd {math.sqrt(tot[B][2]):.2f}) | {tot[B][1]}" for B in BOUNDS) + " |")
    P()
    # error vs max|I|, pooled over the widths of this file
    P("Outcome vs the exact max|I| of the trial, pooled over all widths in this file (each row is one value range of max|I|):\n")
    P("| max|I| | trials | normal | degraded | garbage | exception | relErrRms: median (min..max) over non-throwing | worst coefficient is argmax|I| or its slot partner |")
    P("|---|---|---|---|---|---|---|---|")
    alltr = [(r, base) for a, (ks, tr, base) in rows.items() for r in tr]
    edges = [0, 480, 500, 505, 510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 536, 538, 540, 545, 550, 560, 580, 10 ** 9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(r, b) for r, b in alltr if lo < r["maxAbsI"] <= hi]
        if not sel: continue
        c = {k: sum(1 for r, _ in sel if r["_cls"] == k) for k in ("normal", "degraded", "garbage", "exception")}
        ratios = [r["relErrRms"] for r, b in sel if not r.get("decodeFail") and r.get("relErrRms") is not None]
        loc = [r for r, _ in sel if r.get("argmaxCoefErr") is not None and "argmaxI" in r]
        hit = sum(1 for r in loc if r["argmaxCoefErr"] in (r["argmaxI"], (r["argmaxI"] + N // 2) % N))
        rr = f"{st.median(ratios):.3g} ({min(ratios):.3g}..{max(ratios):.3g})" if ratios else "-"
        P(f"| {lo + 1}..{hi if hi < 10 ** 8 else 'inf'} | {len(sel)} | {c['normal']} | {c['degraded']} | {c['garbage']} | {c['exception']} | {rr} | {hit}/{len(loc)} |")
    P()
    # the sharp statement: smallest max|I| that is not normal, largest max|I| that is normal
    notn = sorted(r["maxAbsI"] for r, _ in alltr if r["_cls"] != "normal"); nor = sorted(r["maxAbsI"] for r, _ in alltr if r["_cls"] == "normal")
    bad = sorted(r["maxAbsI"] for r, _ in alltr if r["_cls"] in ("garbage", "exception")); exc = sorted(r["maxAbsI"] for r, _ in alltr if r["_cls"] == "exception")
    P(f"- trials: {len(alltr)}; largest max|I| with a NORMAL outcome: {nor[-1] if nor else '-'}; smallest max|I| with a non-normal outcome: {notn[0] if notn else '-'}; "
      f"smallest max|I| with garbage-or-exception: {bad[0] if bad else '-'}; smallest max|I| with an exception: {exc[0] if exc else '-'}; "
      f"largest max|I| WITHOUT an exception: {max((r['maxAbsI'] for r, _ in alltr if r['_cls'] != 'exception'), default='-')}")
    P(f"- non-normal trials with max|I| <= 512: {sum(1 for v in notn if v <= 512)}; normal trials with max|I| > 512: {sum(1 for v in nor if v > 512)}")
    P()

open(os.path.join(HERE, "probe_b_tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
