#!/usr/bin/env python3
"""Tables for K768_CPU_REPORT.md, computed ONLY from the JSONL records beside this script and (for the comparison with the
earlier stock-build sweeps) from ../planted_*.jsonl and ../sound_r13_stock.jsonl. Records-only: every number
printed is a function of lines in those files. The `model` columns re-evaluate the library's own constants (both tables,
both K, R) read from vendor/openfhe-development/src/pke/include/scheme/ckksrns/ckksrns-fhe.h in float64, exactly as
../../boot_overflow_ext_table.py does; they are arithmetic, not measurements, and are labelled as such.

usage: analyze_k768.py      -> writes k768_tables.md next to itself
"""
import json, math, os, re, statistics as st, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PREV = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
HDR = os.path.join(REPO, "vendor/openfhe-development/src/pke/include/scheme/ckksrns/ckksrns-fhe.h")
src = open(HDR).read()


def table(name):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", re.search(name + r"\s*\{(.*?)\};", src, re.S).group(1))]


R = int(re.search(r"R_UNIFORM\s*=\s*(\d+)", src).group(1))
TAB = {512: table("g_coefficientsUniform"), 768: table("g_coefficientsUniformExt")}
assert int(re.search(r"\bK_UNIFORM\s*=\s*(\d+)", src).group(1)) == 512 and int(re.search(r"\bK_UNIFORMEXT\s*=\s*(\d+)", src).group(1)) == 768


def g(t, coeffs):   # Chebyshev series (c0/2 + sum c_k T_k), then R double-angle steps y <- 2y^2 - (2pi)^(-2^i), i = 1-R..0
    Tkm, Tk = 1.0, t; s = coeffs[0] / 2 + coeffs[1] * t
    for c in coeffs[2:]:
        Tkm, Tk = Tk, 2 * t * Tk - Tkm; s += c * Tk
    y = s
    for i in range(1 - R, 1):
        y = 2 * y * y - (2 * math.pi) ** (-(2.0 ** i))
    return y


def model_eps(I, K):   # |error| of the library's approximation at an integer overflow (the exact value there is sin(2 pi I) = 0)
    try:
        return abs(g(I / K, TAB[K]))
    except OverflowError:
        return float("inf")


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


def med(v):
    v = list(v); return st.median(v) if v else None


out = []
def P(s=""): out.append(s)


def stderr_of(path):
    p = path[:-6] + ".stderr"
    if not os.path.exists(p): return "(no stderr file)"
    t = open(p).read().strip()
    return t if t else "(empty)"


def arm_of(path):
    """which library arm produced a file, from what the LIBRARY printed (stderr file) -- not from the file name"""
    t = stderr_of(path)
    if "OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1" in t: return "Ext + split (variant B)"
    if "(no split)" in t: return "Ext, no split (variant B library)"
    if "K = 768" in t: return "Ext (variant A)"
    if t == "(empty)": return "stock table (switch off" + (", variant B library)" if os.path.basename(path).startswith("B_") else ")")
    return "?"


def sweep_stats(path):
    """thresholds of one planted sweep, same definitions as ../analyze_probe_b.py (T2s), with the baseline pool generalised
    from '|I*| <= 510' to '|I*| <= K - 2' (K = the header's K: 510 for the stock table, 766 for the Ext table)."""
    recs = load(path); hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: return None
    h = hs[0]; F = h["correctionFactor"]; N = h["ringDim"]; K = h["K"]
    tr = [r for r in recs if r.get("rec") == "trial"]
    if not tr or "plantTarget" not in tr[0]: return None
    Ts = sorted({abs(r["plantTarget"]) for r in tr})
    def grp(T): return [r for r in tr if abs(r["plantTarget"]) == T]
    def medrel(T):
        gl = grp(T); ok = [r["relErrRms"] for r in gl if not r.get("decodeFail") and r.get("relErrRms") is not None]
        if len(ok) == len(gl): return st.median(ok)
        if not ok or len(ok) < len(gl) / 2: return float("inf")
        return st.median(ok)
    base_pool = [r["relErrRms"] for r in tr if abs(r["plantTarget"]) <= K - 2 and r.get("relErrRms") is not None and not r.get("decodeFail")]
    base = st.median(base_pool) if base_pool else None
    def first(cond):
        for T in Ts:
            if cond(T): return T
        return None
    s = dict(h=h, tr=tr, Ts=Ts, grp=grp, medrel=medrel, base=base, nbase=len(base_pool), F=F, N=N, K=K)
    s["thr2"] = first(lambda T: base is not None and medrel(T) > 2 * base)
    s["t3"] = first(lambda T: medrel(T) >= 1e-3); s["t2"] = first(lambda T: medrel(T) >= 1e-2); s["t1"] = first(lambda T: medrel(T) >= 1e-1)
    s["fe"] = first(lambda T: any(r.get("decodeFail") for r in grp(T)))
    ae = None
    for T in reversed(Ts):
        if all(r.get("decodeFail") for r in grp(T)): ae = T
        else: break
    s["ae"] = ae
    s["wd"] = first(lambda T: st.median([r["coefErrMedAbs"] for r in grp(T) if r.get("coefErrMedAbs") is not None] or [0]) > 1)
    s["lastNormal"] = max([T for T in Ts if medrel(T) < 1e-3 and not any(r.get("decodeFail") for r in grp(T))] or [None], key=lambda v: -1 if v is None else v)
    lo, hi = K + 6, K + 28
    ratios = [abs(r["coefErrAtArgmaxI"]) / (model_eps(abs(r["plantTarget"]), K) * 2 ** F) for r in tr
              if lo <= abs(r["plantTarget"]) <= hi and r.get("coefErrAtArgmaxI") is not None and r["argmaxI"] == h["plantIndex"]
              and model_eps(abs(r["plantTarget"]), K) > 0 and math.isfinite(model_eps(abs(r["plantTarget"]), K))]
    s["ratios"] = ratios; s["ratioWin"] = (lo, hi)
    s["nver"] = sum(1 for r in tr if r.get("plantVerified")); s["nadj"] = sum(1 for r in tr if r["adjustMatchesLibrary"])
    s["nhit"] = sum(1 for r in tr if r["plantAchieved"] == r["plantTarget"]); s["nrest"] = sum(1 for r in tr if r.get("bootRestored"))
    s["nthrewBoot"] = sum(1 for r in tr if r.get("bootThrew"))
    s["othermax"] = max(max(abs(v) for v in r["top5I"][1:]) for r in tr)
    return s


def grid_str(Ts):
    if len(Ts) < 2: return str(Ts)
    steps = sorted({b - a for a, b in zip(Ts, Ts[1:])})
    return f"{Ts[0]}..{Ts[-1]} step {steps[0] if len(steps) == 1 else steps}"


def thr_row(label, s):
    rt = s["ratios"]
    return (f"| {label} | {s['h'].get('envUniformExt') or 'unset'} | {s['K']} | {s['N']} | {s['F']} | {grid_str(s['Ts'])} | {fmt(s['base'])} (n = {s['nbase']}) | {s['thr2']} | {s['t3']} | {s['t2']} | {s['t1']} | "
            f"{s['fe']} | {s['ae']} | {s['wd']} | " + (f"{st.median(rt):.3f} ({min(rt):.3f}..{max(rt):.3f}), n = {len(rt)}" if rt else "-") + " |")


THR_HEAD = ("| sweep | OPENFHE_BOOT_UNIFORM_EXT | K (label) | N | F | I* grid | baseline relErrRms (median over |I*| <= K-2) | median relErrRms > 2x baseline | median >= 1e-3 | median >= 1e-2 | median >= 1e-1 | "
            "first decode exception | all trials throw from | whole ciphertext destroyed (median coefficient error > 1) | measured / model at j*, median over K+6 <= |I*| <= K+28 (min..max) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

# ------------------------------------------------------------------ T0. identity of every record file
P("## T0. Record files: header identity, sanity counters, what the library printed\n")
P("`arm` is read from what the LIBRARY printed on stderr (one line per process, only when a switch is on): "
  "A-ext = `[openfhe-k768-patch] EvalBootstrap: OPENFHE_BOOT_UNIFORM_EXT=1 -> Chebyshev degree 118, K = 768`; "
  "B-split = `... -> Chebyshev degree 118, K in the scalar constant = 256 (OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1: the factor 3 is in CoeffsToSlots)`; "
  "B-nosplit = `... K in the scalar constant = 768 (no split)`; off = empty stderr (stock selection). `other stderr` = anything else in the stderr file.\n")
P("| file | arm (from stderr) | OPENFHE_BOOT_UNIFORM_EXT (header) | K label | N | depth | bootDepth (library's GetBootstrapDepth) | towersQ | F | harnessSha256 | trials | adjust==library | bootRestored | EvalBootstrap threw | towersIn -> towersOut | outLevel | outNoiseScaleDeg | other stderr |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
files = sorted(glob.glob(os.path.join(HERE, "ctrl_*.jsonl")) + glob.glob(os.path.join(HERE, "ext_*.jsonl")) + glob.glob(os.path.join(HERE, "B_*.jsonl")))
for f in files:
    recs = load(f); hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; tr = [r for r in recs if r.get("rec") == "trial"]
    done = [r for r in tr if "towersOut" in r]
    _arm = {"Ext + split (variant B)": "B-split", "Ext, no split (variant B library)": "B-nosplit", "Ext (variant A)": "A-ext"}.get(arm_of(f), "off (B library)" if os.path.basename(f).startswith("B_") else "off")
    _other = [l for l in stderr_of(f).split("\n") if l and l != "(empty)" and "[openfhe-k768-patch]" not in l]
    P(f"| {os.path.basename(f)} | {_arm} | {h.get('envUniformExt') or 'unset'} | {h['K']} | {h['ringDim']} | {h['depth']} | {h['bootDepth']} | {h['towersQ']} | {h['correctionFactor']} | {h['harnessSha256']} | {len(tr)} | "
      f"{sum(1 for r in tr if r['adjustMatchesLibrary'])}/{len(tr)} | {sum(1 for r in tr if r.get('bootRestored'))}/{len(tr)} | {sum(1 for r in tr if r.get('bootThrew'))} | "
      f"{sorted({r['towersIn'] for r in tr})} -> {sorted({r['towersOut'] for r in done})} | {sorted({r['outLevel'] for r in done})} | {sorted({r['outNoiseScaleDeg'] for r in done})} | {'none' if not _other else ' / '.join(_other)} |")
P()

# ------------------------------------------------------------------ T1. control vs the earlier stock-build sweep
P("## T1. CONTROL: the patched build with the switch OFF vs the earlier sweeps on the stock build (`../planted_*.jsonl`)\n")
P(THR_HEAD)
for f in sorted(glob.glob(os.path.join(PREV, "planted_r13_F10_pos.jsonl")) + glob.glob(os.path.join(PREV, "planted_r13_F07_pos.jsonl")) + glob.glob(os.path.join(PREV, "planted_r12_F11_pos.jsonl"))):
    s = sweep_stats(f)
    if s:
        s["h"].setdefault("envUniformExt", None)
        P(thr_row("PREVIOUS (stock build) ../" + os.path.basename(f), s))
for f in sorted(glob.glob(os.path.join(HERE, "ctrl_*planted*.jsonl"))):
    s = sweep_stats(f)
    if s: P(thr_row("CONTROL (k768 build, switch off) " + os.path.basename(f), s))
P()

# per-I comparison previous vs control (same grid)
prevf = os.path.join(PREV, "planted_r13_F10_pos.jsonl"); ctrlf = os.path.join(HERE, "ctrl_r13_F10_planted.jsonl")
if os.path.exists(prevf) and os.path.exists(ctrlf):
    sp, sc = sweep_stats(prevf), sweep_stats(ctrlf)
    if sp and sc:
        P("## T1b. CONTROL vs PREVIOUS, per I* (N = 8192, F = 10, 3 trials per I* in both; different random keys and ciphertexts)\n")
        P("`exc` = decode exceptions out of 3; `floor` = median over the 3 trials of the median |coefficient error| over all N coefficients.\n")
        P("| I* | PREVIOUS stock build: relErrRms median | exc | |coef err at j*| median | floor | CONTROL k768 build, switch off: relErrRms median | exc | |coef err at j*| median | floor | model |g(I/512)| * 2^F (arithmetic) |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for T in sorted(set(sp["Ts"]) | set(sc["Ts"])):
            cells = []
            for s in (sp, sc):
                gl = s["grp"](T)
                ok = [r["relErrRms"] for r in gl if not r.get("decodeFail") and r.get("relErrRms") is not None]
                ce = [abs(r["coefErrAtArgmaxI"]) for r in gl if r.get("coefErrAtArgmaxI") is not None and r["argmaxI"] == s["h"]["plantIndex"]]
                fl = [r["coefErrMedAbs"] for r in gl if r.get("coefErrMedAbs") is not None]
                cells += [fmt(med(ok)), f"{sum(1 for r in gl if r.get('decodeFail'))}/{len(gl)}", fmt(med(ce)), fmt(med(fl))]
            if T < 510 and T % 5: continue
            P(f"| {T} | " + " | ".join(cells) + f" | {fmt(model_eps(T, 512) * 2 ** sc['F'])} |")
        P()

# ------------------------------------------------------------------ T2. baselines without planting
P("## T2. Baseline precision WITHOUT planting: stock table (switch off) vs Ext table (switch on), same binary, same trial count\n")
P("| file | OPENFHE_BOOT_UNIFORM_EXT | K label | N | F | trials | decode fails | max|I| over all trials | sigma_I measured (mean) | relErrRms median | relErrRms min..max | relErrMax median | coefErrRms median | floor = median |coef err| (median over trials) | max |coef err| (median over trials) | max/floor (median) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
basefiles = sorted(glob.glob(os.path.join(HERE, "*_base.jsonl")))
prev_base = os.path.join(PREV, "sound_r13_stock.jsonl")
bstats = {}
for f in ([prev_base] if os.path.exists(prev_base) else []) + basefiles:
    recs = load(f); hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; tr = [r for r in recs if r.get("rec") == "trial"]
    ok = [r for r in tr if not r.get("decodeFail") and r.get("relErrRms") is not None]
    if not ok: continue
    name = ("PREVIOUS (stock build) ../" if f == prev_base else "") + os.path.basename(f)
    bstats[os.path.basename(f)] = dict(rel=st.median(r["relErrRms"] for r in ok), floor=st.median(r["coefErrMedAbs"] for r in ok), rms=st.median(r["coefErrRms"] for r in ok), h=h, n=len(ok))
    P(f"| {name} | {h.get('envUniformExt') or 'unset'} | {h['K']} | {h['ringDim']} | {h['correctionFactor']} | {len(tr)} | {sum(1 for r in tr if r.get('decodeFail'))} | {max(r['maxAbsI'] for r in tr)} | "
      f"{st.mean(r['sigmaIhat'] for r in tr):.3f} | {fmt(st.median(r['relErrRms'] for r in ok))} | {fmt(min(r['relErrRms'] for r in ok))}..{fmt(max(r['relErrRms'] for r in ok))} | {fmt(st.median(r['relErrMax'] for r in ok))} | "
      f"{fmt(st.median(r['coefErrRms'] for r in ok))} | {fmt(st.median(r['coefErrMedAbs'] for r in ok))} | {fmt(st.median(r['coefErrMax'] for r in ok))} | {st.median(r['coefErrMax'] / r['coefErrMedAbs'] for r in ok):.1f} |")
P()
P("Ratios to the control of the same ring and F (medians of the table above). `ext_*` / `ctrl_*` pairs share binary (library A) and trial count (40); the `B_*` rows come from "
  "the variant-B binary (10 trials for `B_nosplit` and `B_ctrl`) and are compared with library A's control:\n")
P("| pair | relErrRms Ext / control | bits of precision lost = log2 of that | coefErrRms Ext / control | floor Ext / control |")
P("|---|---|---|---|---|")
for c in sorted(k for k in bstats if k.startswith("ctrl_")):
    tag = c[len("ctrl_"):]
    for e in ("ext_" + tag, "B_split_" + tag, "B_nosplit_" + tag, "B_ctrl_" + tag):
        if e in bstats:
            r = bstats[e]["rel"] / bstats[c]["rel"]
            P(f"| {e} / {c} | {r:.3f} | {math.log2(r):+.2f} | {bstats[e]['rms'] / bstats[c]['rms']:.3f} | {bstats[e]['floor'] / bstats[c]['floor']:.3f} |")
P()

# ------------------------------------------------------------------ T2b. error vs the coefficient's own overflow
P("## T2b. Un-planted trials: decode-free coefficient error as a function of the coefficient's own overflow |I_j| (`errByAbsI` records, pooled over the trials of each file)\n")
groups = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 24), (25, 32), (33, 48), (49, 64), (65, 96), (97, 10 ** 9)]
def pooled(f):
    acc = {gi: [0, 0.0, 0.0] for gi in range(len(groups))}
    for r in load(f):
        if r.get("rec") != "errByAbsI": continue
        W = r["binWidth"]
        for b, n, ssq, mx in r["bins"]:
            lo = b * W   # with W = 1 the bin is one value of |I|
            for gi, (a, z) in enumerate(groups):
                if a <= lo <= z:
                    acc[gi][0] += n; acc[gi][1] += ssq; acc[gi][2] = max(acc[gi][2], mx); break
    return acc
pb = {os.path.basename(f): pooled(f) for f in basefiles}
for c, e in [(c, pre + c[len("ctrl_"):]) for c in sorted(k for k in pb if k.startswith("ctrl_")) for pre in ("ext_", "B_split_")]:
    if e not in pb: continue
    P(f"### {c} (stock table) vs {e} ({'Ext table, variant A' if e.startswith('ext_') else 'Ext table + split, variant B'})\n")
    P("| |I_j| | control: coefficients | control: rms coef error | control: max |coef error| | Ext: coefficients | Ext: rms coef error | Ext: max |coef error| | rms Ext / control |")
    P("|---|---|---|---|---|---|---|---|")
    for gi, (a, z) in enumerate(groups):
        nc, sc_, mc = pb[c][gi]; ne, se, me = pb[e][gi]
        if nc == 0 and ne == 0: continue
        rc = math.sqrt(sc_ / nc) if nc else None; re_ = math.sqrt(se / ne) if ne else None
        lab = f"{a}" if a == z else (f"{a}..{z}" if z < 10 ** 9 else f">= {a}")
        P(f"| {lab} | {nc} | {fmt(rc)} | {fmt(mc) if nc else '-'} | {ne} | {fmt(re_)} | {fmt(me) if ne else '-'} | " + (f"{re_ / rc:.3f}" if rc and re_ else "-") + " |")
    P()
# where do the largest coefficient errors sit?
P("## T2c. Un-planted trials: where the largest coefficient errors sit, their sign, and the shape of the peak at small |I_j|\n")
P("Overflow I_j of the 6 coefficients with the largest error in each un-planted trial (`topCoefErrI`), pooled per file:\n")
P("| file | trials | top-6 entries | entries with |I_j| = 0 | <= 2 | <= 8 | > 8 | median |I_j| of the top-6 | for reference: fraction of ALL coefficients with |I_j| = 0 / <= 2 / <= 8 (from errByAbsI) |")
P("|---|---|---|---|---|---|---|---|---|")
for f in basefiles:
    recs = load(f); tr = [r for r in recs if r.get("rec") == "trial" and "topCoefErrI" in r]
    if not tr: continue
    allI = [abs(v) for r in tr for v in r["topCoefErrI"]]
    acc = pb[os.path.basename(f)]; tot = sum(v[0] for v in acc.values())
    f0 = acc[0][0] / tot; f2 = (acc[0][0] + acc[1][0] + acc[2][0]) / tot; f8 = sum(acc[i][0] for i in range(5)) / tot
    P(f"| {os.path.basename(f)} | {len(tr)} | {len(allI)} | {sum(1 for v in allI if v == 0)} | {sum(1 for v in allI if v <= 2)} | {sum(1 for v in allI if v <= 8)} | {sum(1 for v in allI if v > 8)} | {st.median(allI)} | {f0:.4f} / {f2:.4f} / {f8:.4f} |")
P()

P("Sign of the entries of `topCoefErr` that sit on a coefficient with I_j = 0 (a bias would have one sign):\n")
P("| file | entries at I_j = 0 | positive | negative |")
P("|---|---|---|---|")
for f in basefiles:
    ent = [e for r in load(f) if r.get("rec") == "trial" and "topCoefErrI" in r for (idx, e), I in zip(r["topCoefErr"], r["topCoefErrI"]) if I == 0]
    if ent: P(f"| {os.path.basename(f)} | {len(ent)} | {sum(1 for e in ent if e > 0)} | {sum(1 for e in ent if e < 0)} |")
P()
P("Shape of the peak at small |I_j|: excess of the rms error over the |I_j| = 5..8 bin of the same file, in quadrature, relative to the excess at I_j = 0; "
  "next to the ARITHMETIC ratio of the double-angle amplification A(I) = 1 / (2^R |sin(2 pi (I - 1/4) / 2^R)|), R = 6, rms over +I and -I "
  "(noise injected before the double-angle iterations is multiplied by prod_i 4 a_i |cos(2^i theta_0)| = const * A(I)):\n")
P("| file | excess at 0 | excess at |I| = 1 (ratio to 0) | excess at |I| = 2 (ratio to 0) | arithmetic A(1)/A(0) | arithmetic A(2)/A(0) |")
P("|---|---|---|---|---|---|")
def _A(I): return 1.0 / (2 ** R * abs(math.sin(2 * math.pi * (I - 0.25) / 2 ** R)))
def _Arms(I): return math.sqrt((_A(I) ** 2 + _A(-I) ** 2) / 2)
for f in basefiles:
    acc = pb[os.path.basename(f)]
    if not acc[4][0]: continue
    ref2 = acc[4][1] / acc[4][0]
    ex = []
    for gi in (0, 1, 2):
        v = acc[gi][1] / acc[gi][0] - ref2 if acc[gi][0] else None
        ex.append(math.sqrt(v) if v and v > 0 else None)
    if ex[0] is None: P(f"| {os.path.basename(f)} | none (rms at 0 not above the 5..8 bin) | - | - | {_Arms(1) / _A(0):.3f} | {_Arms(2) / _A(0):.3f} |"); continue
    P(f"| {os.path.basename(f)} | {fmt(ex[0])} | " + (f"{fmt(ex[1])} ({ex[1] / ex[0]:.3f})" if ex[1] else "-") + " | " + (f"{fmt(ex[2])} ({ex[2] / ex[0]:.3f})" if ex[2] else "-")
      + f" | {_Arms(1) / _A(0):.3f} | {_Arms(2) / _A(0):.3f} |")
P()

# ------------------------------------------------------------------ T3. Ext planted sweeps: thresholds
P("## T3. EXT arm, planted sweeps: thresholds (each entry is the smallest |I*| of that sweep at which the condition first holds; `None` = never on that grid)\n")
P(THR_HEAD)
extsweeps = sorted(glob.glob(os.path.join(HERE, "ext_*planted*.jsonl")) + glob.glob(os.path.join(HERE, "B_*planted*.jsonl")))
SW = {}
for f in extsweeps:
    s = sweep_stats(f)
    if s:
        SW[os.path.basename(f)] = s
        P(thr_row(os.path.basename(f) + " [" + arm_of(f) + "]", s))
P()
P("Sanity of the planted sweeps (every file of this directory):\n")
P("| file | bootstraps | plantVerified | achieved == target | adjust==library | bootRestored | EvalBootstrap threw | largest |I| of any NON-planted coefficient |")
P("|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(HERE, "*planted*.jsonl"))):
    s = sweep_stats(f)
    if s: P(f"| {os.path.basename(f)} | {len(s['tr'])} | {s['nver']}/{len(s['tr'])} | {s['nhit']}/{len(s['tr'])} | {s['nadj']}/{len(s['tr'])} | {s['nrest']}/{len(s['tr'])} | {s['nthrewBoot']} | {s['othermax']} |")
P()

# ------------------------------------------------------------------ T4. per-I tables of the Ext sweeps
def per_I(f, s, every=1, note="", lo=None, hi=None):
    h = s["h"]; F = s["F"]; K = s["K"]; N = s["N"]
    P(f"### {os.path.basename(f)} [{arm_of(f)}] -- N = {N}, F = {F}, K label {K}, OPENFHE_BOOT_UNIFORM_EXT = {h.get('envUniformExt') or 'unset'}, plant index {h['plantIndex']}, {len(s['tr'])} bootstraps{note}\n")
    P("| I* | n | decode exceptions | relErrRms (min / median / max over non-throwing) | share of squared coef error on {j*, j*+N/2} (median) | |coef err at j*| (min / median / max) | |coef err at j*+N/2| median | floor | model |g(I/K)|*2^F (arithmetic) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for T in sorted({r["plantTarget"] for r in s["tr"]}, key=lambda v: abs(v)):
        if every > 1 and abs(T) % every: continue
        if (lo is not None and abs(T) < lo) or (hi is not None and abs(T) > hi): continue
        g_ = [r for r in s["tr"] if r["plantTarget"] == T]
        ok = [r for r in g_ if not r.get("decodeFail") and r.get("relErrRms") is not None]
        ce = [abs(r["coefErrAtArgmaxI"]) for r in g_ if r.get("coefErrAtArgmaxI") is not None and r["argmaxI"] == h["plantIndex"]]
        pe = [abs(r["coefErrAtPartner"]) for r in g_ if r.get("coefErrAtPartner") is not None and r["argmaxI"] == h["plantIndex"]]
        fl = [r["coefErrMedAbs"] for r in g_ if r.get("coefErrMedAbs") is not None]
        rel = f"{fmt(min(r['relErrRms'] for r in ok))} / {fmt(st.median(r['relErrRms'] for r in ok))} / {fmt(max(r['relErrRms'] for r in ok))}" if ok else "-"
        cej = f"{fmt(min(ce))} / {fmt(st.median(ce))} / {fmt(max(ce))}" if ce else "-"
        mm = [(r["coefErrAtArgmaxI"] ** 2 + r["coefErrAtPartner"] ** 2) / (N * r["coefErrRms"] ** 2) for r in g_
              if r.get("coefErrRms") and r.get("coefErrAtArgmaxI") is not None and r["argmaxI"] == h["plantIndex"] and math.isfinite(r["coefErrRms"] ** 2)]
        P(f"| {T} | {len(g_)} | {sum(1 for r in g_ if r.get('decodeFail'))} | {rel} | " + (f"{st.median(mm):.4f}" if mm else "-") + f" | {cej} | {fmt(med(pe))} | {fmt(med(fl))} | {fmt(model_eps(abs(T), K) * 2 ** F)} |")
    P()

P("## T4e. The NEW edge under the Ext table, per I* (rows 766..806 of the edge sweeps; the full sweeps are in T4)\n")
P("`share` = (e_j*^2 + e_(j*+N/2)^2) / sum_j e_j^2 over the decode-free coefficient errors. `floor` = median |coefficient error| over all N coefficients (median over the trials of the row). "
  "`model` = the Ext polynomial re-evaluated in float64 at I*/768, times 2^F (ARITHMETIC).\n")
for f in sorted(glob.glob(os.path.join(HERE, "ext_*planted_edge.jsonl"))):
    s_ = sweep_stats(f)
    if s_: per_I(f, s_, lo=766, hi=806)

P("## T4. Per-I* tables\n")
P("`share` = (e_j*^2 + e_(j*+N/2)^2) / sum_j e_j^2 over the decode-free coefficient errors (1.0 = all the damage sits on the planted coefficient and the other half of its slot). "
  "`floor` = median |coefficient error| over all N coefficients (median over the trials of the row). `model` = the library's polynomial for that arm re-evaluated in float64 at I*/K, times 2^F.\n")
for f in sorted(glob.glob(os.path.join(HERE, "ctrl_*planted*.jsonl"))) + extsweeps:
    s = sweep_stats(f)
    if s: per_I(f, s)

# ------------------------------------------------------------------ T5. old failure zone under Ext, summarised
P("## T5. Ext-arm planted sweeps that stay INSIDE the fitted range (|I*| <= 760), which includes the whole OLD failure zone; summarised from T4\n")
P("| file | I* range | bootstraps | decode exceptions | relErrRms: median of per-I* medians (min..max over ALL trials) | the same file's own baseline (|I*| <= 510) | largest per-I* median / baseline | |coef err at j*|: median (max) over all trials | floor: median |")
P("|---|---|---|---|---|---|---|---|---|")
for name, s in SW.items():
    if s["Ts"][-1] > 760 or s["Ts"][0] < 0: continue
    tr = s["tr"]; ok = [r for r in tr if not r.get("decodeFail") and r.get("relErrRms") is not None]
    meds = [s["medrel"](T) for T in s["Ts"]]
    b = [r["relErrRms"] for r in ok if abs(r["plantTarget"]) <= 510]
    base = st.median(b) if b else None
    ce = [abs(r["coefErrAtArgmaxI"]) for r in ok if r["argmaxI"] == s["h"]["plantIndex"]]
    P(f"| {name} | {s['Ts'][0]}..{s['Ts'][-1]} | {len(tr)} | {sum(1 for r in tr if r.get('decodeFail'))} | {fmt(st.median(meds))} ({fmt(min(r['relErrRms'] for r in ok))}..{fmt(max(r['relErrRms'] for r in ok))}) | "
      f"{fmt(base)} | " + (f"{max(meds) / base:.2f}" if base else "-") + f" | {fmt(st.median(ce))} ({fmt(max(ce))}) | {fmt(st.median(r['coefErrMedAbs'] for r in ok))} |")
P()

P("## T5b. Every planted bootstrap with 500 <= |I*| <= 766, pooled per arm, ring and F (all planted files of this directory)\n")
P("| arm | N | F | files | bootstraps | decode exceptions | EvalBootstrap threw | relErrRms min / median / max | sigma_I of the planted ciphertexts (mean of `sigmaIhat`) | for comparison: the un-planted baseline of the same arm, ring and F (median, from T2; its sigma_I is in T2) |")
P("|---|---|---|---|---|---|---|---|---|---|")
poolz = {}
for f in sorted(glob.glob(os.path.join(HERE, "*planted*.jsonl"))):
    recs = load(f); hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; key = (arm_of(f), h["ringDim"], h["correctionFactor"])
    for r in recs:
        if r.get("rec") == "trial" and 500 <= abs(r["plantTarget"]) <= 766:
            poolz.setdefault(key, []).append((r, os.path.basename(f)))
def _base_for(arm, N, F):
    for name, b in bstats.items():
        hh = b["h"]
        if hh["ringDim"] == N and hh["correctionFactor"] == F and name.startswith(("ctrl_", "ext_", "B_")):
            a = arm_of(os.path.join(HERE, name))
            if a == arm and b.get("n", 40) >= 20: return b["rel"]
    return None
for key in sorted(poolz):
    arm, N, F = key; v = poolz[key]
    ok = [r["relErrRms"] for r, _ in v if not r.get("decodeFail") and r.get("relErrRms") is not None]
    P(f"| {arm} | {N} | {F} | {len({n for _, n in v})} | {len(v)} | {sum(1 for r, _ in v if r.get('decodeFail'))} | {sum(1 for r, _ in v if r.get('bootThrew'))} | "
      + (f"{fmt(min(ok))} / {fmt(st.median(ok))} / {fmt(max(ok))}" if ok else "-") + f" | {st.mean(r['sigmaIhat'] for r, _ in v):.2f} | {fmt(_base_for(arm, N, F))} |")
P()

# ------------------------------------------------------------------ T6. error at the planted coefficient INSIDE the fitted range, both arms
P("## T6. Inside the fitted range: decode-free error AT the planted coefficient j* as a function of its overflow I* (all planted files, trials where the planted coefficient is the arg max |I|)\n")
P("Pooled over I* windows; `ratio to floor` = median over trials of |coef err at j*| / (that trial's median |coefficient error|). Only trials with |I*| <= K - 2 of the file's arm are used (the edge is in T3/T4).\n")
P("| arm | file(s) | N | F | I* window | trials | |coef err at j*|: median | rms | max | floor (median over trials) | ratio to floor: median | fraction of trials with |coef err at j*| > 5x floor |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
wins = [(100, 199), (200, 299), (300, 399), (400, 510), (511, 599), (600, 699), (700, 766)]
pool = {}
for f in sorted(glob.glob(os.path.join(HERE, "*planted*.jsonl"))):
    recs = load(f); hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; key = (arm_of(f), h["ringDim"], h["correctionFactor"], h["K"])
    for r in recs:
        if r.get("rec") != "trial" or r.get("coefErrAtArgmaxI") is None or r["argmaxI"] != h["plantIndex"]: continue
        if abs(r["plantTarget"]) > h["K"] - 2 or r.get("coefErrMedAbs") is None: continue
        pool.setdefault(key, []).append((abs(r["plantTarget"]), abs(r["coefErrAtArgmaxI"]), r["coefErrMedAbs"], os.path.basename(f)))
for key in sorted(pool):
    arm, N, F, K = key
    for lo, hi in wins:
        v = [x for x in pool[key] if lo <= x[0] <= hi]
        if not v: continue
        e = [x[1] for x in v]; fl = [x[2] for x in v]; rat = [x[1] / x[2] for x in v]
        files_ = f"{len({x[3] for x in v})} file(s)"
        P(f"| {arm} | {files_} | {N} | {F} | {lo}..{hi} | {len(v)} | {fmt(st.median(e))} | {fmt(math.sqrt(sum(t * t for t in e) / len(e)))} | {fmt(max(e))} | {fmt(st.median(fl))} | {st.median(rat):.2f} | {sum(1 for t in rat if t > 5) / len(rat):.3f} |")
P()

# ------------------------------------------------------------------ T7. the error that is LINEAR in the overflow
from fractions import Fraction
def _is_prime(n):
    if n < 2: return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0: return n == q
    d, r = n - 1, 0
    while d % 2 == 0: d //= 2; r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1: break
        else: return False
    return True
def eta_pred(Kscalar, N, p=59, deg=1):   # same arithmetic as k768_const_rounding.py
    m = 2 * N; q = 1 << p; r = q % m; SF = q + 1 - r + (m if r > 0 else 0)
    while not _is_prime(SF): SF += m
    cSF = Fraction(SF, (1 << deg) * Kscalar * N)
    return float((math.floor(cSF + Fraction(1, 2)) - cSF) / cSF)
P("## T7. Inside the fitted range the error at the planted coefficient is LINEAR in its overflow: measured slope vs the rounding of the scalar constant (arithmetic)\n")
P("Least-squares line through the origin, signed `coefErrAtArgmaxI` = slope * I*, over the trials with 100 <= |I*| <= K - 2 whose planted coefficient is the arg max |I|; "
  "`eta measured` = slope / 2^F. `eta predicted` = (round(c*SF) - c*SF)/(c*SF) for the scalar constant c = pre/(K_scalar*N) the arm multiplies the raised ciphertext by "
  "(K_scalar = 512 stock, 768 Ext, 256 Ext + split), SF = FirstPrime(59, 2N): ARITHMETIC on the library's rules (`k768_const_rounding.py`), not a measurement. "
  "`residual rms` = rms of (error - slope * I*), to be compared with the floor.\n")
P("| arm | file | N | F | trials | I* range | same sign as I* * eta_pred | eta measured = slope / 2^F | standard error | eta predicted (arithmetic) | measured / predicted | residual rms | floor (median) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(HERE, "*planted*.jsonl"))):
    recs = load(f); hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; arm = arm_of(f); F = h["correctionFactor"]; N = h["ringDim"]
    pts = [(r["plantTarget"], r["coefErrAtArgmaxI"], r["coefErrMedAbs"]) for r in recs if r.get("rec") == "trial" and r.get("coefErrAtArgmaxI") is not None
           and r["argmaxI"] == h["plantIndex"] and 100 <= abs(r["plantTarget"]) <= h["K"] - 2 and not r.get("decodeFail")]
    if len(pts) < 6: continue
    sxx = sum(x * x for x, _, _ in pts); sxy = sum(x * y for x, y, _ in pts)
    slope = sxy / sxx; res = [y - slope * x for x, y, _ in pts]
    rr = math.sqrt(sum(t * t for t in res) / len(res)); se = math.sqrt(sum(t * t for t in res) / (len(res) - 1) / sxx)
    Ks = 256 if "split (variant B)" in arm and "no split" not in arm else h["K"]
    ep = eta_pred(Ks, N)
    same = sum(1 for x, y, _ in pts if (x * ep > 0) == (y > 0))
    P(f"| {arm} | {os.path.basename(f)} | {N} | {F} | {len(pts)} | {min(x for x, _, _ in pts)}..{max(x for x, _, _ in pts)} | {same}/{len(pts)} | {slope / 2 ** F:+.4e} | {se / 2 ** F:.1e} | {ep:+.4e} | "
      f"{(slope / 2 ** F) / ep:.3f} | {fmt(rr)} | {fmt(st.median(z for _, _, z in pts))} |")
P()

P("Pooled over the files of one arm, ring and F (same selection of trials):\n")
P("| arm | N | F | trials | same sign as I* * eta_pred | eta measured | standard error | eta predicted (arithmetic) | measured / predicted |")
P("|---|---|---|---|---|---|---|---|---|")
pp = {}
for f in sorted(glob.glob(os.path.join(HERE, "*planted*.jsonl"))):
    recs = load(f); hs = [r for r in recs if r.get("rec") == "header"]
    if not hs: continue
    h = hs[0]; arm = arm_of(f)
    for r in recs:
        if r.get("rec") == "trial" and r.get("coefErrAtArgmaxI") is not None and r["argmaxI"] == h["plantIndex"] and 100 <= abs(r["plantTarget"]) <= h["K"] - 2 and not r.get("decodeFail"):
            pp.setdefault((arm, h["ringDim"], h["correctionFactor"], h["K"]), []).append((r["plantTarget"], r["coefErrAtArgmaxI"]))
for key in sorted(pp):
    arm, N, F, K = key; pts = pp[key]
    if len(pts) < 6: continue
    sxx = sum(x * x for x, _ in pts); slope = sum(x * y for x, y in pts) / sxx
    res = [y - slope * x for x, y in pts]; se = math.sqrt(sum(t * t for t in res) / (len(res) - 1) / sxx)
    ep = eta_pred(256 if arm == "Ext + split (variant B)" else K, N)
    P(f"| {arm} | {N} | {F} | {len(pts)} | {sum(1 for x, y in pts if (x * ep > 0) == (y > 0))}/{len(pts)} | {slope / 2 ** F:+.4e} | {se / 2 ** F:.1e} | {ep:+.4e} | {(slope / 2 ** F) / ep:.3f} |")
P()

# ------------------------------------------------------------------ T8. ARITHMETIC: the measured Ext thresholds priced at N = 2^17
P("## T8. ARITHMETIC (not a measurement): Gaussian-tail probability of reaching the MEASURED thresholds at N = 2^17, uniform ternary secret\n")
P("P(max_j |I_j| >= B) = 1 - (1 - 2 Q((B - 0.5) / sigma_I))^N with sigma_I = sqrt((2N/3 + 1)/12) = 85.33, as in `../demo_rate_from_thresholds.py`; thresholds B are the ones measured "
  "on CPU at N = 2^13 (T1 / T3), NOT at 2^17, and the tail law itself is the extrapolation discussed in `../PROBE_C_REPORT.md`.\n")
def Qf(x): return 0.5 * math.erfc(x / math.sqrt(2))
N17 = 1 << 17; sig17 = math.sqrt((2 * N17 / 3 + 1) / 12)
P("| sweep (arm) | F of the sweep | threshold | B | B / sigma_I | P per bootstrap | P per 480-bootstrap tick |")
P("|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(HERE, "ctrl_r13_F10_planted.jsonl")) + glob.glob(os.path.join(HERE, "ext_r13_*planted_edge.jsonl"))):
    s_ = sweep_stats(f)
    if not s_: continue
    for lab, B in (("median relErrRms > 2x baseline", s_["thr2"]), ("median relErrRms >= 0.1", s_["t1"]), ("first decode exception", s_["fe"]), ("whole ciphertext destroyed", s_["wd"])):
        if B is None: continue
        p1 = 2 * Qf((B - 0.5) / sig17); pb = -math.expm1(N17 * math.log1p(-p1)); pt = -math.expm1(480 * math.log1p(-pb))
        P(f"| {os.path.basename(f)} [{arm_of(f)}] | {s_['F']} | {lab} | {B} | {B / sig17:.2f} | {pb:.2e} | {pt:.2e} |")
P()

open(os.path.join(HERE, "k768_tables.md"), "w").write("\n".join(out) + "\n")
print("wrote k768_tables.md,", len(out), "lines")
