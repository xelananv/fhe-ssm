#!/usr/bin/env python3
"""Publication figures, read from the run records only.

Every plotted number is read from a named record file; the script prints each one with its record path.
Run from anywhere:  python3 figures/make_figures.py --out figures
Writes <name>.png (200 dpi) and <name>.svg per figure, and figures/sources.json (sha256 of every record read).

Timing rule: wall-clock is a property of the build and the machine. No figure divides a timing from one build by a
timing from another. The only ratios drawn are inside one session or one build (prefill vs decode in one session;
32 conversations vs 1 conversation of the same model on the same build).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics as st
import textwrap
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

LR = "results/dense-demo-s31/pod_longrun_20260919"
BP = "results/perseus-20261006/fork_port/gpu_20261011/broadcast_port_20261030"
NP = "results/rwkv7-fhe-20261104/new_pod"
OLD = "results/rwkv7-fhe-20261104/old_pod_mirror"

# ---------------------------------------------------------------- style (restrained, colorblind-checked palette)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # blue, orange, aqua: validated all-pairs for CVD
WIDTH = 7.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "text.color": INK,
    "lines.linewidth": 1.4,
    "svg.hashsalt": "fhe-ssm-figures",
})

READ = {}  # relative path -> sha256


def P(rel):
    path = os.path.join(ROOT, rel)
    if rel not in READ:
        with open(path, "rb") as f:
            READ[rel] = hashlib.sha256(f.read()).hexdigest()
    return path


def say(fig, label, value, rel):
    print(f"[{fig}] {label} = {value}    <- {rel}")


def jsonl(rel):
    with open(P(rel)) as f:
        return [json.loads(line) for line in f if line.strip()]


def jload(rel):
    with open(P(rel)) as f:
        return json.load(f)


def source_line(fig, lines, width=165):
    text = " ".join(lines)
    wrapped = textwrap.wrap(text, width=width, break_on_hyphens=False)
    fig.text(0.01, 0.008, "\n".join(wrapped), ha="left", va="bottom", fontsize=6.3, color=MUTED, linespacing=1.35)


def save(fig, out, name):
    for ext in ("png", "svg"):
        kw = {"dpi": 200} if ext == "png" else {"metadata": {"Date": None, "Creator": None, "Format": None, "Type": None}}
        fig.savefig(os.path.join(out, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print(f"wrote {name}.png, {name}.svg")


def block_medians(xs, ys, size=10):
    """Median of the values in each block [b0, b0+size) of the x index (the last block may be partial: its centre is
    the mean x of its members); returns (centres, medians)."""
    by = {}
    for x, y in zip(xs, ys):
        by.setdefault(int(x) // size, []).append((x, y))
    cs, ms = [], []
    for b in sorted(by):
        cs.append(st.mean(x for x, _ in by[b]))
        ms.append(st.median(y for _, y in by[b]))
    return cs, ms


# ---------------------------------------------------------------- record readers
def read_0919():
    rel = f"{LR}/plots_data/LONG_per_tick.csv"
    with open(P(rel)) as f:
        rows = list(csv.DictReader(f))
    tick = [int(r["tick"]) for r in rows]
    wall = [float(r["wall_layer_loop_s"]) for r in rows]
    err = [float(r["relerr_rms_median"]) for r in rows]
    return rel, tick, wall, err


def read_1102_free():
    rel = f"{BP}/gpu_20261102_148/run_best_free148/record.json"
    r = jload(rel)
    t = r["ticks"]
    return rel, r, t


def read_rwkv_free():
    rel = f"{NP}/f2_01b/run_best_free148.jsonl"
    rows = [x for x in jsonl(rel) if x.get("kind") == "token"]
    return rel, rows


# ---------------------------------------------------------------- figure 1
def fig_cost(out):
    F = "cost_per_step"
    rel_a, tick_a, wall_a, _ = read_0919()
    rel_b, rec_b, ticks_b = read_1102_free()
    rel_c, rows_c = read_rwkv_free()
    xb = [t["pos"] for t in ticks_b if t["fenced"] is False]
    yb = [t["seconds"] for t in ticks_b if t["fenced"] is False]
    xc = [r["pos"] for r in rows_c]
    yc = [r["seconds"] for r in rows_c]
    ma, mb, mc = st.median(wall_a), st.median(yb), st.median(yc)
    say(F, "(a) 2026-09-19 ticks", f"{len(wall_a)} (tick {tick_a[0]}..{tick_a[-1]})", rel_a)
    say(F, "(a) wall_layer_loop_s median, all ticks", f"{ma:.3f} s", rel_a)
    say(F, "(a) median, warm ticks (tick >= 1)", f"{st.median(wall_a[1:]):.3f} s", rel_a)
    say(F, "(a) tick 0 (cold)", f"{wall_a[0]:.3f} s", rel_a)
    say(F, "(a) min/max warm", f"{min(wall_a[1:]):.3f} / {max(wall_a[1:]):.3f} s", rel_a)
    say(F, "(b) 2026-11-02 unfenced ticks", f"{len(yb)} of {len(ticks_b)}", rel_b)
    say(F, "(b) seconds median / mean / min / max (unfenced)",
        f"{mb:.3f} / {st.mean(yb):.3f} / {min(yb):.3f} / {max(yb):.3f} s", rel_b)
    say(F, "(b) build commit / keys", f"{rec_b['build_id']['commit']} / {rec_b['keys']['mode']}", rel_b)
    say(F, "(c) RWKV-7 0.1B token rows", f"{len(yc)} (pos {xc[0]}..{xc[-1]})", rel_c)
    say(F, "(c) seconds median / min / max", f"{mc:.4f} / {min(yc):.4f} / {max(yc):.4f} s", rel_c)
    for i in range(0, len(yc), 20):
        w = yc[i:i + 20]
        say(F, f"(c) median of positions {xc[i]}-{xc[i + len(w) - 1]}", f"{st.median(w):.3f} s", rel_c)

    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    fig.subplots_adjust(left=0.09, right=0.82, top=0.90, bottom=0.27)
    ax.plot(tick_a, wall_a, color=C1, lw=1.4, label="pbd430a, 64 conversations\nFIDESlib, ring 2^17, 2 cards, 2026-09-19")
    ax.plot(xb, yb, color=C2, lw=1.4, marker="o", ms=2.2, label="pbd430a, 64 conversations\nPerseus, ring 2^16, 1 card, 2026-11-02")
    ax.plot(xc, yc, color=C3, lw=1.4, label="RWKV-7 0.1B, 1 conversation\nPerseus, ring 2^16, 1 card, 2026-11-05")
    ax.set_yscale("log")
    ax.set_ylim(5, 400)
    ax.set_xlim(-2, 158)
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("step (token position in the free run)")
    ax.set_ylabel("wall clock per step (s, log scale)")
    ax.set_title("Per-step wall clock over a 148-step free run: flat in the step index")
    for m, dy, txt in ((ma, 0, f"median {ma:.1f} s"), (mb, -7, f"median {mb:.2f} s"),
                       (mc, 7, f"median {mc:.2f} s")):
        ax.annotate(txt, xy=(158, m), xytext=(6, dy), textcoords="offset points", va="center", fontsize=8,
                    color=INK, annotation_clip=False)
    ax.annotate(f"cold first tick {wall_a[0]:.0f} s", xy=(tick_a[0], wall_a[0]), xytext=(12, 300), fontsize=7.5,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=7.2, handlelength=1.6)
    source_line(fig, [
        f"Sources: {rel_a} (wall_layer_loop_s); {rel_b} (ticks[].seconds, fenced == false);",
        f"{rel_c} (kind token, seconds). Three builds on three hosts (same card model for the two 1-card runs): the curves are descriptors side by side, not a ratio.",
    ])
    save(fig, out, F)


# ---------------------------------------------------------------- figure 2
def fig_error(out):
    F = "error_per_step"
    rel_a, tick_a, _, err_a = read_0919()
    rel_b, rec_b, ticks_b = read_1102_free()
    rel_c, rows_c = read_rwkv_free()
    xb = [t["pos"] for t in ticks_b]
    rb = [t["relErrRms_vs_exact"] for t in ticks_b]
    lb = [t["logits_relErrRms_vs_exact"] for t in ticks_b]
    xc = [r["pos"] for r in rows_c]
    lc = [r["logits_relErrRms"] for r in rows_c]

    # (a) the power law, refitted from the per-tick medians the way the run's own trend script fits it:
    # mean of per-tick medians per full 10-tick block, least squares of ln(level) on ln(block centre b0 + 5).
    cen, lev = [], []
    for b0 in range(0, max(tick_a) + 1, 10):
        v = [e for t, e in zip(tick_a, err_a) if b0 <= t < b0 + 10]
        if len(v) == 10:
            cen.append(b0 + 5)
            lev.append(sum(v) / len(v))
    X, Y = np.log(cen), np.log(lev)
    sl, ic = np.polyfit(X, Y, 1)
    amp = math.exp(ic)
    say(F, "(a) block means of per-tick median (first, last, max)",
        f"{lev[0]:.6f}, {lev[-1]:.6f}, {max(lev):.6f} over {len(lev)} blocks", rel_a)
    say(F, "(a) power law refitted through block centres", f"{amp:.6f} * tick^{sl:.3f}", rel_a)
    ca, ma = block_medians(tick_a, err_a)
    say(F, "(a) 10-tick block medians (last block 140-147 partial)", ", ".join(f"{v:.6f}" for v in ma), rel_a)

    # (b) saturating fit c - d exp(-k/60) to the reply series, tau fixed at 60, linear least squares in (c, d)
    k = np.array(xb, float)
    A = np.vstack([np.ones_like(k), -np.exp(-k / 60.0)]).T
    (cc, dd), *_ = np.linalg.lstsq(A, np.array(rb), rcond=None)
    say(F, "(b) reply relErrRms_vs_exact first / last / first-10 mean / last-10 mean",
        f"{rb[0]:.6f} / {rb[-1]:.6f} / {st.mean(rb[:10]):.6f} / {st.mean(rb[-10:]):.6f}", rel_b)
    say(F, "(b) saturating fit c - d*exp(-k/60), k = pos", f"c = {cc:.6f} (plateau), d = {dd:.6f}", rel_b)
    say(F, "(b) logits_relErrRms_vs_exact first-10 mean / last-10 mean",
        f"{st.mean(lb[:10]):.6f} / {st.mean(lb[-10:]):.6f}", rel_b)
    cb, mlb = block_medians(xb, lb)
    _, mrb = block_medians(xb, rb)
    say(F, "(b) 10-tick block medians of logits (last block 140-147 partial)", ", ".join(f"{v:.5f}" for v in mlb), rel_b)
    say(F, "(b) 10-tick block medians of reply (last block 140-147 partial)", ", ".join(f"{v:.5f}" for v in mrb), rel_b)
    say(F, "(c) RWKV-7 0.1B logits_relErrRms median / max",
        f"{st.median(lc):.5f} / {max(lc):.5f} over {len(lc)} positions", rel_c)
    for i in range(0, len(lc), 20):
        w = lc[i:i + 20]
        say(F, f"(c) median of positions {xc[i]}-{xc[i + len(w) - 1]}", f"{st.median(w):.2e}", rel_c)

    # the bound's shape: per-step error e on a channel with decay a accumulates as e (1 - a^k) / (1 - a)
    rel_d = f"{BP}/runs_d/anneal_out/decay_stats.txt"
    with open(P(rel_d)) as f:
        hit = [ln for ln in f if ln.startswith("pbd430a ")]
    assert len(hit) == 1
    a = float(re.search(r"decay a .* max ([\d.]+)", hit[0]).group(1))
    say(F, "slowest trained decay a_max of pbd430a", f"{a}", rel_d)
    kk = np.arange(1, len(xb) + 1)
    shape = (1 - a ** kk) / (1 - a)
    bound = rb[0] * shape  # equals series (b)'s first value at k = 1
    say(F, "bound shape at k = 1 / k = 148", f"{bound[0]:.6f} / {bound[-1]:.6f}", rel_b)

    fig, ax = plt.subplots(figsize=(WIDTH, 4.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.35)
    ax.plot(tick_a, err_a, color=C1, lw=1.0, alpha=0.9, label="pbd430a, FIDESlib 2^17, 2026-09-19: logits relErrRms (median of 64)")
    xa_fit = np.arange(5, 136)
    ax.plot(xa_fit, amp * xa_fit ** sl, color=C1, lw=1.6, ls=(0, (5, 2)),
            label=f"  power law {amp:.6f}·tick^{sl:.3f} (block-mean fit)")
    ax.plot(ca, ma, ls="none", marker="o", ms=5.5, mfc=C1, mec=SURFACE, mew=1.0, label="  10-step block medians")
    ax.plot(xb, rb, color=C2, lw=1.4, label="pbd430a, Perseus 2^16, 2026-11-02: reply/state relErrRms")
    ax.plot(xb, lb, color=C2, lw=1.0, ls=(0, (1, 1.2)), label="pbd430a, Perseus 2^16, 2026-11-02: logits relErrRms")
    ax.plot(xb, cc - dd * np.exp(-k / 60.0), color=INK2, lw=1.4, ls=(0, (5, 2)),
            label=f"  saturating fit to reply: {cc:.5f} − {dd:.5f}·exp(−k/60)")
    ax.plot(cb, mrb, ls="none", marker="o", ms=5.5, mfc=C2, mec=SURFACE, mew=1.0, label="  10-step block medians")
    ax.plot(cb, mlb, ls="none", marker="o", ms=5.5, mfc=C2, mec=SURFACE, mew=1.0)
    ax.plot(xc, lc, color=C3, lw=0.75, alpha=0.6, label="RWKV-7 0.1B, Perseus 2^16, 2026-11-05: logits relErrRms (1 conversation)")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 0.25)
    # inset, its own y-range: the geometric-series bound shape against the measured reply series
    ins = ax.inset_axes([0.035, 0.57, 0.30, 0.28])
    ins.plot(xb, bound, color=MUTED, lw=1.0, ls=(0, (1, 1.5)))
    ins.plot(xb, rb, color=C2, lw=1.0)
    ins.set_yscale("log")
    ins.set_xlim(-2, 150)
    ins.tick_params(labelsize=6, length=2)
    ins.set_xticks([0, 50, 100, 150])
    ins.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(True)
        ins.spines[sp].set_color(MUTED)
        ins.spines[sp].set_linewidth(0.6)
    ins.set_title(f"bound shape e(1−a^k)/(1−a), a = {a:.5f} (dotted)\nvs 11-02 reply series (solid), e = its first value",
                  fontsize=6, loc="left", color=INK2, pad=2)
    ax.set_xlim(-2, 158)
    ax.set_xlabel("step (token position in the free run)")
    ax.set_ylabel("relative error, rms (log scale)")
    ax.set_title("Per-step error over the same free runs: reported per run, not one law")
    ax.legend(loc="upper center", bbox_to_anchor=(0.47, -0.12), ncol=2, fontsize=6.9, handlelength=2.2, columnspacing=1.2)
    source_line(fig, [
        f"Sources: {rel_a} (relerr_rms_median); {rel_b} (ticks[].relErrRms_vs_exact, logits_relErrRms_vs_exact);",
        f"{rel_c} (logits_relErrRms); inset: decay a = max of the trained decays of pbd430a, {rel_d}.",
        "Errors are measured against each model's own plaintext (fidelity, not quality).",
    ])
    save(fig, out, F)


# ---------------------------------------------------------------- shared readers for throughput / prefill
def pub_gen():
    rel = f"{BP}/gpu_20261102_pub/run_cfg-best-em_pub/record.json"
    r = jload(rel)
    s = [t["seconds"] for t in r["ticks"] if t["fenced"] is False and str(t.get("config", "")).startswith("gen")]
    return rel, r, s


def rwkv_tokens(rel, mode=None):
    rows = [x for x in jsonl(rel) if x.get("kind") == "token" and (mode is None or x.get("mode") == mode)]
    return [x["seconds"] for x in rows]


def rwkv_passes(rel):
    return [x["seconds"] for x in jsonl(rel) if x.get("kind") == "pass"]


# ---------------------------------------------------------------- figure 3
def fig_throughput(out):
    F = "throughput_latency"
    cats = []
    rel, r, s = pub_gen()
    m = st.mean(s)
    say(F, "pbd430a: unfenced gen passes", f"{len(s)}, mean {m:.4f} s (min {min(s):.3f}, max {max(s):.3f})", rel)
    say(F, "pbd430a: build commit / keys", f"{r['build_id']['commit']} / {r['keys']['mode']}", rel)
    cats.append(("pbd430a 405M, 64 conversations", 64 / m, m, C1, "64 / mean pass"))
    say(F, "pbd430a: aggregate tok/s = 64 / mean", f"{64 / m:.3f}", rel)

    p_rels = [f"{NP}/f2_01b/run_best_P{i}.jsonl" for i in (1, 2, 3)]
    allp = []
    for pr in p_rels:
        v = rwkv_tokens(pr)
        allp += v
        say(F, "RWKV-7 0.1B decode median (this file)", f"{st.median(v):.4f} s over {len(v)} tokens", pr)
    d01 = st.median(allp)
    say(F, "RWKV-7 0.1B decode median, P1-P3 pooled", f"{d01:.4f} s; aggregate {1 / d01:.4f} tok/s", ", ".join(p_rels))
    cats.append(("RWKV-7 0.1B, 1 conversation", 1 / d01, d01, C2, ""))

    br = f"{NP}/f2_01b/batch_k32_filled_best_pob2.jsonl"
    v = rwkv_passes(br)
    b01 = st.median(v)
    say(F, "RWKV-7 0.1B K=32 pass median", f"{b01:.4f} s over {len(v)} passes (min {min(v):.3f}, max {max(v):.3f}); "
        f"aggregate 32/median {32 / b01:.4f} tok/s", br)
    cats.append(("RWKV-7 0.1B, 32 conversations", 32 / b01, b01, C2, ""))

    t_rels = [f"{NP}/f2_04b/run_pf32_best_L{i}.jsonl" for i in (1, 5)]
    allt = []
    for tr in t_rels:
        v = rwkv_tokens(tr, "decode")
        allt += v
        say(F, "RWKV-7 0.4B decode-tail median (this file)", f"{st.median(v):.4f} s over {len(v)} tokens", tr)
    d04 = st.median(allt)
    say(F, "RWKV-7 0.4B decode-tail median, L1+L5 pooled", f"{d04:.4f} s; aggregate {1 / d04:.4f} tok/s", ", ".join(t_rels))
    cats.append(("RWKV-7 0.4B, 1 conversation", 1 / d04, d04, C3, ""))

    br4 = f"{NP}/f2_04b/batch_k32_filled_best_pob2.jsonl"
    v = rwkv_passes(br4)
    b04 = st.median(v)
    say(F, "RWKV-7 0.4B K=32 pass median", f"{b04:.4f} s over {len(v)} passes (min {min(v):.3f}, max {max(v):.3f}); "
        f"aggregate 32/median {32 / b04:.4f} tok/s", br4)
    cats.append(("RWKV-7 0.4B, 32 conversations", 32 / b04, b04, C3, ""))
    r01 = (32 / b01) / (1 / d01)
    r04 = (32 / b04) / (1 / d04)
    say(F, "same-build ratio, 0.1B: aggregate 32 conv / 1 conv", f"{r01:.1f}x", f"{br} vs P1-P3")
    say(F, "same-build ratio, 0.4B: aggregate 32 conv / 1 conv", f"{r04:.1f}x", f"{br4} vs L1/L5 decode tails")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.9), sharey=True)
    fig.subplots_adjust(left=0.29, right=0.97, top=0.82, bottom=0.27, wspace=0.12)
    y = np.arange(len(cats))[::-1]
    for ax, idx, xl in ((a1, 1, "tokens per second, all conversations (full lanes)"),
                        (a2, 2, "seconds per token, one conversation")):
        vals = [c[idx] for c in cats]
        ax.barh(y, vals, height=0.62, color=[c[3] for c in cats], edgecolor=SURFACE, linewidth=2)
        for yy, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yy, f"{v:.2f}" if v < 10 else f"{v:.1f}", va="center", fontsize=8, color=INK)
        ax.set_xlabel(xl)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y)
    a1.set_yticklabels([c[0] for c in cats], fontsize=8, color=INK)
    for yy, val, rr in ((y[2], 32 / b01, r01), (y[4], 32 / b04, r04)):
        a1.text(val + 1.05, yy, f"{rr:.1f}× its 1-conversation rate (same build)", va="center", fontsize=7, color=INK2)
    a1.set_title("Aggregate throughput", fontsize=9.5)
    a2.set_title("Per-conversation latency", fontsize=9.5)
    fig.suptitle("Batched serving on one RTX PRO 6000 Blackwell, ring 2^16, Perseus runtime",
                 x=0.01, ha="left", fontsize=10.5)
    source_line(fig, [
        f"Sources: {rel} (unfenced gen passes); {NP}/f2_01b/run_best_P{{1,2,3}}.jsonl; {br};",
        f"{NP}/f2_04b/run_pf32_best_L{{1,5}}.jsonl (decode tails); {br4}.",
        "pbd430a: Perseus speedrun build 50666f6f, 2026-11-02 host. RWKV-7: perseus-rwkv7 build (_pc 1ef01cc1), another host (600 W).",
        "Different builds and hosts: no ratio between pbd430a and RWKV-7. The 16x ratios compare one model on one build.",
    ])
    save(fig, out, F)


# ---------------------------------------------------------------- figure 4
def fig_prefill(out):
    F = "prefill"
    groups = []
    # RWKV-7: prefill rows give a pass of T prompt tokens; decode tail = token rows with mode "decode"
    for name, rels, col in (("RWKV-7 0.1B", [f"{NP}/f2_01b/run_pf32_best_L{i}.jsonl" for i in (1, 5, 9, 25)], C2),
                            ("RWKV-7 0.4B", [f"{NP}/f2_04b/run_pf32_best_L{i}.jsonl" for i in (1, 5)], C3)):
        pf, dec = [], []
        for rel in rels:
            rows = jsonl(rel)
            p = [x for x in rows if x.get("kind") == "prefill"]
            d = [x["seconds"] for x in rows if x.get("kind") == "token" and x.get("mode") == "decode"]
            per = [x["seconds"] / x["T"] for x in p]
            say(F, f"{name} prefill passes (seconds, T)", ", ".join(f"{x['seconds']:.2f}/{x['T']}" for x in p), rel)
            say(F, f"{name} decode tail median", f"{st.median(d):.4f} s over {len(d)} tokens", rel)
            pf += per
            dec += d
        a, b = st.median(pf), st.median(dec)
        say(F, f"{name} s per prompt token (median over {len(pf)} passes) / s per generated token (median)",
            f"{a:.4f} / {b:.4f}; within-session ratio {b / a:.1f}x", "; ".join(rels))
        groups.append((name + "\n1 conversation, T = 32 per pass", a, b, col, f"{min(pf):.3f}-{max(pf):.3f}"))
    rel = f"{BP}/gpu_20261102_best/gpu/run_cfg-best-em_own/record.json"
    r = jload(rel)
    rd = [t for t in r["ticks"] if t["fenced"] is False and str(t["config"]).startswith("readA")]
    gn = [t for t in r["ticks"] if t["fenced"] is False and str(t["config"]).startswith("gen")]
    pos = {t["token_values"] for t in rd}
    assert len(pos) == 1, pos
    npos = pos.pop()
    a = st.mean(t["seconds"] for t in rd) / npos
    b = st.mean(t["seconds"] for t in gn)
    say(F, "pbd430a read-A passes (unfenced): count, mean s, positions per pass",
        f"{len(rd)}, {st.mean(t['seconds'] for t in rd):.4f}, {npos} ({rd[0]['describe']})", rel)
    say(F, "pbd430a gen passes (unfenced): count, mean s", f"{len(gn)}, {b:.4f}", rel)
    say(F, "pbd430a s per prompt token / s per generated token per conversation",
        f"{a:.4f} / {b:.4f}; within-session ratio {b / a:.1f}x", rel)
    say(F, "pbd430a build commit / keys", f"{r['build_id']['commit']} / {r['keys']['mode']}", rel)
    groups.insert(0, ("pbd430a 405M\nread: 64 positions of a document per pass", a, b, C1, ""))

    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    fig.subplots_adjust(left=0.33, right=0.79, top=0.86, bottom=0.25)
    y = np.arange(len(groups))[::-1] * 1.0
    h = 0.34
    for yy, (lab, a, b, col, _) in zip(y, groups):
        ax.barh(yy + h / 2 + 0.02, a, height=h, color=col, alpha=0.45, edgecolor=SURFACE, linewidth=1)
        ax.barh(yy - h / 2 - 0.02, b, height=h, color=col, edgecolor=SURFACE, linewidth=1)
        ax.text(a * 1.12, yy + h / 2 + 0.02, f"{a:.3f} s per prompt token read", va="center", fontsize=7.5, color=INK)
        ax.text(b * 1.12, yy - h / 2 - 0.02, f"{b:.2f} s per generated token", va="center", fontsize=7.5, color=INK)
        ax.text(1.03, yy, f"{b / a:.1f}×", transform=ax.get_yaxis_transform(), va="center", fontsize=10, color=INK)
        if lab.startswith("pbd430a"):
            ax.text(1.03, yy - 0.2, "= the lane count: a reading\npass and a generation\npass cost the same",
                    transform=ax.get_yaxis_transform(), va="top", fontsize=6.3, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 150)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(y)
    ax.set_yticklabels([g[0] for g in groups], fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("seconds per token, one conversation (log scale); pale = prompt read, solid = generation")
    ax.text(1.03, 1.0, "ratio, gen. / read,\nsame session", transform=ax.transAxes, fontsize=7,
            color=INK2, va="bottom")
    ax.set_title("Reading vs generating, within one session per model")
    source_line(fig, [
        f"Sources: {NP}/f2_01b/run_pf32_best_L{{1,5,9,25}}.jsonl and {NP}/f2_04b/run_pf32_best_L{{1,5}}.jsonl (kind prefill: seconds / T;",
        f"decode-tail token rows: median); {rel} (unfenced readA and gen passes, mean).",
        "Ratios are within one session and one build each. pbd430a (Perseus speedrun 50666f6f) and RWKV-7 (perseus-rwkv7) are different",
        "builds on different hosts: compare the ratios, not the bars across models. A pbd430a gen pass carries one token for each of 64 conversations.",
    ])
    save(fig, out, F)


# ---------------------------------------------------------------- figure 5
def fig_gate(out):
    F = "rwkv7_gate"
    rows = []
    for m, exact_lambada in (("0.1b", ("log", f"{OLD}/out_d_run_0.1b.log")),
                             ("0.4b", ("jsonl", f"{NP}/gate/0.4b/eval.jsonl"))):
        rel = f"{NP}/gate/{m}/eval_final.jsonl"
        ev = jsonl(rel)
        un = [x for x in ev if x.get("tag") == "final_unclamped" and x.get("clamp") is False]
        cl = [x for x in ev if x.get("tag") == "final_clamp" and x.get("clamp") is True]
        assert len(un) == 1 and len(cl) == 1
        un, cl = un[0], cl[0]
        kind, lrel = exact_lambada
        if kind == "log":
            with open(P(lrel)) as f:
                hit = [ln for ln in f if re.search(r"\] exact rc=0", ln)]
            assert len(hit) == 1
            ex = json.loads(hit[0][hit[0].index("{"):])
        else:
            ex = [x for x in jsonl(lrel) if x.get("tag") == "exact"]
            assert len(ex) == 1
            ex = ex[0]
        assert abs(ex["wt_ppl"] - un["wt_ppl_exact_same_run"]) < 1e-9
        say(F, f"{m} exact WT103 ppl", f"{un['wt_ppl_exact_same_run']:.4f}", rel)
        say(F, f"{m} polynomial unclamped WT103 ppl", f"{un['wt_ppl']:.4f} (clamped {cl['wt_ppl']:.4f})", rel)
        say(F, f"{m} exact LAMBADA acc", f"{100 * ex['lambada_acc']:.2f} %", lrel)
        say(F, f"{m} polynomial unclamped LAMBADA acc", f"{100 * un['lambada_acc']:.2f} % (clamped {100 * cl['lambada_acc']:.2f} %)", rel)
        say(F, f"{m} top-1 agreement vs exact (unclamped)", f"{100 * un['top1_agree']:.2f} %", rel)
        rows.append((m.replace("b", "B"), un["wt_ppl_exact_same_run"], un["wt_ppl"], 100 * ex["lambada_acc"],
                     100 * un["lambada_acc"], 100 * un["top1_agree"], "pass"))
    rel = f"{NP}/vd_runs/1.5b_pipe/verdict_final.json"
    v = jload(rel)
    say(F, "1.5b exact WT103 ppl / LAMBADA", f"{v['exact']['wt_ppl']:.4f} / {100 * v['exact']['lambada_acc']:.2f} %", rel)
    say(F, "1.5b polynomial unclamped WT103 ppl / LAMBADA / top-1",
        f"{v['unclamped']['wt_ppl']:.4f} / {100 * v['unclamped']['lambada_acc']:.2f} % / {100 * v['unclamped']['top1_agree']:.2f} %", rel)
    say(F, "1.5b verdict, checks", f"{v['verdict']}, {v['checks']}", rel)
    say(F, "gate bars", json.dumps(v["bars"]), rel)
    rows.append(("1.5B", v["exact"]["wt_ppl"], v["unclamped"]["wt_ppl"], 100 * v["exact"]["lambada_acc"],
                 100 * v["unclamped"]["lambada_acc"], 100 * v["unclamped"]["top1_agree"], v["verdict"].lower()))
    bars = v["bars"]

    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 3.6))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.27, wspace=0.32)
    x = np.arange(len(rows))
    w = 0.38
    labels = [f"{r[0]}" + ("\nFAILS gate" if r[6] == "fail" else "") for r in rows]
    for ax, (ie, ip, title, fmt) in zip(axs[:2], ((1, 2, "WikiText-103 token perplexity", "{:.2f}"),
                                                  (3, 4, "LAMBADA accuracy (%)", "{:.2f}"))):
        e = [r[ie] for r in rows]
        p = [r[ip] for r in rows]
        ax.bar(x - w / 2 - 0.01, e, w, color=MUTED, edgecolor=SURFACE, linewidth=1.5, label="exact model")
        ax.bar(x + w / 2 + 0.01, p, w, color=C1, edgecolor=SURFACE, linewidth=1.5, label="polynomial, unclamped")
        for xx, a, b in zip(x, e, p):
            ax.text(xx - w / 2, a, fmt.format(a), ha="center", va="bottom", fontsize=6.5, color=INK)
            ax.text(xx + w / 2, b, fmt.format(b), ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(e + p) * 1.15)
    t = [r[5] for r in rows]
    axs[2].bar(x, t, 0.55, color=[C2 if r[6] == "fail" else C1 for r in rows], edgecolor=SURFACE, linewidth=1.5)
    for xx, v_ in zip(x, t):
        axs[2].text(xx, v_ + 0.1, f"{v_:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    axs[2].axhline(100 * bars["top1"], color=INK2, lw=0.9, ls=(0, (4, 2)))
    axs[2].text(2, 100 * bars["top1"] + 0.15, f"gate bar {100 * bars['top1']:.0f} %", ha="center", fontsize=6.8,
                color=INK2)
    axs[2].set_ylim(90, 100)
    axs[2].set_title("Top-1 agreement with exact (%)", fontsize=9)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, fontsize=8)
    axs[2].grid(axis="x", visible=False)
    axs[0].set_ylabel("perplexity")
    axs[1].set_ylabel("accuracy (%)")
    axs[2].set_ylabel("agreement (%)")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.2, 1.24), ncol=2, fontsize=7.5)
    fig.suptitle("Polynomial RWKV-7 vs the exact model, unclamped (plaintext evaluation)",
                 x=0.01, y=0.985, ha="left", fontsize=10.5)
    source_line(fig, [
        f"Sources: {NP}/gate/{{0.1b,0.4b}}/eval_final.jsonl (tag final_unclamped; exact ppl = wt_ppl_exact_same_run);",
        f"exact LAMBADA: {OLD}/out_d_run_0.1b.log ('exact' line), {NP}/gate/0.4b/eval.jsonl (tag exact); {rel}.",
        f"Gate bars (from the verdict record): ppl within {100 * bars['ppl_rel']:.1f} %, LAMBADA within {bars['lambada_pts']} points, "
        f"top-1 >= {100 * bars['top1']:.0f} %. 1.5B fails on ppl (+{100 * v['ppl_rel_vs_exact']:.1f} %) and top-1. Perplexity y-axis starts at 0.",
    ])
    save(fig, out, F)


# ---------------------------------------------------------------- figure 6
def sheet_window(label_re):
    rel = f"{NP}/sheet.log"
    with open(P(rel)) as f:
        hits = [ln for ln in f if re.search(label_re, ln)]
    assert len(hits) == 1, (label_re, hits)
    m = re.match(r"\[pod3 (\S+)\] .* rc=(\d+) (\d+) s", hits[0])
    end = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    dur = int(m.group(3))
    return rel, end.timestamp() - dur, end.timestamp(), hits[0].strip(), int(m.group(2))


def sampler_max(t0, t1):
    rel = f"{NP}/monitor/usage_20261104.csv"
    ram, vram, n = [], [], 0
    with open(P(rel)) as f:
        for row in csv.reader(f):
            ts = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
            if t0 <= ts <= t1:
                ram.append(int(row[8]))   # column 9: host RAM used, MiB (free -m)
                vram.append(int(row[3]))  # column 4: GPU memory used, MiB
                n += 1
    return rel, max(ram), st.median(ram), max(vram), n


def mem_total(rel):
    h = jsonl(rel)[0]
    m = re.search(r"total ([\d.]+) MB", h["mem_report_text"])
    return float(m.group(1)), h


def fig_memory(out):
    F = "memory"
    cases = [
        ("0.1B, 32 conv., no flag", f"{NP}/f2_01b/batch_k32_best_pob2.jsonl", r"\] 01b batched K32 b2 rc="),
        ("0.1B, 32 conv., host flag\n(RWKV7_SHED_TWIN=1)", f"{NP}/f2_01b/batch_k32_best_pob2_shed.jsonl",
         r"\] 01b batched K32 b2 SHED"),
        ("0.1B, 32 conv., both flags\n(+ RWKV7_LEAN_PREFIX=1)", f"{NP}/f2_01b/batch_k32_best_pob2_lean.jsonl",
         r"\] 01b batched K32 b2 LEAN"),
        ("0.4B, 32 conv., both flags", f"{NP}/f2_04b/batch_k32_filled_best_pob2.jsonl",
         r"\] 04b-filled batched K32 b2 LEAN"),
    ]
    rows = []
    for lab, rel, pat in cases:
        dev, h = mem_total(rel)
        srel, t0, t1, line, rc = sheet_window(pat)
        urel, hmax, hmed, vmax, n = sampler_max(t0, t1)
        summ = [x for x in jsonl(rel) if x.get("kind") == "summary"]
        passes = len([x for x in jsonl(rel) if x.get("kind") == "pass"])
        flags = f"shed_twin={'shed_twin' in h}, lean_prefix={h.get('lean_prefix', False)}"
        say(F, f"{lab.splitlines()[0]}: device total (mem_report_text, MiB)", f"{dev:,.1f}", rel)
        say(F, f"{lab.splitlines()[0]}: header flags; passes; summary wall_s",
            f"{flags}; {passes}; {summ[0]['wall_s'] if summ else 'no summary'}", rel)
        say(F, f"{lab.splitlines()[0]}: sheet window", f"{datetime.fromtimestamp(t0, timezone.utc):%H:%M:%S}Z-"
            f"{datetime.fromtimestamp(t1, timezone.utc):%H:%M:%S}Z (rc={rc}) from line: {line[:90]}", srel)
        say(F, f"{lab.splitlines()[0]}: host-wide RAM max / median (MiB); GPU used max (MiB); samples",
            f"{hmax:,} / {hmed:,.0f}; {vmax:,}; {n}", urel)
        rows.append((lab, dev, hmax, C2 if lab.startswith("0.1B") else C3))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.6), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.27, wspace=0.12)
    y = np.arange(len(rows))[::-1]
    for ax, idx, xl in ((a1, 1, "device memory, runtime report (MiB)"),
                        (a2, 2, "host RAM in use, whole host (MiB)")):
        vals = [r[idx] for r in rows]
        ax.barh(y, vals, height=0.6, color=[r[3] for r in rows], edgecolor=SURFACE, linewidth=2)
        for yy, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yy, f"{v:,.0f}", va="center", fontsize=8, color=INK)
        ax.set_xlabel(xl)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    a1.set_yticks(y)
    a1.set_yticklabels([r[0] for r in rows], fontsize=7.8, color=INK)
    a1.set_title("Device (one card)", fontsize=9.5)
    a2.set_title("Host (sampler maximum over the session)", fontsize=9.5)
    fig.suptitle("Memory of a 32-conversation RWKV-7 session, before and after the two memory flags",
                 x=0.01, ha="left", fontsize=10.5)
    source_line(fig, [
        f"Sources: {NP}/f2_01b/batch_k32_best_pob2{{,_shed,_lean}}.jsonl, {NP}/f2_04b/batch_k32_filled_best_pob2.jsonl (header mem_report_text 'total');",
        f"{NP}/monitor/usage_20261104.csv (column 9, free -m used, 5 s rows) over each session's window from {NP}/sheet.log.",
        "The runtime's 'MB' are MiB. Host RAM is host-wide, not per process. The both-flags 0.1B run stopped at pass 52 of 61 (no summary).",
        "0.1B no-flag run: the current (non-filled) layout; 0.4B: filled layout. Same card model; one build line (perseus-rwkv7, _pc 1ef01cc1).",
    ])
    save(fig, out, F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_cost(a.out)
    fig_error(a.out)
    fig_throughput(a.out)
    fig_prefill(a.out)
    fig_gate(a.out)
    fig_memory(a.out)
    man = {"generated_by": "figures/make_figures.py", "algorithm": "sha256",
           "records": {k: READ[k] for k in sorted(READ)}}
    with open(os.path.join(HERE, "sources.json"), "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    print(f"sources.json: {len(READ)} records")


if __name__ == "__main__":
    main()
