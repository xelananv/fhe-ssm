#!/usr/bin/env python3
"""Render a recorded demo session into one Markdown report (2026-09-04).

Reads the client session dir written by ml-eval/fhe_client.py + 71_mac_demo_session.sh
(tokens.json, transcript.json, lanes_text.json, audit.log, fid_mac/fidelity.json), the
pulled pod records (build/BUILD_IDENTITY.txt, server.log served lines, contract_r17.json)
and the verification kit's manifests, and writes DEMO_SESSION_REPORT.md beside the
session. Every number is transcribed from a named file; nothing is computed from memory.
"""
import argparse, json, os, re, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True, help="client session dir (tokens.json, transcript.json ...)")
ap.add_argument("--pod", default=None, help="pulled pod records dir (archive's pod/)")
ap.add_argument("--kit", default=None, help="verification kit dir")
ap.add_argument("--out", default=None)
a = ap.parse_args()
S = a.session
out = a.out or os.path.join(S, "DEMO_SESSION_REPORT.md")

def load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

tokens = load(os.path.join(S, "tokens.json")) or {}
transcript = load(os.path.join(S, "transcript.json")) or []
lanes_text = load(os.path.join(S, "lanes_text.json")) or {}
fid = load(os.path.join(S, "fid_mac", "fidelity.json"))
L = []
L.append(f"# Demo session report — `{os.path.basename(S.rstrip('/'))}`\n")
L.append("Fully-secure mode: keys minted on the Mac, only eval material uploaded, every ciphertext "
         "encrypted and decrypted on the Mac. Source files are named beside each number.\n")
# ---- build / pod identity
if a.pod:
    bi = os.path.join(a.pod, "build", "BUILD_IDENTITY.txt")
    if os.path.exists(bi):
        L.append("## Server build identity (`pod/build/BUILD_IDENTITY.txt`)\n")
        L.append("```\n" + open(bi).read().strip() + "\n```\n")
    ct = load(os.path.join(a.pod, "contract_r17.json"))
    if ct:
        L.append("## Key contract (`pod/contract_r17.json`)\n")
        L.append(f"ring {ct.get('ring')}, slots {ct.get('slots')}, depth {ct.get('depth')}, extraDepth {ct.get('extraDepth')}, "
                 f"scaleBits {ct.get('scaleBits')}, rotation amounts {len(ct.get('rotationAmounts', [])) if isinstance(ct.get('rotationAmounts'), list) else ct.get('rotationAmounts')}, "
                 f"automorphism indices {len(ct.get('autoIndices', [])) if isinstance(ct.get('autoIndices'), list) else ct.get('autoIndices')}, "
                 f"lanesBlock {ct.get('lanesBlock')}, secure {ct.get('secure')}\n")
# ---- session geometry
L.append("## Session\n")
L.append(f"- lanes: **{tokens.get('lanes')}**, ragged: {tokens.get('ragged')}, tag `{tokens.get('tag')}`, d {tokens.get('d')}, dpad {tokens.get('dpad')}, vocab {tokens.get('vocab')} (`tokens.json`)")
L.append(f"- ticks recorded: **{len(transcript)}** (`transcript.json`)\n")
# ---- per-tick table
if transcript:
    L.append("## Per-tick timing (`transcript.json`; seconds; serve = upload + server tick + download)\n")
    L.append("| tick | prompt lanes | generating | retired | enc | serve | dec |")
    L.append("|---|---|---|---|---|---|---|")
    srv = []
    for t in transcript:
        ph = t.get("phases", [])
        e, s_, d = t.get("encSec"), t.get("serveSec"), t.get("decSec")
        if isinstance(s_, (int, float)): srv.append(s_)
        L.append(f"| {t.get('tick')} | {ph.count('prompt')} | {ph.count('gen')} | {ph.count('retired')} | "
                 f"{'' if e is None else round(e,1)} | {'' if s_ is None else round(s_,1)} | {'' if d is None else round(d,1)} |")
    if srv:
        L.append(f"\nserve seconds: mean **{statistics.mean(srv):.1f}**, median {statistics.median(srv):.1f}, "
                 f"min {min(srv):.1f}, max {max(srv):.1f}, sum {sum(srv):.0f} over {len(srv)} ticks "
                 f"(includes the wire both ways; the server-side per-request numbers are in `pod/server.log`).\n")
# ---- server-side served lines
if a.pod and os.path.exists(os.path.join(a.pod, "server.log")):
    served = []
    for ln in open(os.path.join(a.pod, "server.log"), errors="replace"):
        if ln.startswith('{"serve":"served"'):
            try: served.append(json.loads(ln))
            except Exception: pass
    if served:
        L.append("## Server-side per-request record (`pod/server.log`, `{\"serve\":\"served\"}` lines)\n")
        L.append("| req | reqMsPerToken | reqEvalMs | reqBootMs | reqEncPtMs | reqBoots | peakVramGB |")
        L.append("|---|---|---|---|---|---|---|")
        for r in served:
            L.append(f"| {r.get('req')} | {r.get('reqMsPerToken')} | {r.get('reqEvalMs')} | {r.get('reqBootMs')} | {r.get('reqEncPtMs')} | {r.get('reqBoots')} | {r.get('peakVramGB')} |")
        ms = [r.get("reqMsPerToken") for r in served if isinstance(r.get("reqMsPerToken"), (int, float))]
        if ms:
            L.append(f"\nserver tick (reqMsPerToken): mean **{statistics.mean(ms)/1000:.1f} s**, median {statistics.median(ms)/1000:.1f} s, "
                     f"last {ms[-1]/1000:.1f} s over {len(ms)} requests; per conversation-token at {tokens.get('lanes')} lanes: "
                     f"**{statistics.median(ms)/1000/max(1, int(tokens.get('lanes') or 1)):.2f} s** (median tick / lanes; arithmetic on the record).\n")
# ---- fidelity probe
if fid:
    if isinstance(fid, dict) and "top1Agree" in fid:
        L.append("## Fidelity probe through the live server, Mac keys (`fid_mac/fidelity.json`)\n")
        L.append(f"- comparisons {fid.get('comparisons')}, top-1 agreement **{fid.get('top1Agree')}/{fid.get('comparisons')}** "
                 f"({fid.get('top1AgreeFrac')}), relErrRms median {fid.get('relErrRmsMedian')}, worst {fid.get('relErrRmsWorst')}, mode: {fid.get('mode')}\n")
    elif isinstance(fid, list):
        n = len(fid); agree = sum(1 for r in fid if r.get("top1")); worst = max((r.get("relErrRms", 0) for r in fid), default=None)
        L.append("## Fidelity probe through the live server, Mac keys (`fid_mac/fidelity.json`)\n")
        L.append(f"- comparisons {n}, top-1 agreement **{agree}/{n}**, relErrRms worst {worst}\n")
# ---- audit
al = os.path.join(S, "audit.log")
if os.path.exists(al):
    tail = open(al, errors="replace").read().strip().splitlines()[-6:]
    L.append("## Ragged-lane audit (`audit.log`, spec_decode/audit_ragged.py A/B/C)\n")
    L.append("```\n" + "\n".join(tail) + "\n```\n")
# ---- lane texts
if lanes_text:
    L.append(f"## The {sum(1 for k in lanes_text if str(k).isdigit())} conversations, unedited (`lanes_text.json`; prompt then generated text)\n")
    prompts = tokens.get("prompts") or []
    for k in sorted((k for k in lanes_text if str(k).isdigit()), key=lambda x: int(x)):
        txt = lanes_text[k]
        p = prompts[int(k)] if int(k) < len(prompts) else ""
        gen = txt[len(p):] if p and txt.startswith(p) else txt
        L.append(f"**{k}.** {p.strip()} **→** {gen.strip()}\n")
# ---- kit
if a.kit and os.path.isdir(a.kit):
    L.append("## Verification kit\n")
    mf = os.path.join(a.kit, "MANIFEST.sha256")
    n = sum(1 for _ in open(mf)) if os.path.exists(mf) else 0
    L.append(f"- `{a.kit}`: {n} files in MANIFEST.sha256; see HOWTO.md there for the exact re-decryption commands.")
    for sub in ("session_cts", "fid_mac_cts"):
        m = os.path.join(a.kit, sub, "MANIFEST.jsonl")
        if os.path.exists(m):
            rows = [json.loads(x) for x in open(m) if x.strip()]
            L.append(f"- `{sub}`: {len(rows)} files, {sum(r.get('bytes', 0) for r in rows)/1e9:.2f} GB, ticks {len(set(r.get('tick') for r in rows))}")
    L.append("")
open(out, "w").write("\n".join(L) + "\n")
print("wrote", out, len(L), "lines")
