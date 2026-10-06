#!/usr/bin/env bash
# 91_s37_serve.sh -- ONE served S3.7 arm on the POD's own keys, through 60_fidelity_gate.sh's pod-side driver
#                    (spec_decode/fidelity_tick.py: 4 lanes, teacher-forced, every reply decrypted and compared
#                    with the plaintext model on the pod). No Mac in the loop: this is the one-card session-1 form
#                    of the plan's served arms (A2-half, A5, A6, A7) at ring 2^15, and it works at 2^17 unchanged.
#
# usage   ARM=<name> [XBIN=0|1] [ARM_FLAGS="--schedule parent-first --newton-depth2"] [TICKS=8]
#         [PROMPTS=<4-line file>] [STORE=<path>] bash 91_s37_serve.sh
#         env per _lib.sh: LOG_RING EXTRA_DEPTH SECURE LANES_BLOCK EXTRA_FLAGS (e.g. "--canonical-carry
#         --periodic-encode") DEVICES KEYS_POD. The arm's flags are appended to EXTRA_FLAGS for the server.
#         Default STORE = a FRESH file per arm ($DEMO/s37/serve_<ARM>/store.bin; tick 0 builds it along the
#         arm's own trajectory -- the stamp differs per schedule/drop arm); pass STORE= to reuse one.
#         PROMPTS default: results/dense-demo-s31/sessions/fid_prompts4_2identical.txt (lanes 0 and 1 identical
#         = the S3.5 s2b divergence check for free; lanes 2-3 different). On the PLAIN shape (LANES_BLOCK=0) every
#         lane must be identical -> fid_prompts4_identical.txt.
# records $DEMO/s37/serve_<ARM>/{fidelity.json,logs/server.jsonl,...} (60's FID_OUT), run.log (60's output),
#         TICKS.md (per-tick table + verdict lines), ARM.env (the exact env), CAMPAIGN_LOG.md line.
# verdict lines (records, never a pass/fail against the plan -- that is written by hand per arm):
#         enc0From = first tick from which reqEncPtMs == 0 stays 0 (store warm); bootsConst; lanesIdentical (the
#         identical-prompt lanes agree on encArgmax every tick); rowsComplete (every reply decrypted and compared);
#         failedReqs; relErrRms median/worst; top1 agree; per-tick reqMsPerToken / reqBoots / peakVramGB / storeAppends.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
: "${ARM:?ARM=<name> required}"; : "${XBIN:=0}"; : "${ARM_FLAGS:=}"; : "${TICKS:=8}"; : "${S37:=$DEMO/s37}"
: "${BINX:=$WORK/build-demo/gpu_real_model_x}"
: "${PROMPTS:=$POD_SRC/results/dense-demo-s31/sessions/fid_prompts4_2identical.txt}"
O="$S37/serve_$ARM"; mkdir -p "$O"
B="$BIN"; [ "$XBIN" = 1 ] && B="$BINX"
[ -x "$B" ] || { log "MISSING binary $B"; exit 2; }
[ -s "$PROMPTS" ] || { log "MISSING prompts $PROMPTS"; exit 2; }
: "${STORE:=$O/store.bin}"
EF="${EXTRA_FLAGS:-} $ARM_FLAGS"
{ echo "ARM=$ARM XBIN=$XBIN BIN=$B"; echo "LOG_RING=$LOG_RING EXTRA_DEPTH=$EXTRA_DEPTH SECURE=$SECURE LANES_BLOCK=${LANES_BLOCK:-1} DEVICES=${DEVICES:-}";
  echo "EXTRA_FLAGS(server)=$EF"; echo "STORE=$STORE"; echo "PROMPTS=$PROMPTS TICKS=$TICKS KEYS_POD=$KEYS_POD"; echo "binary_sha256 $(sha256sum "$B" | cut -c1-64)"; echo "utc $(utc)"; } > "$O/ARM.env"
log "serving arm $ARM: $(cat "$O/ARM.env" | tr '\n' ' ')"
BIN="$B" STORE="$STORE" EXTRA_FLAGS="$EF" FID_OUT="$O" FID_TICKS="$TICKS" FID_PROMPTS="$PROMPTS" \
  bash "$(dirname "$(readlink -f "$0")")/60_fidelity_gate.sh" > "$O/run.log" 2>&1
RC=$?
echo "60 rc=$RC (0 GO / 2 CONDITIONAL / 1 NO-GO; the arm's record is the table below, not this code)" | tee -a "$O/run.log"
python3 - "$O" "$PROMPTS" "$TICKS" <<'PY' | tee "$O/TICKS.md"
import json, sys, os, collections
O, PR, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
sl = os.path.join(O, "logs", "server.jsonl"); fj = os.path.join(O, "fidelity.json")
served, other, summ = [], [], None
if os.path.exists(sl):
    for l in open(sl, errors="replace"):
        l = l.strip()
        if not l.startswith("{"): continue
        try: d = json.loads(l)
        except Exception: continue
        if d.get("serve") == "served": served.append(d)
        elif d.get("summary") is True: summ = d
        elif any(k in d for k in ("canonicalCarry","keysLoadSeeded","x1Levers","uDrop","x8Drop","periodicEncodeCount","ptCacheHits","fatal","storeLoad","storeSave")): other.append(l[:240])
print(f"## arm {os.path.basename(O)} -- {len(served)} served ticks (asked {N})\n")
print("| tick | reqMsPerToken | reqMsPerTokenExBoot | reqEncPtMs | reqBoots | storeAppends | peakVramGB | failed |")
print("|---|---|---|---|---|---|---|---|")
for i, s in enumerate(served):
    print(f"| {i} | {s.get('reqMsPerToken')} | {s.get('reqMsPerTokenExBoot','')} | {s.get('reqEncPtMs')} | {s.get('reqBoots')} | {s.get('storeAppends','')} | {s.get('peakVramGB')} | {s.get('failed')} |")
enc = [int(s.get("reqEncPtMs") or 0) for s in served]; boots = [s.get("reqBoots") for s in served]
enc0 = None
for i in range(len(enc)):
    if all(e == 0 for e in enc[i:]): enc0 = i; break
print(f"\nVERDICT enc0From: {enc0} (first tick from which reqEncPtMs stays 0; the plan's s2b wants 0 encodes from tick 1 -- from tick 0 on a pre-built store)")
print(f"VERDICT bootsConst: {len(set(boots[1:])) <= 1 if len(boots) > 1 else 'n/a'} (ticks 1..: {sorted(set(boots[1:])) if len(boots)>1 else boots})")
print(f"VERDICT failedReqs: {sum(1 for s in served if s.get('failed'))}")
rows = []
if os.path.exists(fj):
    fjd = json.load(open(fj)); rows = fjd.get("rows", []) if isinstance(fjd, dict) else fjd   # fidelity_tick writes a LIST
prompts = [l.rstrip("\n") for l in open(PR) if l.strip()]
groups = collections.defaultdict(list)
for i, p in enumerate(prompts): groups[p].append(i)
ident = [g for g in groups.values() if len(g) > 1]
by = collections.defaultdict(dict)
for r in rows: by[r["tick"]][r["lane"]] = r
bad = []
for t in sorted(by):
    for g in ident:
        vals = {by[t][l]["encArgmax"] for l in g if l in by[t]}
        if len(vals) > 1: bad.append((t, g, sorted(vals)))
print(f"VERDICT lanesIdentical: {'n/a (no identical lanes in the prompt file)' if not ident else (not bad)} identical groups {ident} disagreements {bad}")
print(f"VERDICT rowsComplete: {len(rows)} rows vs {len(served)} ticks x {len(prompts)} lanes = {len(served)*len(prompts)}")
print(f"VERDICT decFailedTicks: {sorted({r['tick'] for r in rows if r.get('decFailed')})} (client decode failed: 'approximation error is too high' = the reply is noise-dominated)")
if rows:
    rel = sorted(r["relErrRms"] for r in rows); agree = sum(1 for r in rows if r["top1"])
    print(f"VERDICT fidelity: top1 {agree}/{len(rows)}, relErrRms median {rel[len(rel)//2]:.4g} worst {rel[-1]:.4g}; serveSec per tick {[r['serveSec'] for r in rows if r['lane']==0]}")
if summ: print("SUMMARY " + json.dumps({k: summ.get(k) for k in ("boots","storeBytes","storeHits","storeMisses","poolBuilds","peakVramGB","procPeakVramGB","periodicEncodeCount","periodicVerifyCount","periodicVerifyOk","periodicFallbackCount","ptCacheEntries","ptCacheHits","ptCacheMisses","dropsApplied","dropsSkippedDeeper","rsqrtItersSum","canonBoots","failed","harnessSha256") if k in summ}))
for l in other[:12]: print("line " + l)
PY
campaign_log "s37 serve $ARM | xbin=$XBIN flags '$EF' | rc60=$RC | $(grep -a 'VERDICT enc0From\|VERDICT lanesIdentical\|VERDICT fidelity' "$O/TICKS.md" | tr '\n' ' ' | cut -c1-400) | $O"
