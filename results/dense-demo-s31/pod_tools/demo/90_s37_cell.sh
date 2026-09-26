#!/usr/bin/env bash
# 90_s37_cell.sh -- ONE S3.7 batch cell (POD_PLAN_S37.md section 5/6): one binary, the demo base flags, a flag
#                   delta, optionally a store file; everything recorded under $DEMO/s37/<CELL>/.
#
# PURPOSE   The arms A0-A3 and the store builds of session 1 are T=1 batch cells on the real binary
#           (gpu_real_model) or the experimental copy (gpu_real_model_x, XBIN=1); the schedule / drop / wire
#           arms are SERVED sessions (70_demo_server.sh + 71_mac_demo_session.sh) and are not run here.
#           R6: one binary, one box, one load state per comparison; correctness fields cross binaries, ms do not.
# INPUTS    env per _lib.sh (DEVICES=0,1 -> --devices 0,1 --store-multi-gpu + the NCCL/shm exchange env;
#           EXTRA_FLAGS, e.g. "--canonical-carry"; LANES_BLOCK=0 for the plain block shape; KEYS).
#             CELL=<name>         required; the record dir is $DEMO/s37/<CELL>/ (an existing one is renamed .prev)
#             XBIN=0|1            0 = $BIN (real), 1 = $BINX (default $WORK/build-demo/gpu_real_model_x)
#             MODE=cell|selftest  cell (default): --tokens $TOKENS --passes $PASSES [--store-file $STOREF]
#                                 selftest: --selftest --self-d 1024 --self-dff 4096 (A1 rmsnormSeeded, 2-device smoke)
#             FLAGS="..."         the flag delta (word-split), e.g. "--no-norm-eps", "--periodic-encode --periodic-verify-every 64"
#             STOREF=<path>       --store-file (cell mode); empty = in-memory store only
#             TOKENS=1 PASSES=2   cell mode (85_schedule_ab.sh's form: pass 1 builds/loads, pass 2 is the warm pass)
#             KEYS_LOAD=1         add --keys-dir $KEYS --keys-load (the Mac eval material; no secret on the box)
#             NO_TRACE=1          drop --trace-boots (a long store build does not need the boot trace)
# OUTPUTS   $DEMO/s37/<CELL>/{CMD,loadstate.txt,BUILD_IDENTITY.txt,console.log,summary.json,passes.jsonl,
#           lines.jsonl,vram.jsonl,FIELDS.txt}; one CAMPAIGN_LOG.md line.
# PASS      no CUDA/fatal text (F74: never the rc); cell: a {"summary":true} line with failed:false;
#           selftest: a {"selftestSummary"} line. Fidelity/boots are printed, never judged here (the plan's
#           expected values are per arm; write the verdict into the arm's row by hand, records only).
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
: "${CELL:?CELL=<name> required}"; : "${XBIN:=0}"; : "${MODE:=cell}"; : "${FLAGS:=}"; : "${STOREF:=}"
: "${TOKENS:=1}"; : "${PASSES:=2}"; : "${KEYS_LOAD:=0}"; : "${NO_TRACE:=0}"; : "${S37:=$DEMO/s37}"
: "${BINX:=$WORK/build-demo/gpu_real_model_x}"
B="$BIN"; [ "$XBIN" = 1 ] && B="$BINX"
[ -x "$B" ] || { log "MISSING binary $B (XBIN=$XBIN) -- 10_bringup.sh build (BUILD_X=1 builds the copy)"; exit 2; }
[ "$MODE" = selftest ] || require_file "$ART/bundle_${TAG}.bin" "20_stage_artifacts.sh"
O="$S37/$CELL"; [ -e "$O" ] && mv "$O" "$O.$(date -u +%Y%m%dT%H%M%SZ).prev"; mkdir -p "$O"
L="$O/console.log"
read -r -a XF <<< "$FLAGS"
KF=(); [ "$KEYS_LOAD" = 1 ] && KF=(--keys-dir "$KEYS" --keys-load)
if [ "$MODE" = selftest ]; then
  MF=(--selftest --self-d 1024 --self-dff 4096)
else
  MF=(--tokens "$TOKENS" --passes "$PASSES" --sync-timers --proc-vram --vram-trace)
  [ "$NO_TRACE" = 1 ] || MF+=(--trace-boots)
  [ -n "$STOREF" ] && MF+=(--store-file "$STOREF")
fi
# shellcheck disable=SC2206
CMD=("$B" "${MODEL_FLAGS[@]}" "${DEMO_FLAGS[@]}" "${BLOCK_FLAGS[@]}" ${DEVICES_FLAGS:-} "${MF[@]}" ${KF[@]+"${KF[@]}"} ${XF[@]+"${XF[@]}"})
busy_acquire
printf '%s\n' "${CMD[*]}" > "$O/CMD"
{ record_load_state; echo "--- top CPU ---"; ps -eo pcpu,rss,comm --sort=-pcpu 2>/dev/null | head -8; echo "--- nvidia-smi ---"; nvidia-smi 2>&1 | head -40; } > "$O/loadstate.txt"
{ echo "binary $B"; echo "binary_sha256 $(sha256sum "$B" | cut -c1-64)"; echo "--- $DEMO/build/BUILD_IDENTITY.txt ---"; cat "$DEMO/build/BUILD_IDENTITY.txt" 2>/dev/null; } > "$O/BUILD_IDENTITY.txt"
{ echo "=== S3.7 CELL $CELL ($MODE) $(utc) ==="; record_load_state; echo "CMD: ${CMD[*]}"; } > "$L"
gpu_idle_or_warn || { echo "LOADSTATE CONTENDED" | tee -a "$L" >> "$O/loadstate.txt"; }
T0=$(date +%s)
"${CMD[@]}" >> "$L" 2>&1
echo "RC=$? (recorded, not trusted: F74)" >> "$L"; echo "WALL_S=$(( $(date +%s) - T0 ))" >> "$L"
busy_release

# ---------------------------------------------------------------- records
grep -a '^{"summary":true' "$L" | tail -1 > "$O/summary.json"
grep -a '"passTiming"' "$L" > "$O/passes.jsonl" || true
grep -aE '^\{"(keysLoadSeeded|canonicalCarry|x1Levers|x8Drop|uDrop|deferredScan|branchOut|periodic[A-Za-z]*|ptCache[A-Za-z]*|wvCheck|selftest|selftestSummary|storeLoad|storeSave|storeVerify|harness|fatal)"' "$L" > "$O/lines.jsonl" || true
grep -a '"vramTrace"' "$L" > "$O/vram.jsonl" || true
P=()
cuda_ok "$L" || P+=("CUDA/fatal text: $(cuda_problems "$L" | tr '\n' ';')")
if [ "$MODE" = selftest ]; then
  grep -aq '"selftestSummary"' "$L" || P+=("no selftestSummary line")
  { echo "selftest lines:"; grep -a '"selftest"' "$L" | cut -c1-200; grep -a '"selftestSummary"' "$L" | tail -1; grep -a '"keysLoadSeeded"' "$L"; } | tee "$O/FIELDS.txt"
else
  S=$(cat "$O/summary.json")
  [ -n "$S" ] || P+=("no {\"summary\":true} line")
  [ -z "$S" ] || [ "$(jget "$S" failed)" = false ] || P+=("summary failed:$(jget "$S" failed)")
  python3 - "$O/summary.json" "$O/passes.jsonl" "$O/lines.jsonl" <<'PY' | tee "$O/FIELDS.txt"
import json, sys
try: S = json.loads(open(sys.argv[1]).read().strip() or "{}")
except Exception as e: S = {}; print("summary parse error:", e)
keys = ["tag","tokens","passes","secure","ringDim","depth","extraDepth","boots","canonBoots","pfBoots","encPtCount","encPtMs",
        "msPerToken","wallMsPerToken","msPerTokenExBoot","normBootMs","gpuEvalMs","gpuEvalMsExBoot","gpuBootstrapMs","totalWallMs","setupMs",
        "unaccountedMs","unaccountedMsExBoot","unaccountedMsAfterStages","layerLoopMs","layerLoopTimedMs","layerLoopResidualMs",
        "hostPtEncodeMs","hostPtEncodeCount","hostPtEncodeStageMs","hostPtEncodeStageCount","stageMs",
        "peakVramGB","procPeakVramGB","poolHighRun","poolBuilds","poolBuildMs",
        "storeOn","storeHits","storeMisses","storeDead","storeUncompressible","storeBytes","storeFetchMs",
        "periodicEncodeOn","periodicVerifyEvery","periodicEncodeCount","periodicVerifyCount","periodicVerifyOk","periodicFallbackCount",
        "ptCacheEntries","ptCacheHits","ptCacheMisses","ptCacheMissMs","dropsApplied","dropsSettled","dropsSkippedDeeper","dropsAtTarget","rsqrtItersSum",
        "schedule","newtonDepth2","normEps","canonicalCarry","failed","harnessSha256","harnessCommit","buildUtc"]
print("FIELDS " + json.dumps({k: S.get(k) for k in keys if k in S}))
print("fidelity " + json.dumps(S.get("fidelity")))
try:
    pb = []
    for l in open(sys.argv[2]):
        d = json.loads(l); pb.append({k: d.get(k) for k in ("pass","boots","msPerToken","wallMsPerToken","msPerTokenExBoot","encPtMs","storeHits","storeMisses") if k in d})
    print("passes " + json.dumps(pb))
except Exception: pass
for l in open(sys.argv[3]):
    if l.startswith('{"keysLoadSeeded"') or l.startswith('{"canonicalCarry"') or l.startswith('{"x1Levers"') or l.startswith('{"storeLoad"') or l.startswith('{"storeSave"') or l.startswith('{"fatal"'):
        print("line " + l.strip()[:300])
PY
  echo "wvCheck(layer0 relErr): $(grep -ao '{"wvCheck":true,"layer":0[^}]*}' "$L" | grep -ao '"relErr":[0-9.e+-]*' | cut -d: -f2 | tr '\n' ' ')" | tee -a "$O/FIELDS.txt"
  echo "=== VRAM ===" | tee -a "$O/FIELDS.txt"; vram_table "$L" "$O/vram.jsonl" | tail -6 | tee -a "$O/FIELDS.txt"
fi
grep -a 'WALL_S\|^RC=' "$L" | tee -a "$O/FIELDS.txt"
if [ ${#P[@]} -gt 0 ]; then
  for p in "${P[@]}"; do log "CELL PROBLEM [$CELL]: $p"; done
  echo "=== VERDICT: FAIL ($CELL) ===" | tee -a "$L"
  campaign_log "FAIL s37 $CELL | bin $(sha256sum "$B" | cut -c1-16) | ${P[*]} | $L"; exit 1
fi
echo "=== VERDICT: RAN ($CELL) -- judge the fields against POD_PLAN_S37.md section 5 ===" | tee -a "$L"
campaign_log "RAN s37 $CELL | bin $(sha256sum "$B" | cut -c1-16) xbin=$XBIN | flags '$FLAGS' | store '${STOREF:-mem}' | $(head -c 300 "$O/FIELDS.txt" | tr '\n' ' ') | $L"
