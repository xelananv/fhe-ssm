#!/usr/bin/env bash
# 40_fit_and_store.sh -- THE FIT TEST + the compressed-store build, on the pod's
#                        OWN (in-process) keys, batch mode.
#
# PURPOSE   Does the demo circuit (2^17 secure, xd12, --lanes-block store) fit
#           this card, and build the store file the server will load in ~1 min
#           instead of ~1 h. --stateful exists only with --serve
#           (gpu_real_model.cu:789-791), so the batch equivalent is run:
#           T=4 tokens through the block circuit -- the scan carries across the
#           4 tokens inside the pass, which is what a stateful tick does.
# INPUTS    env per _lib.sh; FIT_TOKENS=4; STORE_MIN_BYTES=40000000000;
#           optional FIT_LANES_FULL=4 (runs --tag ${TAG}lanes4 --lanes-full 4 --
#           needs the lanes bundle from `20_stage_artifacts.sh export-lanes 4`;
#           NOT the default because the store must be built from the SAME
#           bundle the server serves (R2), and --lanes-full needs a LANES
#           bundle, gpu_real_model.cu:3073-3082).
# OUTPUTS   $DEMO/fit.log, $DEMO/fit_vram.jsonl, $STORE
# PASS      no "out of memory"/"Cuda failure" in the log AND summary
#           failed:false AND storeUncompressible 0 AND fidelity.top1Agree 1 AND
#           $STORE exists with size > STORE_MIN_BYTES. The rc is recorded and
#           ignored (F74: an OOM exits 0, lanes17_ladder.log:20-22).
# RECORD    fit.log summary (peakVramGB/procPeakVramGB/storeBytes/msPerToken),
#           fit_vram.jsonl per-stage table, the store file.
# DURATION  from records (single-lane, no lanes keys): store-build pass 3,510 s
#           (msPerToken 3.50972e+06, pod/state/runs/s31_t1b_store_secure_r17_prov_a2/
#           console.log:11, tag pbd430, CONTENDED) / 3,556 s (…/s31_t1b_store_secure_r17/
#           console.log:11, pbd430a, clean); all-hits pass 165,078 ms (prov_a2:16) /
#           171,941 ms (clean:16); setupMs 34,069 (clean:6). T=4 => ~3,556 + 3x172
#           + setup ~ 4,100 s, plus writing ~61 GB to disk => ~1.2 h.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
: "${FIT_TOKENS:=4}"; : "${STORE_MIN_BYTES:=40000000000}"
require_bin
mkdir -p "$DEMO"
L="$DEMO/fit.log"
if [ -n "${FIT_LANES_FULL:-}" ]; then
  MTAG="${TAG}lanes${FIT_LANES_FULL}"
  require_file "$ART/bundle_${MTAG}.bin" "20_stage_artifacts.sh export-lanes $FIT_LANES_FULL"
  MF=(--tag "$MTAG" --bundle-dir "$ART" --device "$DEVICE"); XF=(--lanes-full "$FIT_LANES_FULL")
  # review 2026-09-03: a lanes bundle is a DIFFERENT store (the stamp binds tag +
  # bundle content); never share $STORE with the demo server's pbd430a store.
  STORE="${STORE%.bin}_${MTAG}.bin"
  log "FIT_LANES_FULL=$FIT_LANES_FULL: store path for this run is $STORE (not the demo store)"
else
  require_file "$ART/bundle_${TAG}.bin" "20_stage_artifacts.sh"
  MF=("${MODEL_FLAGS[@]}"); XF=()
fi
busy_acquire
[ -f "$STORE" ] && log "NOTE: $STORE exists ($(file_bytes "$STORE") B) -- this run LOADS it (sample-verified) and only appends"
{
  echo "=== FIT $(utc) ==="; record_load_state
  echo "CMD: $BIN ${MF[*]} ${DEMO_FLAGS[*]} --tokens $FIT_TOKENS ${BLOCK_FLAGS[*]} ${XF[*]-} ${DEVICES_FLAGS:-} --store-file $STORE --vram-trace --sync-timers --proc-vram"
} > "$L"
gpu_idle_or_warn || echo "LOADSTATE CONTENDED" >> "$L"
T0=$(date +%s)
"$BIN" "${MF[@]}" "${DEMO_FLAGS[@]}" --tokens "$FIT_TOKENS" "${BLOCK_FLAGS[@]}" ${XF[@]+"${XF[@]}"} ${DEVICES_FLAGS:-} \
  --store-file "$STORE" --vram-trace --sync-timers --proc-vram >> "$L" 2>&1
echo "RC=$? (recorded, not trusted: F74)" >> "$L"
echo "WALL_S=$(( $(date +%s) - T0 ))" >> "$L"

# ---------------------------------------------------------------- verdict
P=()
cuda_ok "$L" || P+=("CUDA/fatal text: $(cuda_problems "$L" | tr '\n' ';')")
S=$(summary_line "$L")
[ -n "$S" ] || P+=("no {\"summary\":true} line -- no record produced")
if [ -n "$S" ]; then
  [ "$(jget "$S" failed)" = false ] || P+=("summary failed:$(jget "$S" failed)")
  [ "$(jget "$S" storeUncompressible)" = 0 ] || P+=("storeUncompressible $(jget "$S" storeUncompressible) != 0")
  python3 -c "import sys; sys.exit(0 if float('$(jget "$S" fidelity.top1Agree)' or 0) >= 1.0 else 1)" || P+=("fidelity.top1Agree $(jget "$S" fidelity.top1Agree) != 1")
fi
SB=$(file_bytes "$STORE")
[ "$SB" -gt "$STORE_MIN_BYTES" ] || P+=("store file $STORE is $SB B (want > $STORE_MIN_BYTES)")
echo "=== VRAM TRACE ===" | tee -a "$L"
vram_table "$L" "$DEMO/fit_vram.jsonl" | tee -a "$L"
echo "=== PASS TIMINGS ===" | tee -a "$L"; grep -a '"passTiming"' "$L" | cut -c1-220
grep -a '"storeLoad"\|"storeSave"\|storeVerify' "$L" | head -5
echo "=== SUMMARY ===" | tee -a "$L"; print_summary_fields "$L" | tee -a "$L"
if [ -n "$S" ]; then
  python3 - "$(jget "$S" storeBytes)" "$FIT_TOKENS" <<'PY' | tee -a "$L"
import sys
sb = float(sys.argv[1] or 0); T = int(sys.argv[2]); base = 60850962432.0   # single-token 2^17 record (clean:18)
extra = sb - base
per_tok = extra / max(T - 1, 1)
print("STORE GROWTH: storeBytes %.2f GB vs single-token record 60.85 GB -> +%.2f GB over %d extra token(s) = %.2f GB/token" % (sb/1e9, extra/1e9, T-1, per_tok/1e9))
print("  naive 64-tick projection (growth stops once every (diag,level) has been seen -- at 2^15 the encode hit 0 by tick 9, genRb server.jsonl): %.0f GB host RAM upper bound" % ((base + per_tok*63)/1e9))
PY
fi
if [ ${#P[@]} -gt 0 ]; then
  for p in "${P[@]}"; do log "FIT PROBLEM: $p"; done
  echo "=== VERDICT: FAIL (does not fit / no store) ===" | tee -a "$L"
  campaign_log "FAIL fit | ${P[*]} | wall $(grep WALL_S "$L") | $L"
  echo "If the failure is an OOM: README.md section 'If 40 OOMs' (bigger card first; the depth ladder costs a new key contract + upload)."
  busy_release; exit 1
fi
echo "=== VERDICT: PASS -- fits; store $SB B at $STORE ===" | tee -a "$L"
campaign_log "PASS fit | peakVramGB $(jget "$S" peakVramGB) procPeak $(jget "$S" procPeakVramGB) storeBytes $(jget "$S" storeBytes) storeFile $SB msPerToken $(jget "$S" msPerToken) top1 $(jget "$S" fidelity.top1Agree) harness $(jget "$S" harnessSha256) wall $(grep WALL_S "$L") | $L"
busy_release
