#!/bin/bash
# 41_mgpu_ladder.sh -- the TWO-CARD route, as a ladder that records where it
# dies (2026-09-03). The author cannot rent a 141 GB card, so --devices 0,1 is
# the demo route, not the fallback. The 2026-09-01 two-card attempt OOMed with
# no stage line (sessions/final/demo_smoke/mgpu17.log); every rung here prints
# per-device VRAM at each milestone (--vram-trace) and is judged by the log,
# never by $? (F74).
#
# RUNGS (each ~one setup + one short pass; stop at the first red rung):
#   0  peer_pool_probe        : can a kernel on device 0 write device 1's pool
#                               memory once cudaMemPoolSetAccess is granted?
#                               (FIDESlib limbs are cudaMallocAsync memory.)
#   1  keys+precomp, no store : --devices 0,1, T=1, no --compressed-store --
#                               does the sharded context itself fit two cards?
#   2  + store, single lane   : --compressed-store --store-multi-gpu
#                               --store-file $STORE_MGPU, T=1 (builds the base
#                               store; global-id limb walk; peer writes)
#   3  + lanes-block, T=4     : the demo circuit shape (dual-half masks)
#   4  serve smoke            : --serve --stateful, 4 lanes, 3 ticks via the
#                               pod-side client (keys-load on the pod's own
#                               keys; the Mac keys come in 50/60/70)
# PASS per rung: no "out of memory"/"Cuda failure"/fatal in the log AND
#   summary failed:false AND (rungs 2-4) top1Agree 1 / relLogitErrRmsMean in
#   the single-device class (0.0374048 at pbd430a, s31_t1b_store_secure_r17/
#   console.log:17) -- a permuted or peer-broken store shows up HERE, as bad
#   fidelity fields, never as a clean number.
# RECORD: $DEMO/mgpu_rung{0..4}.log, $DEMO/mgpu_vram.jsonl (all vramTrace lines)
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
[ -n "${DEVICES:-}" ] || DEVICES=0,1      # STORE_MGPU / DEVICES_FLAGS come from _lib.sh (2026-09-03)
: "${RUNG:=all}"
require_bin
mkdir -p "$DEMO"
MGF=(--tag "$TAG" --bundle-dir "$ART" --device "${DEVICES%%,*}" --devices "$DEVICES")
judge_rung() {   # judge_rung LOG NAME want_fidelity
  local L=$1 NAME=$2 FID=$3; local P=()
  cuda_ok "$L" || P+=("CUDA/fatal text: $(cuda_problems "$L" | tr '\n' ';')")
  local SL; SL=$(summary_line "$L")
  [ -n "$SL" ] || P+=("no summary line")
  if [ -n "$SL" ]; then
    [ "$(jget "$SL" failed)" = "false" ] || P+=("summary failed:true")
    if [ "$FID" = 1 ]; then
      local T1 RE; T1=$(jget "$SL" fidelity.top1Agree 2>/dev/null || echo "?"); RE=$(jget "$SL" fidelity.relLogitErrRmsMean 2>/dev/null || echo "?")
      python3 -c "import sys; t=float('$T1' or 0); r=float('$RE' or 1); sys.exit(0 if t>=0.999 and r<0.06 else 1)" 2>/dev/null \
        || P+=("fidelity top1Agree=$T1 relLogitErrRmsMean=$RE outside the single-device class (1.0 / <=0.06)")
    fi
  fi
  grep -a '"vramTrace"' "$L" >> "$DEMO/mgpu_vram.jsonl"
  echo "--- per-device VRAM at each milestone ($NAME):"; grep -a '"vramTrace"' "$L" | python3 -c "
import sys,json
for ln in sys.stdin:
    try: d=json.loads(ln)
    except Exception: continue
    print('  %-22s dev%d used %6.1f GB free %6.1f GB' % (d['vramTrace'], d['dev'], d['usedGB'], d['freeGB']))" | tail -24
  if [ ${#P[@]} -gt 0 ]; then for p in "${P[@]}"; do log "PROBLEM [$NAME]: $p"; done; campaign_log "FAIL $NAME | ${P[*]} | $L"; return 1; fi
  campaign_log "PASS $NAME | $L"; return 0
}
run_rung() {   # run_rung NAME want_fidelity -- harness args...
  local NAME=$1 FID=$2; shift 2
  local L="$DEMO/mgpu_$NAME.log"
  busy_acquire
  { echo "=== $NAME $(utc) ==="; record_load_state; echo "CMD: $BIN $*"; } > "$L"
  gpu_idle_or_warn || echo "LOADSTATE CONTENDED" >> "$L"
  local T0; T0=$(date +%s)
  "$BIN" "$@" >> "$L" 2>&1; echo "RC=$? (recorded, not trusted: F74)" >> "$L"
  echo "WALL_S=$(( $(date +%s) - T0 ))" >> "$L"
  busy_release
  judge_rung "$L" "$NAME" "$FID"
}
rung0() {
  local P="$WORK/build-demo/peer_pool_probe" L="$DEMO/mgpu_rung0_peerpool.log"
  if [ ! -x "$P" ]; then
    log "building peer_pool_probe (nvcc -arch=native)"
    ( cd "$WORK/build-demo" && nvcc -arch=native -O2 -o peer_pool_probe "$POD_SRC/hpc_gpu_port/peer_pool_probe.cu" ) > "$L" 2>&1 || { log "peer_pool_probe build failed: $L"; campaign_log "FAIL rung0 build | $L"; return 1; }
  fi
  "$P" >> "$L" 2>&1; echo "RC=$?" >> "$L"
  if grep -q '"storePathViable":true' "$L"; then log "rung0 PASS: $(grep -o '{"peerPoolProbe".*}' "$L")"; campaign_log "PASS rung0 peerpool | $(grep -o '"withoutPoolAccess":"[a-z-]*","withPoolAccess":"[a-z-]*"' "$L")"; return 0; fi
  log "rung0 FAIL: $(tail -2 "$L" | tr '\n' ' ')"; campaign_log "FAIL rung0 peerpool | $L"; return 1
}
rung1() { run_rung rung1_nostore 1 "${MGF[@]}" "${DEMO_FLAGS[@]}" --tokens 1 --pack-tokens 1 --vram-trace --sync-timers --proc-vram; }
# 2026-09-04 (pod O-2086900): the store stamp carries the layout flags, so a store
# built WITHOUT --lanes-block is refused by every lanes-block run (rung 3 died on
# "built for a DIFFERENT configuration" after a 54-min pass). Rung 2 now builds the
# DEMO store (BLOCK_FLAGS = --pack-tokens 1 --lanes-block --compressed-store) so
# rungs 3/4, 60 and 70 all load the same file.
rung2() { run_rung rung2_store 1 "${MGF[@]}" "${DEMO_FLAGS[@]}" --tokens 1 "${BLOCK_FLAGS[@]}" --store-multi-gpu --store-file "$STORE_MGPU" --vram-trace --sync-timers --proc-vram; }
rung3() { run_rung rung3_lanesblock 1 "${MGF[@]}" "${DEMO_FLAGS[@]}" --tokens 4 "${BLOCK_FLAGS[@]}" --store-multi-gpu --store-file "$STORE_MGPU" --vram-trace --sync-timers --proc-vram; }
rung4() {
  # serve smoke on the pod's own keys: needs a --keys-save set at the demo flags
  local K="$DEMO/podkeys_r17"
  if [ ! -s "$K/evalrot.bin" ] && [ -z "$(ls "$K"/evalrot.part*.bin 2>/dev/null)" ]; then
    log "rung4: minting the pod's own key set into $K (~46.5 GB, minutes)"
    mkdir -p "$K"
    # --store-multi-gpu: BLOCK_FLAGS carry --compressed-store, which the harness refuses under --devices without it
    run_rung rung4_keygen 0 "${MGF[@]}" "${DEMO_FLAGS[@]}" --selftest --self-d 1024 --self-dff 4096 --keys-dir "$K" --keys-save "${BLOCK_FLAGS[@]}" --store-multi-gpu || return 1
  fi
  log "rung4: serve smoke = 60_fidelity_gate.sh on the two-card server (pod keys, 4 lanes, ${FID_TICKS:-3} ticks)"
  DEVICES_FLAGS="--devices $DEVICES --store-multi-gpu" KEYS_POD="$K" STORE="$STORE_MGPU" FID_TICKS="${FID_TICKS:-3}" bash "$(dirname "$(readlink -f "$0")")/60_fidelity_gate.sh"
}
: > "$DEMO/mgpu_vram.jsonl"
case "$RUNG" in
  0) rung0;; 1) rung1;; 2) rung2;; 3) rung3;; 4) rung4;;
  all) rung0 && rung1 && rung2 && rung3 && rung4 && log "=== TWO-CARD ROUTE: ALL RUNGS PASS ===" || { log "=== TWO-CARD ROUTE: stopped at a red rung; read $DEMO/mgpu_vram.jsonl for the last milestone per device ==="; exit 1; };;
  *) echo "usage: $SCRIPT (RUNG=0|1|2|3|4|all)"; exit 2;;
esac
