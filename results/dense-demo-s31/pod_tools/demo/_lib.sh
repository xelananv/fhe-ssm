#!/usr/bin/env bash
# _lib.sh -- shared defaults + helpers for the demo-pod scripts. SOURCED, never run.
#
# PURPOSE   one place for the env contract, the marker-file mutex, the
#           campaign-log line, and the log-grepping verdict helpers every
#           script uses. Nothing here touches the GPU.
# INPUTS    environment overrides (export before calling any script):
#             POD_SRC=/root/src            repo clone on the pod
#             WORK=/root/fhe-main-demo     FIDESlib main + install prefix
#             BIN=$WORK/build-demo/gpu_real_model   the DEMO_SER no-LTO binary
#             ART=$POD_SRC/ml-eval/artifacts        bundles / weights / configs
#             KEYS=/root/mac_keys_r17      Mac-minted EVAL material (no secret.key)
#             KEYS_POD=$DEMO/keys_pod_r17  pod-minted keys (secret included; 40/60 only)
#             DEMO=/root/demo              every log / record of the day
#             STORE=$DEMO/store_pbd430a_r17_xd12_lanesblock.bin  compressed store file
#             DEVICES=            two-card route (2026-09-03, MGPU_FIT_MODEL_20260903.md):
#                                 export DEVICES=0,1 and every script adds
#                                 DEVICES_FLAGS="--devices 0,1 --store-multi-gpu" to the
#                                 harness and points STORE at the _mgpu file that
#                                 41_mgpu_ladder.sh rung 2 builds (STORE_MGPU). Route is
#                                 live only past rung 3; rung 0 = peer_pool_probe.
#             TAG=pbd430a  VENV=/root/venv  DEVICE=0  ENC_THREADS=24
#             LOG_RING=17  EXTRA_DEPTH=12  BOOT_FLOOR=3
# OUTPUTS   $DEMO/CAMPAIGN_LOG.md (one UTC line per script run), $DEMO/.busy marker
# RULES     F74: a CUDA OOM exits 0 -- every verdict greps the log, never $?.
#           T7 : never pkill/pgrep -f a pattern that is in your own command line;
#                mutual exclusion is the $DEMO/.busy marker file, stopping is a
#                `stop` file or `kill <pid-from-pidfile>`.
#           R6 : record the load state (uptime, GPU compute apps) beside every
#                timing; never compare ms across binaries or load states.
#
# DEMO_FLAGS = the proven 2^17 secure flag set, transcribed from
#   campaign/queues/dense_demo_s31.json  cell s31_t1b_store_secure_r17 "cmd"
#   (--log-ring 17 --secure --level-budget 3 3 --ms-norm 5.5 --enc-threads 24
#    --boot-floor 3 --matvec-margin 4 --extra-depth 12) -- the record behind it is
#   results/dense-demo-s31/pod/state/runs/s31_t1b_store_secure_r17/console.log.
#   (sessions/final/demo_smoke/fid17_run.log:2 used --boot-floor 4 with --enc-threads
#    12 on a 2-GPU box; the demo value is the queue's 3 / 24.)
# BLOCK_FLAGS = the D14 block layout: --pack-tokens 1 --lanes-block --compressed-store
#   (--stateful requires --pack-tokens 1, gpu_real_model.cu:783-788; --lanes-block
#    requires --compressed-store and no --interleave, :836-840).

: "${POD_SRC:=/root/src}"
: "${WORK:=/root/fhe-main-demo}"
: "${BIN:=$WORK/build-demo/gpu_real_model}"
: "${ART:=$POD_SRC/ml-eval/artifacts}"
: "${KEYS:=/root/mac_keys_r17}"
: "${DEMO:=/root/demo}"
: "${KEYS_POD:=$DEMO/keys_pod_r${LOG_RING:-17}}"
# 2026-09-17 (one-card session 1 at 2^15): the ring, depth and block shape are env (LOG_RING/EXTRA_DEPTH/LANES_BLOCK,
# below); the default store and pod-key names follow them so a 2^15 plain-shape store never collides with the 2^17
# demo store (at the defaults the names are exactly the 2026-09-04 ones). SECURE=0 drops --secure (2^15 is the
# STRUCTURAL class: at depth 38 HEStd_128_classic would move the ring; 85_schedule_ab.sh quick never passed it).
: "${LOG_RING:=17}"; : "${EXTRA_DEPTH:=12}"; : "${SECURE:=1}"; : "${LANES_BLOCK:=1}"
if [ "$LANES_BLOCK" = 1 ]; then _SHAPE=lanesblock; else _SHAPE=plain; fi
: "${STORE_SINGLE:=$DEMO/store_pbd430a_r${LOG_RING}_xd${EXTRA_DEPTH}_${_SHAPE}.bin}"
: "${STORE_MGPU:=${STORE_SINGLE%.bin}_mgpu.bin}"   # what 41 rung 2 builds and 50/60/70 load
: "${DEVICES:=}"
case "$DEVICES" in
  *,*) : "${DEVICES_FLAGS:=--devices $DEVICES --store-multi-gpu}"; : "${STORE:=$STORE_MGPU}"
       # 2026-09-04 (pod O-2086900, PCIe NODE links): FIDESlib's default cross-card
       # exchange (cudaMemcpyPeerAsync) returns garbage on this box -- 2-device 2^17
       # selftest: rotate err NaN, then a lockstep hang. With the exchange on NCCL and
       # NCCL forced off P2P (shared host memory) the same selftest passes: rotate
       # 1.16e-10, matvec 1.13e-11, bootstrap 1.79e-3 (/root/demo/mgpu_selftest_r17_nccl.log).
       export FIDESLIB_USE_MEMCPY_PEER="${FIDESLIB_USE_MEMCPY_PEER:-0}" NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" ;;
  *)   : "${DEVICES_FLAGS:=}"; : "${STORE:=$STORE_SINGLE}" ;;
esac
: "${TAG:=pbd430a}"
: "${VENV:=/root/venv}"
: "${DEVICE:=0}"
: "${ENC_THREADS:=24}"
: "${LOG_RING:=17}"
: "${EXTRA_DEPTH:=12}"
: "${BOOT_FLOOR:=3}"
: "${SCRIPT:=$(basename "${BASH_SOURCE[1]:-$0}")}"
: "${TOOLS_DIR:=$POD_SRC/results/dense-demo-s31/pod_tools}"
: "${DEMO_TOOLS:=$TOOLS_DIR/demo}"

export LD_LIBRARY_PATH="$WORK/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v nvcc >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:$PATH"

DEMO_FLAGS=(--log-ring "$LOG_RING")
[ "$SECURE" = 1 ] && DEMO_FLAGS+=(--secure)
DEMO_FLAGS+=(--level-budget 3 3 --ms-norm 5.5 --enc-threads "$ENC_THREADS" --boot-floor "$BOOT_FLOOR" --matvec-margin 4
             --extra-depth "$EXTRA_DEPTH")
# EXTRA_FLAGS (2026-09-04): extra harness flags appended to every DEMO_FLAGS call, e.g. EXTRA_FLAGS=--canonical-carry
if [ -n "${EXTRA_FLAGS:-}" ]; then read -r -a _EXTRA <<< "$EXTRA_FLAGS"; DEMO_FLAGS+=("${_EXTRA[@]}"); fi
MODEL_FLAGS=(--tag "$TAG" --bundle-dir "$ART" --device "$DEVICE")
# LANES_BLOCK=0 (2026-09-04): the single-lane (broadcast) demo shape -- block layout WITHOUT the
# dual-half masks; it loads the plain store (STORE must point at it).
if [ "${LANES_BLOCK:-1}" = 1 ]; then BLOCK_FLAGS=(--pack-tokens 1 --lanes-block --compressed-store)
else BLOCK_FLAGS=(--pack-tokens 1 --compressed-store); fi
# depth = 10 + 19 + extraDepth (mac_fhe_client.cpp:83; record "depth":41 at xd12 in
# s31_t1b_store_secure_r17/console.log:18)
EXPECT_DEPTH=$((10 + 19 + EXTRA_DEPTH))

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(utc)] $*"; }
campaign_log() { mkdir -p "$DEMO"; echo "$(utc) | $SCRIPT | $*" >> "$DEMO/CAMPAIGN_LOG.md"; }

# ---- marker-file mutex (T7: never a pgrep pattern) --------------------------
BUSY_HELD=0
busy_release() { if [ "$BUSY_HELD" = 1 ]; then rm -f "$DEMO/.busy"; BUSY_HELD=0; fi; }
busy_acquire() {
  mkdir -p "$DEMO"
  local B="$DEMO/.busy"
  if [ -f "$B" ]; then
    if [ "${BUSY_WAIT:-0}" = 1 ]; then
      log "waiting for $B to clear (held: $(cat "$B"))"
      while [ -f "$B" ]; do sleep 30; done
      sleep 20
    else
      log "REFUSED: $B is held ($(cat "$B")). Finish/stop that step, or BUSY_WAIT=1 to queue."
      exit 3
    fi
  fi
  echo "$SCRIPT pid=$$ since=$(utc)" > "$B"
  BUSY_HELD=1
  trap busy_release EXIT
}

# ---- verdict helpers (F74: read the console, never the rc) ------------------
CUDA_BAD_RE='out of memory|Cuda failure|cudaError|CUDA_ERROR|map::at|out_of_range|terminate called|"fatal"'
cuda_ok() { ! grep -aqiE "$CUDA_BAD_RE" "$1"; }
cuda_problems() { grep -aoiE "($CUDA_BAD_RE)[^\"]{0,80}" "$1" | sort | uniq -c | head -5; }
summary_line() { grep -a '^{"summary":true' "$1" | tail -1; }
# jget '<json line>' key.subkey  -> value or empty (python3 is on every pod image)
jget() {
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    print(""); sys.exit(0)
v = d
for k in sys.argv[2].split("."):
    v = v.get(k) if isinstance(v, dict) else None
if isinstance(v, bool): print(str(v).lower())
else: print("" if v is None else v)
PY
}
require_file() { [ -s "$1" ] || { log "MISSING: $1  ($2)"; exit 2; }; }
require_bin() { [ -x "$BIN" ] || { log "MISSING binary $BIN -- run 10_bringup.sh first"; exit 2; }; }
file_bytes() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0; }

# R6: the load state beside every timing.
record_load_state() {
  echo "LOADSTATE $(utc) uptime: $(uptime | sed 's/^ *//')"
  echo "LOADSTATE gpu compute apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "LOADSTATE busy marker: $(cat "$DEMO/.busy" 2>/dev/null || echo none)"
}
gpu_idle_or_warn() {
  local apps; apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${apps:-0}" -gt 0 ]; then
    log "WARNING: $apps GPU compute process(es) already running -- a timing taken now is CONTENDED (MEASUREMENTS.md 1d, 19.7x swing). Recorded as such."
    return 1
  fi
  return 0
}

# vram_table LOG OUT.jsonl : pull every {"vramTrace":...} line and print a table
vram_table() {
  grep -a '"vramTrace"' "$1" > "$2" 2>/dev/null || true
  python3 - "$2" <<'PY'
import json, sys
rows = []
for l in open(sys.argv[1]):
    l = l.strip()
    if not l: continue
    try: rows.append(json.loads(l))
    except Exception: pass
if not rows:
    print("  (no vramTrace lines -- was --vram-trace passed, and does this binary implement it? 10_bringup.sh checks the flag string)")
    sys.exit(0)
print("  %-32s %4s %10s %10s" % ("stage", "dev", "usedGB", "freeGB"))
for r in rows:
    print("  %-32s %4s %10s %10s" % (r.get("vramTrace"), r.get("dev"), r.get("usedGB"), r.get("freeGB")))
try:
    print("  peak usedGB over the trace: %.3f" % max(float(r.get("usedGB") or 0) for r in rows))
except Exception:
    pass
PY
}

# print_summary_fields LOG : the fields every record reader wants, from the summary line
print_summary_fields() {
  local S; S=$(summary_line "$1")
  [ -n "$S" ] || { echo "  (no {\"summary\":true} line)"; return 1; }
  python3 - "$S" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
keys = ("tag","tokens","packTokens","secure","ringDim","slots","depth","extraDepth","boots","encPtCount",
        "encPtMs","msPerToken","wallMsPerToken","totalWallMs","setupMs","peakVramGB","procPeakVramGB",
        "storeOn","storeHits","storeMisses","storeDead","storeUncompressible","storeBytes","failed",
        "worstLogitErr","harnessSha256","harnessCommit","buildUtc")
print("  " + json.dumps({k: d.get(k) for k in keys}))
print("  fidelity: " + json.dumps(d.get("fidelity")))
PY
}
