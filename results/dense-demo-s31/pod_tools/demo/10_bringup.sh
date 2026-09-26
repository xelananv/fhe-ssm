#!/usr/bin/env bash
# 10_bringup.sh -- toolchain + FIDESlib main + DEMO_SER no-LTO build + torch venv,
#                  then the A-4 anchor immutability cell.
#
# PURPOSE   Reproduce the last pod's build EXACTLY by calling the same scripts
#           (never copying their bodies): pod_tools/00_toolchain.sh (if nvcc is
#           missing), pod_tools/s31_bootstrap.sh (which itself calls
#           provisioning/10_torch.sh, campaign/scripts/rider_build_fideslib.sh
#           --ref main, campaign/scripts/rider_build_demo_nolto_nccl_v2.sh), then
#           the A-4 anchor exactly as queue cell s31_anchor_a4_v3.
# INPUTS    env per _lib.sh. Subcommand: build | anchor | all (default).
#           The anchor needs $ART/bundle_agnd768b.bin (+ .index.txt) -- it is NOT
#           in the git clone (ml-eval/artifacts is untracked): ship it with
#           20_stage_artifacts.sh first, then `10_bringup.sh anchor`.
# OUTPUTS   $DEMO/build/{toolchain,bootstrap}.log (+ /root/build_fideslib.log,
#           /root/build_demo.log, /root/torch.log written by s31_bootstrap.sh),
#           $DEMO/build/BUILD_IDENTITY.txt, $DEMO/anchor_a4/fhe_gate.console.log.
# PASS      build: RIDER:harness:ok, DEMOBUILD:fideslib-nolto:ok,
#           DEMOBUILD:gpu_real_model:ok, DEMOBUILD:fideslib_gnu_lto_sections:0,
#           STAGE10 DONE (torch), $BIN executable and carrying the four new flag
#           strings (--dump-rot-indices --store-file --vram-trace evalrot.part).
#           anchor: FHEGATE:verdict:PASS AND EXACT encPtCount 2244, boots 548,
#           fidelity.top1Agree 1, worstLogitErr in [0.217, 0.219]
#           (queue pass_criteria; record s31_anchor_a4_v3/console.log:138:
#            encPtCount 2244, boots 548, top1Agree 1, worstLogitErr 0.217541,
#            msPerToken 184.878 on harnessSha256 a02fc68064c64b9c).
# RECORD    BUILD_IDENTITY.txt = the binary sha256 + FIDESlib commit + LTO
#           section count every later record must be read against (R6).
#           anchor_a4/fhe_gate.console.log = the immutability record.
# DURATIONS on record: x_build_v212 3 min, x_build_main 2 min
#           (results/explore-s29-rtxpro6000bw/pod/state/CAMPAIGN_LOG.md:43,121,124,166),
#           s31_build_demo_v2 1 min (results/dense-demo-s31/pod/state/CAMPAIGN_LOG.md:532,542),
#           anchor totalWallMs 55,116 (s31_anchor_a4_v3/console.log:138).
#           The apt CUDA toolchain install is UNTIMED on record
#           (s31_bootstrap.sh:5-9 allows it up to 60 min). Budget 30-45 min.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
MODE=${1:-all}
mkdir -p "$DEMO/build"

do_build() {
  busy_acquire
  # ---- step 0: toolchain (only if nvcc is absent; 00_toolchain.sh has set -e)
  if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda-12.9/bin/nvcc ]; then
    log "nvcc missing -> running $TOOLS_DIR/00_toolchain.sh (apt cuda-toolkit-12-9; untimed on record)"
    bash "$TOOLS_DIR/00_toolchain.sh" > "$DEMO/build/toolchain.log" 2>&1
    grep -q 'STAGE0 DONE' "$DEMO/build/toolchain.log" || { log "TOOLCHAIN FAILED -- see $DEMO/build/toolchain.log"; campaign_log "FAIL toolchain"; exit 1; }
    log "toolchain OK"
  fi
  export PATH=/usr/local/cuda-12.9/bin:/usr/local/cuda/bin:$PATH
  command -v rsync >/dev/null 2>&1 || { export DEBIAN_FRONTEND=noninteractive; apt-get install -y -qq rsync >/dev/null 2>&1 || log "WARN: rsync install failed (20 needs it on this side)"; }
  # ---- step 1: the exact S3.1 bootstrap chain
  if [ "$POD_SRC" = /root/src ] && [ "$WORK" = /root/fhe-main-demo ]; then
    log "running $TOOLS_DIR/s31_bootstrap.sh (venv + fideslib main + demo build; logs in /root/*.log)"
    bash "$TOOLS_DIR/s31_bootstrap.sh" > "$DEMO/build/bootstrap.log" 2>&1
    FL=/root/build_fideslib.log; DL=/root/build_demo.log; TL=/root/torch.log
  else
    log "non-default POD_SRC/WORK -> calling the riders directly with --work $WORK --repo $POD_SRC"
    ( bash "$POD_SRC/provisioning/10_torch.sh" > "$DEMO/build/torch.log" 2>&1 ) &
    TP=$!
    bash "$POD_SRC/campaign/scripts/rider_build_fideslib.sh" --ref main --work "$WORK" --repo "$POD_SRC" > "$DEMO/build/build_fideslib.log" 2>&1
    bash "$POD_SRC/campaign/scripts/rider_build_demo_nolto_nccl_v2.sh" --work "$WORK" --repo "$POD_SRC" > "$DEMO/build/build_demo.log" 2>&1
    wait $TP || true
    FL=$DEMO/build/build_fideslib.log; DL=$DEMO/build/build_demo.log; TL=$DEMO/build/torch.log
  fi
  # ---- gates: markers, never exit codes
  P=()
  grep -q 'RIDER:harness:ok' "$FL" || P+=("RIDER:harness:ok missing ($FL)")
  grep -q 'DEMOBUILD:fideslib-nolto:ok' "$DL" || P+=("DEMOBUILD:fideslib-nolto:ok missing ($DL)")
  grep -q 'DEMOBUILD:gpu_real_model:ok' "$DL" || P+=("DEMOBUILD:gpu_real_model:ok missing ($DL)")
  grep -q 'DEMOBUILD:fideslib_gnu_lto_sections:0' "$DL" || P+=("fideslib.a is not LTO-free (DEMOBUILD:fideslib_gnu_lto_sections != 0)")
  grep -q 'STAGE10 DONE' "$TL" || P+=("torch venv: STAGE10 DONE missing ($TL)")
  [ -x "$BIN" ] || P+=("binary missing: $BIN")
  "$VENV/bin/python" -c 'import torch, transformers, numpy' 2>/dev/null || P+=("venv import torch/transformers/numpy failed")
  for s in --dump-rot-indices --store-file --vram-trace evalrot.part; do
    grep -qa -- "$s" "$BIN" 2>/dev/null || P+=("flag string '$s' NOT in the binary -- the new harness flags did not land (the 2026-09-02 parser makes an unknown flag FATAL, so a stale binary would refuse every new flag loudly rather than ignore it)")
  done
  {
    echo "BUILD_IDENTITY $(utc)"
    echo "binary $BIN"
    echo "binary_sha256 $(sha256sum "$BIN" 2>/dev/null | cut -c1-64)"
    echo "binary_x $WORK/build-demo/gpu_real_model_x"
    echo "binary_x_sha256 $(sha256sum "$WORK/build-demo/gpu_real_model_x" 2>/dev/null | cut -c1-64)"
    echo "fideslib_commit $(cat "$WORK/FIDESLIB_COMMIT.txt" 2>/dev/null)"
    echo "fideslib_a_gnu_lto_sections $(readelf -S "$WORK/install/lib/fideslib.a" 2>/dev/null | grep -c 'gnu\.lto')"
    echo "nvcc $(nvcc --version 2>/dev/null | tail -1)"
    echo "cuda_arch $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr -d '.' | paste -sd ';' -)"
    echo "repo_commit $(git -C "$POD_SRC" rev-parse HEAD 2>/dev/null) dirty=$(git -C "$POD_SRC" status --porcelain 2>/dev/null | grep -c . )"
  } | tee "$DEMO/build/BUILD_IDENTITY.txt"
  if [ ${#P[@]} -gt 0 ]; then
    for p in "${P[@]}"; do log "BUILD PROBLEM: $p"; done
    campaign_log "FAIL build | ${P[*]}"
    busy_release; exit 1
  fi
  log "BUILD OK -- $(grep binary_sha256 "$DEMO/build/BUILD_IDENTITY.txt")"
  campaign_log "PASS build | $(grep -E 'binary_sha256|fideslib_commit|gnu_lto' "$DEMO/build/BUILD_IDENTITY.txt" | tr '\n' ' ')"
  busy_release
}

do_anchor() {
  require_bin
  if [ ! -s "$ART/bundle_agnd768b.bin" ] || [ ! -s "$ART/bundle_agnd768b.index.txt" ]; then
    log "anchor bundle missing: $ART/bundle_agnd768b.bin -- ship it with 20_stage_artifacts.sh (print -> rsync -> verify), then: $SCRIPT anchor"
    campaign_log "SKIP anchor | bundle_agnd768b.bin not staged"
    return 2
  fi
  busy_acquire
  # ANCHOR_EXTRA (2026-09-17): S3.7 V1 puts the RMSNorm eps back into the circuit BY DEFAULT, so the default path
  # is no longer the recorded chain; the immutability record (worstLogitErr band) is reproduced with
  # ANCHOR_EXTRA=--no-norm-eps (V1_eps/REPORT.md). boots/encPtCount/top1 are exact either way. ANCHOR_OUT names the dir.
  OUT="$DEMO/${ANCHOR_OUT:-anchor_a4}"; mkdir -p "$OUT"
  record_load_state | tee "$OUT/loadstate.txt"
  gpu_idle_or_warn || true
  # EXACT argv of queue cell s31_anchor_a4_v3 (campaign/queues/dense_demo_s31.json)
  bash "$POD_SRC/campaign/scripts/fhe_gate_v2.sh" --require-top1 0.99 --out "$OUT" -- \
    "$BIN" --tag agnd768b --bundle-dir "$ART" --device "$DEVICE" --log-ring 15 --level-budget 3 3 \
    --ms-norm 5.5 --enc-threads 24 --boot-floor 3 --matvec-margin 4 --interleave --extra-depth 9 \
    --tokens 128 --pack-tokens 16 ${ANCHOR_EXTRA:-} > "$OUT/gate.stdout" 2>&1
  L="$OUT/fhe_gate.console.log"
  S=$(summary_line "$L")
  P=()
  grep -q 'FHEGATE:verdict:PASS' "$OUT/gate.stdout" || P+=("fhe_gate_v2 verdict not PASS: $(grep FHEGATE:problem "$OUT/gate.stdout" | tr '\n' ';')")
  EP=$(jget "$S" encPtCount); BO=$(jget "$S" boots); T1=$(jget "$S" fidelity.top1Agree); WL=$(jget "$S" worstLogitErr)
  [ "$EP" = 2244 ] || P+=("encPtCount $EP != 2244")
  [ "$BO" = 548 ] || P+=("boots $BO != 548")
  python3 -c "import sys; sys.exit(0 if float('${T1:-0}') >= 1.0 else 1)" || P+=("top1Agree $T1 != 1")
  python3 -c "import sys; w=float('${WL:-9}'); sys.exit(0 if 0.217 <= w <= 0.219 else 1)" || P+=("worstLogitErr $WL outside [0.217,0.219] (records 0.217001 / 0.219085 / 0.217541)")
  echo "ANCHOR record: $(jget "$S" harnessSha256) commit $(jget "$S" harnessCommit) build $(jget "$S" buildUtc) msPerToken $(jget "$S" msPerToken) (a02fc68064c64b9c gave 184.878 -- R6: calibration, not a verdict)"
  print_summary_fields "$L"
  if [ ${#P[@]} -gt 0 ]; then
    for p in "${P[@]}"; do log "ANCHOR PROBLEM: $p"; done
    log "STOP: the unflagged path is not bit-identical -- fix the harness before anything is timed (queue notes for s31_anchor_a4_v3)."
    campaign_log "FAIL anchor_a4 | ${P[*]} | $L"
    busy_release; return 1
  fi
  log "ANCHOR PASS (encPtCount 2244, boots 548, top1 1, worstLogitErr $WL)"
  campaign_log "PASS anchor_a4 | encPtCount $EP boots $BO top1 $T1 worstLogitErr $WL msPerToken $(jget "$S" msPerToken) harness $(jget "$S" harnessSha256) | $L"
  busy_release
}

case "$MODE" in
  build)  do_build ;;
  anchor) do_anchor ;;
  all)    do_build && do_anchor ;;
  *) echo "usage: $SCRIPT [build|anchor|all]"; exit 2 ;;
esac
