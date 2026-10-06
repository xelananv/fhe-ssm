#!/bin/bash
# S3.1 bootstrap: wait for toolchain, then venv + fideslib(main) + demo build.
set -uo pipefail
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
# 2026-09-17 (session-1 prep): the toolchain may be a preinstalled CUDA at /usr/local/cuda (RunPod/Lambda images)
# or the apt cuda-toolkit-12-9 of 00_toolchain.sh; wait for ANY nvcc, never only the 12.9 path (a 60-min dead wait).
CUDA_BIN=""
for i in $(seq 1 120); do
  CUDA_BIN=""
  for c in "${CUDA_HOME:-/nonexistent}/bin" /usr/local/cuda-12.9/bin /usr/local/cuda-12.8/bin /usr/local/cuda-12.6/bin /usr/local/cuda/bin; do
    [ -x "$c/nvcc" ] && { CUDA_BIN=$c; break; }
  done
  [ -z "$CUDA_BIN" ] && command -v nvcc >/dev/null 2>&1 && CUDA_BIN=$(dirname "$(command -v nvcc)")
  if [ -n "$CUDA_BIN" ] && ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then break; fi
  sleep 30
done
log "toolchain ready (nvcc at ${CUDA_BIN:-MISSING}: $($CUDA_BIN/nvcc --version 2>/dev/null | tail -1))"
export PATH=${CUDA_BIN:-/usr/local/cuda/bin}:$PATH
( bash /root/src/provisioning/10_torch.sh > /root/torch.log 2>&1 \
    && log TORCH-OK || log TORCH-FAIL ) &
TORCH_PID=$!
log "fideslib build starting (ref main -> /root/fhe-main-demo)"
bash /root/src/campaign/scripts/rider_build_fideslib.sh --ref main \
  --work /root/fhe-main-demo --repo /root/src > /root/build_fideslib.log 2>&1
if grep -q "RIDER:harness:ok" /root/build_fideslib.log; then
  log "fideslib+harness OK"
else
  log "FIDESLIB BUILD FAILED -- see build_fideslib.log"
fi
log "demo (no-LTO DEMO_SER) build starting"
bash /root/src/campaign/scripts/rider_build_demo_nolto_nccl_v2.sh \
  --work /root/fhe-main-demo --repo /root/src > /root/build_demo.log 2>&1
grep -E "DEMOBUILD:[a-z_]+:(ok|fail)" /root/build_demo.log | while read -r l; do log "$l"; done
wait $TORCH_PID || true
log "BOOTSTRAP DONE"
