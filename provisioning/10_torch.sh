#!/usr/bin/env bash
# Stage 10: python/torch environment for the ml-eval tooling.
#
# CLOSES DEFERRED_WORK.md I-9. The original 10_torch.sh was never harvested
# from the retired pod, but its complete `set -x` execution trace survives in
# results/a100-secure-20260730/training/10_torch.log AND
# results/gpu-verify-20260728/logs/10_torch.log (identical); this file is that
# trace reconstructed 1:1, plus the `datasets zstandard` layer the A100 chain
# added separately (recovered/pod-scripts/11_chain_a100.sh:31 — required for
# FineWeb-Edu streaming in train_fhe_native_ssm*.py), plus an HF_TOKEN hook.
#
# Known-good resolved snapshot from the two recorded runs (2026-07-28 RTX PRO
# 5000 sm_120, 2026-07-30 A100 sm_80): torch 2.11.0+cu128 on python 3.12.3,
# pip 24.0, Ubuntu 24.04. No version pins exist anywhere in the repo (verified
# 2026-08-13); the cu128 wheel needs a >=12.8-capable DRIVER, independent of
# the toolkit version. Venv is ALWAYS /root/venv — ~25 recovered pod launchers
# hardcode /root/venv/bin/{python,torchrun,pip}; do not change it.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get install -y -qq --no-install-recommends python3-pip python3-venv
python3 -m venv /root/venv
/root/venv/bin/pip install -q --upgrade pip
/root/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
/root/venv/bin/pip install -q transformers pyarrow huggingface_hub numpy
# streaming leg (was layered on by 11_chain_a100.sh, folded in here):
/root/venv/bin/pip install -q datasets zstandard

# HF auth: the datamix pilot ran unauthenticated and was rate-limit-warned
# (campaign-sessions/S2.5_l40s_infrastructure.md §6). Put the token in
# /root/.hf_token (mode 600, NEVER in the repo or any log) before staging.
if [ -f /root/.hf_token ]; then
  chmod 600 /root/.hf_token
  echo "HF token present at /root/.hf_token (export HF_TOKEN=\$(cat /root/.hf_token) in run env)"
else
  echo "WARNING: /root/.hf_token missing — HF downloads will run anonymous and rate-limited"
fi

echo '=== TORCH CHECK ==='
/root/venv/bin/python -c '
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
x=torch.randn(4096,4096,device="cuda"); y=x@x; torch.cuda.synchronize()
print("matmul ok", float(y.float().abs().mean()))
'
echo '=== STAGE10 DONE ==='
