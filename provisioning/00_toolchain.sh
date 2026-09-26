#!/usr/bin/env bash
# Stage 0: install build toolchain + CUDA 12.9 on the bare pod.
# CUDA 12.9 chosen: Blackwell sm_120 needs >= 12.8; staying on 12.x for
# FIDESlib/OpenFHE compatibility (13.x drops some deprecated APIs).
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

# The image ships an ubuntu2004 CUDA repo but the OS is 24.04 — add the
# matching 2404 repo so we don't pull 20.04-targeted deps onto a 24.04 glibc.
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates gnupg curl wget

install -d /usr/share/keyrings
wget -qO /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
# drop the stale 20.04 list so apt resolves against 24.04 only
rm -f /etc/apt/sources.list.d/cuda.list
apt-get update -qq

apt-get install -y -qq --no-install-recommends \
  build-essential g++ gcc make cmake ninja-build git python3 python3-dev \
  libomp-dev libgomp1 autoconf automake libtool pkg-config \
  libssl-dev zlib1g-dev time

apt-get install -y -qq cuda-toolkit-12-9

echo "=== TOOLCHAIN VERSIONS ==="
/usr/local/cuda-12.9/bin/nvcc --version
cmake --version | head -1
g++ --version | head -1
git --version
nproc
echo "=== STAGE0 DONE ==="
