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

# 2026-09-17: the keyring repo must match the RUNNING release (22.04 images exist: RunPod/Lambda), and the
# toolkit version must not exceed what the DRIVER supports (nvidia-smi's "CUDA Version"): CUDA_VER=12.9 default,
# lowered to the driver's ceiling when that is a 12.x below it (12.4 is the floor rider_build_fideslib.sh accepts).
UBU="ubuntu$(. /etc/os-release; echo "${VERSION_ID//./}")"
CUDA_VER="${CUDA_VER:-12.9}"
SMI_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
if [ -n "$SMI_CUDA" ] && awk -v d="$SMI_CUDA" -v w="$CUDA_VER" 'BEGIN{exit !(d+0 < w+0 && int(d) == 12)}'; then
  echo "driver supports CUDA $SMI_CUDA < requested $CUDA_VER -> installing cuda-toolkit-${SMI_CUDA/./-} instead"
  CUDA_VER="$SMI_CUDA"
fi
echo "release $UBU, toolkit cuda-toolkit-${CUDA_VER/./-} (driver ceiling ${SMI_CUDA:-unknown})"
install -d /usr/share/keyrings
wget -qO /tmp/cuda-keyring.deb \
  "https://developer.download.nvidia.com/compute/cuda/repos/${UBU}/x86_64/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/cuda-keyring.deb
# drop the stale 20.04 list so apt resolves against 24.04 only
rm -f /etc/apt/sources.list.d/cuda.list
apt-get update -qq

apt-get install -y -qq --no-install-recommends \
  build-essential g++ gcc make cmake ninja-build git python3 python3-dev \
  libomp-dev libgomp1 autoconf automake libtool pkg-config \
  libssl-dev zlib1g-dev time

apt-get install -y -qq "cuda-toolkit-${CUDA_VER/./-}"
# s31_bootstrap.sh / _lib.sh look at /usr/local/cuda-12.9 first, then /usr/local/cuda: make sure the latter points here
[ -x /usr/local/cuda/bin/nvcc ] || ln -sfn "/usr/local/cuda-${CUDA_VER}" /usr/local/cuda

echo "=== TOOLCHAIN VERSIONS ==="
"/usr/local/cuda-${CUDA_VER}/bin/nvcc" --version
cmake --version | head -1
g++ --version | head -1
git --version
nproc
echo "=== STAGE0 DONE ==="
