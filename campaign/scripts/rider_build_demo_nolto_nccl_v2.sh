#!/usr/bin/env bash
# ON-RUN-AUTHORED 2026-08-23 by strategist — purpose: fix —
# re: CAMPAIGN_LOG.md 2026-08-23T00:05:00Z — delta vs rider_build_demo_nolto_nccl.sh:
# adds `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DCMAKE_CXX_FLAGS=-fno-lto` to the
# FIDESlib configure, plus a hard post-install assertion that the archive really
# has zero .gnu.lto sections.
#
# WHY v1 WAS WRONG, and it was my error. v1 is called "nolto" and never disables
# LTO. I authored it by copying rider_build_mgpu.sh's cmake block and DROPPED the
# two flags that do the work (rider_build_mgpu.sh:34). FIDESlib turns LTO on by
# itself -- CMakeLists.txt:164-168 runs check_ipo_supported() and sets
# CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE when supported -- so without those flags
# a FRESH configure produces an LTO archive.
#
# v1 nevertheless SUCCEEDED on /root/fhe-v212, by accident: build-nolto already
# existed there, created by rider_build_mgpu.sh, whose cache carries
# CMAKE_INTERPROCEDURAL_OPTIMIZATION:UNINITIALIZED=OFF. v1 inherited that cache
# and never had to set the flag itself. On the fresh /root/fhe-main-demo prefix
# there was no cache to inherit, so it installed a 12,910-section LTO fideslib.a,
# marked DEMOBUILD:fideslib-nolto:ok anyway, and the DEMO_SER link then died with
#   lto1: internal compiler error: in add_symbol_to_partition_1, at lto/lto-partition.cc:152
# Measured, both prefixes, same libraries:
#   /root/fhe-main-demo/install/lib/fideslib.a   gnu.lto = 12910
#   /root/fhe-v212/install/lib/fideslib.a        gnu.lto = 0
# (the three libOPENFHE*_static.a are 0 on both, so FIDESlib was the only source).
#
# INVALIDATES NOTHING. The one binary v1 produced,
# /root/fhe-v212/build-demo/gpu_real_model, is genuinely no-LTO -- verified by the
# 0-section reading above, not by trusting v1 -- so the quiet-trio demo
# measurement (193.501 ms/token) and every build-demo record stand.
#
# The assertion is the real lesson: a step that reports ok while doing the
# opposite is this repo's instruments-that-lie class, so v2 checks the artifact
# rather than the directory name.
set -uxo pipefail
WORK=""; REPO=""
while [ $# -gt 0 ]; do case "$1" in
  --work) WORK=$2; shift 2;; --repo) REPO=$2; shift 2;; *) shift;; esac; done
: "${WORK:?--work required}"; : "${REPO:?--repo required}"
mark() { echo "DEMOBUILD:$1:$2"; }
die() { mark "$1" fail; exit 1; }
command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || die nvcc-missing
INSTALL="$WORK/install"
[ -d "$INSTALL/include/openfhe" ] || die "install-missing($INSTALL) -- run rider_build_fideslib.sh first"
[ -d "$WORK/FIDESlib" ] || die "fideslib-src-missing($WORK/FIDESlib)"
CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr -d '.' | paste -sd ';' -)"
[ -n "$CUDA_ARCH" ] || die arch-detect
echo "DETECTED CUDA_ARCH=$CUDA_ARCH"
JOBS="$(nproc)"

# ---- (1) no-LTO FIDESlib with NCCL visible, installed into the shared prefix
# 2026-09-17: on 2026-09-04 this apt-get collided with the torch venv's dpkg lock (10_torch.sh runs in parallel
# from s31_bootstrap.sh) and the demo build died on DEMOBUILD:nccl-install:fail; wait for the lock first (<= 20 min).
for i in $(seq 1 120); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 10; done
command -v dpkg >/dev/null && dpkg -l | grep -q libnccl-dev || \
  apt-get install -y -qq libnccl2 libnccl-dev || die nccl-install
mark nccl ok
cmake -S "$WORK/FIDESlib" -B "$WORK/FIDESlib/build-nolto" \
  -DCMAKE_BUILD_TYPE=Release -DFIDESLIB_INSTALL_OPENFHE=OFF -DFIDESLIB_ARCH="$CUDA_ARCH" \
  -DFIDESLIB_INSTALL_PREFIX="$INSTALL" -DOPENFHE_INSTALL_PREFIX="$INSTALL" \
  -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DCMAKE_CXX_FLAGS=-fno-lto \
  || die fideslib-nolto-configure
cmake --build "$WORK/FIDESlib/build-nolto" --target install -j"$JOBS" || die fideslib-nolto-build
sha256sum "$INSTALL/lib/fideslib.a" || true
# VERIFY the archive is actually no-LTO instead of trusting the directory name --
# v1 marked this step ok while installing a 12,910-section LTO archive.
LTOSEC=$(readelf -S "$INSTALL/lib/fideslib.a" 2>/dev/null | grep -c "gnu\.lto")
echo "DEMOBUILD:fideslib_gnu_lto_sections:${LTOSEC:-unknown}"
[ "${LTOSEC:-1}" -eq 0 ] || die "fideslib-still-lto(${LTOSEC} .gnu.lto sections -- the -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF did not take)"
mark fideslib-nolto ok

# ---- (2) demo harness, linking nccl explicitly
rm -rf "$WORK/build-demo"
cmake -S "$REPO/hpc_gpu_port" -B "$WORK/build-demo" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$INSTALL" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_EXE_LINKER_FLAGS="-lnccl" \
  -DCMAKE_CUDA_FLAGS="-DFHE_SSM_DEMO_SER -I$INSTALL/include/openfhe -I$INSTALL/include/openfhe/third-party/include -I$INSTALL/include/openfhe/core -I$INSTALL/include/openfhe/pke -I$INSTALL/include/openfhe/binfhe" \
  || die configure
mark configure ok
cmake --build "$WORK/build-demo" -j"$JOBS" --target gpu_real_model || die build-gpu_real_model
[ -x "$WORK/build-demo/gpu_real_model" ] || die binary-missing
mark gpu_real_model ok
# S3.7 batch 2 (2026-09-17): the EXPERIMENTAL copy gpu_real_model_x (hpc_gpu_port/experimental/, every lever behind
# its own flag, default OFF) from the SAME configure + -DFHE_SSM_EXPERIMENTAL=ON. Non-fatal: the real binary is the
# gate; BUILD_X=0 skips it.
if [ "${BUILD_X:-1}" = 1 ]; then
  cmake -DFHE_SSM_EXPERIMENTAL=ON "$WORK/build-demo" >/dev/null \
    && cmake --build "$WORK/build-demo" -j"$JOBS" --target gpu_real_model_x \
    && [ -x "$WORK/build-demo/gpu_real_model_x" ] && mark gpu_real_model_x ok || mark gpu_real_model_x fail
fi
if [ -f "$REPO/hpc_gpu_port/gpu_ring_native.cu" ]; then
  cmake --build "$WORK/build-demo" -j"$JOBS" --target gpu_ring_native && mark gpu_ring_native ok || mark gpu_ring_native fail
fi
# S3.1 T0.3(a): delivery-split bench (additive CMake guard, same pattern)
if [ -f "$REPO/hpc_gpu_port/gpu_delivery_bench.cu" ]; then
  cmake --build "$WORK/build-demo" -j"$JOBS" --target gpu_delivery_bench && mark gpu_delivery_bench ok || mark gpu_delivery_bench fail
fi
echo "=== DEMO BUILD DONE (arch $CUDA_ARCH, no-LTO fideslib + -lnccl) ==="
echo "binary: $WORK/build-demo/gpu_real_model   (runtime: LD_LIBRARY_PATH=$INSTALL/lib)"
