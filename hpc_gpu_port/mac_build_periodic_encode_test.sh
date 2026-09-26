#!/bin/bash
# Builds hpc_gpu_port/periodic_encode_test (S3.7 V5, 2026-09-10) with the Mac recipe (OpenFHE 1.5.1 at
# vendor/install, WITH_OPENMP=OFF). Run from the repo (or worktree) root; R/V overrides as in the other
# mac_build_*.sh scripts. The test #includes periodic_encode.hpp from its own directory.
set -e
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
V=${V:-$REPO/vendor/install}
CXX=${CXX:-clang++}
"$CXX" -std=c++17 -O2 -DOPENFHE_VERSION=1.5.1 -DMATHBACKEND=4 -o "$R/hpc_gpu_port/periodic_encode_test" "$R/hpc_gpu_port/periodic_encode_test.cpp" \
  -I "$V/include/openfhe" -I "$V/include/openfhe/third-party/include" -I "$V/include/openfhe/core" \
  -I "$V/include/openfhe/pke" -I "$V/include/openfhe/binfhe" -I "$R/hpc_gpu_port" \
  -L "$V/lib" -lOPENFHEpke -lOPENFHEcore -Wl,-rpath,"$V/lib"
echo built
