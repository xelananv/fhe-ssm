#!/bin/bash
# Builds hpc_gpu_port/keyparts_test (same prefix/flags as mac_build_client.sh;
# the test #includes mac_fhe_client.cpp so it links nothing extra).
set -e
R=$REPO
V=$R/vendor/install
clang++ -std=c++17 -O2 -o $R/hpc_gpu_port/keyparts_test $R/hpc_gpu_port/keyparts_test.cpp \
  -I $V/include/openfhe -I $V/include/openfhe/third-party/include \
  -I $V/include/openfhe/core -I $V/include/openfhe/pke -I $V/include/openfhe/binfhe \
  -L $V/lib -lOPENFHEpke -lOPENFHEcore -Wl,-rpath,$V/lib
echo built
