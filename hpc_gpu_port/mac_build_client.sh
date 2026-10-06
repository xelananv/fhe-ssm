#!/bin/bash
set -e
V=$REPO/vendor/install
clang++ -std=c++17 -O2 -o hpc_gpu_port/mac_fhe_client hpc_gpu_port/mac_fhe_client.cpp \
  -I $V/include/openfhe -I $V/include/openfhe/third-party/include \
  -I $V/include/openfhe/core -I $V/include/openfhe/pke -I $V/include/openfhe/binfhe \
  -L $V/lib -lOPENFHEpke -lOPENFHEcore -Wl,-rpath,$V/lib
echo built
