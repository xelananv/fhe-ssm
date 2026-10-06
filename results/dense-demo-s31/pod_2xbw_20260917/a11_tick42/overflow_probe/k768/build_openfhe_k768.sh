#!/bin/bash
# Build + install the PATCHED COPY of the vendored OpenFHE v1.5.1 (vendor/openfhe-k768-src, see openfhe_k768_env.patch)
# into vendor/install-k768. Same cmake options as harness/build_openfhe.sh (which produced vendor/install):
# Release, WITH_OPENMP=OFF, no unittests/examples/benchmarks/extras, Ninja, pip cmake/ninja from .venv/bin.
# The stock tree (vendor/openfhe-development) and the stock install (vendor/install) are NOT touched.
# one heavy process, under tools/memguard.sh 8; -j4 so the guard is not tripped.
# Run from the repo root:  bash results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_openfhe_k768.sh
set -u
OUT=results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768
LOG=$OUT/build_openfhe_k768.log
export PATH="$PWD/.venv/bin:$PATH"
{
  echo "# build log: patched OpenFHE copy -> vendor/install-k768   start $(date -u +%FT%TZ)"
  echo "# source sha256:"; shasum -a 256 vendor/openfhe-k768-src/src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp vendor/openfhe-k768-src/src/pke/include/scheme/ckksrns/ckksrns-fhe.h
  echo "# pristine sha256:"; shasum -a 256 vendor/openfhe-development/src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp vendor/openfhe-development/src/pke/include/scheme/ckksrns/ckksrns-fhe.h
  echo "# files that differ between the pristine tree and the copy:"; diff -rq vendor/openfhe-development vendor/openfhe-k768-src
  echo "# toolchain: $(cmake --version | head -1); ninja $(ninja --version); $(/usr/bin/c++ --version | head -1)"
  echo "# load state before: $(tools/mem_probe.sh)"; uptime
  echo "# cmd: tools/memguard.sh 8 cmake -S vendor/openfhe-k768-src -B vendor/build-k768 -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/vendor/install-k768 -DWITH_OPENMP=OFF -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXTRAS=OFF"
} > $LOG 2>&1
tools/memguard.sh 8 cmake -S vendor/openfhe-k768-src -B vendor/build-k768 -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/vendor/install-k768" \
  -DWITH_OPENMP=OFF -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_EXTRAS=OFF >> $LOG 2>&1
rc=$?; echo "# configure exit=$rc $(date -u +%FT%TZ)" >> $LOG
[ $rc -eq 0 ] || exit $rc
echo "# cmd: caffeinate -i tools/memguard.sh 8 ninja -C vendor/build-k768 -j4 install" >> $LOG
caffeinate -i tools/memguard.sh 8 ninja -C vendor/build-k768 -j4 install >> $LOG 2>&1
rc=$?
{
  echo "# ninja install exit=$rc $(date -u +%FT%TZ)"
  echo "# load state after: $(tools/mem_probe.sh)"
  echo "# installed libraries:"; ls -la vendor/install-k768/lib/ 2>&1
  shasum -a 256 vendor/install-k768/lib/libOPENFHEpke.1.5.1.dylib vendor/install-k768/lib/libOPENFHEcore.1.5.1.dylib vendor/install-k768/lib/libOPENFHEbinfhe.1.5.1.dylib 2>&1
} >> $LOG 2>&1
exit $rc
