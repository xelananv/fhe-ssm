#!/bin/bash
# VARIANT B = the K768 env patch (openfhe_k768_env.patch) PLUS the split of the division by K = 768 = 3 * 2^8
# (openfhe_k768_split_variantB.patch: OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1 folds the 3 into the CoeffsToSlots precomputation and
# leaves the scalar constant pre/(256*N), a power of two). Separate source copy (vendor/openfhe-k768b-src), build tree
# (vendor/build-k768b) and install prefix (vendor/install-k768b), so that the library that produced the main records
# (vendor/install-k768) is not touched. Same cmake options as harness/build_openfhe.sh. Then the SAME probe source is
# linked against that install into harness/build/boot_overflow_probe_k768b (commands as in build_probe_k768.sh).
# Run from the repo root:  bash results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_variantB.sh
set -u
OUT=results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768
LOG=$OUT/build_variantB.log
export PATH="$PWD/.venv/bin:$PATH"
R=$PWD; INST=$R/vendor/install-k768b
{
  echo "# build log: VARIANT B library + probe   start $(date -u +%FT%TZ)"
  echo "# source sha256:"; shasum -a 256 vendor/openfhe-k768b-src/src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp vendor/openfhe-k768b-src/src/pke/include/scheme/ckksrns/ckksrns-fhe.h
  echo "# files that differ between the pristine tree and the variant-B copy:"; diff -rq vendor/openfhe-development vendor/openfhe-k768b-src
  echo "# files that differ between the k768 copy and the variant-B copy:"; diff -rq vendor/openfhe-k768-src vendor/openfhe-k768b-src
  echo "# load state before: $(tools/mem_probe.sh)"
} > $LOG 2>&1
tools/memguard.sh 8 cmake -S vendor/openfhe-k768b-src -B vendor/build-k768b -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INST" \
  -DWITH_OPENMP=OFF -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_EXTRAS=OFF >> $LOG 2>&1
rc=$?; echo "# configure exit=$rc $(date -u +%FT%TZ)" >> $LOG; [ $rc -eq 0 ] || exit $rc
caffeinate -i tools/memguard.sh 8 ninja -C vendor/build-k768b -j4 install >> $LOG 2>&1
rc=$?; echo "# ninja install exit=$rc $(date -u +%FT%TZ)" >> $LOG; [ $rc -eq 0 ] || exit $rc
SRC=$R/harness/boot_overflow_probe.cpp; OBJ=$R/harness/build/boot_overflow_probe_k768b.o; BIN=$R/harness/build/boot_overflow_probe_k768b
SHA16=$(shasum -a 256 "$SRC" | cut -c1-16); GITSHA=$(git rev-parse --short=12 HEAD)
DIRTY=1; [ -z "$(git status --porcelain --untracked-files=no)" ] && DIRTY=0
UTC=$(date -u +%FT%TZ)
FLAGS="-Wall -Werror -DOPENFHE_VERSION=1.5.1 -O3 -DMATHBACKEND=4 -Wno-unknown-pragmas -O3 -DNDEBUG -arch arm64"
{
  echo "# installed libraries:"; shasum -a 256 $INST/lib/libOPENFHEpke.1.5.1.dylib $INST/lib/libOPENFHEcore.1.5.1.dylib $INST/lib/libOPENFHEbinfhe.1.5.1.dylib
  echo "# installed headers vs the stock install:"; diff -rq vendor/install/include $INST/include && echo "identical"
  echo "# ===== probe: boot_overflow_probe_k768b  $UTC ====="
  echo "# source: $(shasum -a 256 harness/boot_overflow_probe.cpp)"
  echo "# stamp: harnessSha256=$SHA16 commit=$GITSHA dirty=$DIRTY buildUtc=$UTC"
  set -x
  tools/memguard.sh 6 /usr/bin/c++ \
    -DFHE_SSM_BUILD_UTC=\"$UTC\" -DFHE_SSM_GIT_DIRTY=$DIRTY -DFHE_SSM_GIT_SHA=\"$GITSHA\" \
    -DFHE_SSM_HARNESS_FILE=\"boot_overflow_probe.cpp\" -DFHE_SSM_HARNESS_SHA256=\"$SHA16\" \
    -I$INST/include/openfhe -I$INST/include/openfhe/third-party/include -I$INST/include/openfhe/core \
    -I$INST/include/openfhe/pke -I$INST/include/openfhe/binfhe -I$R/harness/../hpc_gpu_port \
    $FLAGS -std=gnu++17 -o $OBJ -c $SRC
  rc1=$?
  tools/memguard.sh 6 /usr/bin/c++ $FLAGS -Wl,-search_paths_first -Wl,-headerpad_max_install_names $OBJ -o $BIN \
    -L$INST/lib -Wl,-rpath,$INST/lib \
    $INST/lib/libOPENFHEpke.1.5.1.dylib $INST/lib/libOPENFHEbinfhe.1.5.1.dylib $INST/lib/libOPENFHEcore.1.5.1.dylib
  rc2=$?
  set +x
  echo "# compile exit=$rc1 link exit=$rc2 $(date -u +%FT%TZ)"
  echo "# binary: $(shasum -a 256 harness/build/boot_overflow_probe_k768b)"
  echo "# rpaths:"; otool -l $BIN | grep -A2 LC_RPATH | grep path
  echo "# libraries actually loaded at run time:"
  DYLD_PRINT_LIBRARIES=1 $BIN --log-ring 16 2>&1 | grep -i "openfhe\|W5"
  echo "# load state after: $(tools/mem_probe.sh)"
} >> $LOG 2>&1
grep -n "^# " $LOG | tail -24
