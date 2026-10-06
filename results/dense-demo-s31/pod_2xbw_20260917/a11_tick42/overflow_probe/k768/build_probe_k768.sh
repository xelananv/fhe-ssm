#!/bin/bash
# Build harness/boot_overflow_probe.cpp against the PATCHED install (vendor/install-k768) into
# harness/build/boot_overflow_probe_k768, by hand: the compile and link commands are the ones CMake generated for
# the stock probe (harness/build/build.ninja, target boot_overflow_probe: same compiler driver, same flags, same
# defines scheme), with the include / lib / rpath directories pointed at vendor/install-k768. harness/build's own
# ninja graph is not touched, so harness/build/boot_overflow_probe (stock install, sha256 bcaaea2a47d6...) stays as is.
# Run from the repo root:  bash results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/build_probe_k768.sh
set -u
OUT=results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768
LOG=$OUT/build_boot_overflow_probe_k768.log
R=$PWD
INST=$R/vendor/install-k768
SRC=$R/harness/boot_overflow_probe.cpp
OBJ=$R/harness/build/boot_overflow_probe_k768.o
BIN=$R/harness/build/boot_overflow_probe_k768
SHA16=$(shasum -a 256 "$SRC" | cut -c1-16)
GITSHA=$(git rev-parse --short=12 HEAD)
DIRTY=1; [ -z "$(git status --porcelain --untracked-files=no)" ] && DIRTY=0
UTC=$(date -u +%FT%TZ)
FLAGS="-Wall -Werror -DOPENFHE_VERSION=1.5.1 -O3 -DMATHBACKEND=4 -Wno-unknown-pragmas -O3 -DNDEBUG -arch arm64"
{
  echo; echo "# ===== build: boot_overflow_probe_k768  $UTC ====="
  echo "# source: $(shasum -a 256 harness/boot_overflow_probe.cpp)"
  echo "# stamp: harnessSha256=$SHA16 commit=$GITSHA dirty=$DIRTY buildUtc=$UTC"
  echo "# install: $INST"; shasum -a 256 $INST/lib/libOPENFHEpke.1.5.1.dylib $INST/lib/libOPENFHEcore.1.5.1.dylib $INST/lib/libOPENFHEbinfhe.1.5.1.dylib
  echo "# $(/usr/bin/c++ --version | head -1)"
  echo "# load state: $(tools/mem_probe.sh)"
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
  echo "# binary: $(shasum -a 256 harness/build/boot_overflow_probe_k768)"
  echo "# otool -L:"; otool -L $BIN
  echo "# rpaths:"; otool -l $BIN | grep -A2 LC_RPATH | grep path
  echo "# libraries actually loaded at run time (DYLD_PRINT_LIBRARIES=1; --log-ring 16 makes the probe exit at its W5 check before any work):"
  DYLD_PRINT_LIBRARIES=1 $BIN --log-ring 16 2>&1 | grep -i "openfhe\|W5"
} >> $LOG 2>&1
tail -22 $LOG
