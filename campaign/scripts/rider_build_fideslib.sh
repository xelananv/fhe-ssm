#!/usr/bin/env bash
# FIDESlib rider build (S2.5 §8): one stack (v2.1.2 or main) + patched OpenFHE
# + the gpu_real_model harness, into an isolated work dir, on any CUDA>=12.4
# box. Uses the 01b route (the one that works — C12: FIDESLIB_INSTALL_OPENFHE=ON
# fails out-of-tree) with the known fixups: --remote submodule sed, -j12 sed,
# C10 PeerUtils patch (harmless on >=12.4), arch auto-detect (L40S -> 89).
# Tolerant chain style: per-step RIDER:<step>:<ok|fail> markers, exits on fail.
#
#   rider_build_fideslib.sh --ref v2.1.2 --work /root/fhe-v212 --repo /root/src
set -uxo pipefail
REF=v2.1.2; WORK=""; REPO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF=$2; shift 2;;
    --work) WORK=$2; shift 2;;
    --repo) REPO=$2; shift 2;;
    *) shift;;
  esac
done
: "${WORK:?--work required}"; : "${REPO:?--repo required}"
JOBS="$(nproc)"
mark() { echo "RIDER:$1:$2"; }
die() { mark "$1" fail; exit 1; }
mkdir -p "$WORK"

# ---- step 0: toolchain probes (training pods ship python, not cmake) --------
command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
if ! command -v nvcc >/dev/null; then
  die nvcc-missing   # installing the toolkit is 00_toolchain.sh's job — escalate
fi
NVCC_VER=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
awk -v v="$NVCC_VER" 'BEGIN{exit !(v+0 >= 12.4)}' || die "cuda-too-old-$NVCC_VER"
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -y -qq --no-install-recommends \
    build-essential cmake ninja-build git libssl-dev zlib1g-dev libomp-dev \
    || die apt-toolchain
fi
# 2026-09-17: FIDESlib's CMakeLists requires cmake >= 3.25.2 (hpc_gpu_port/CMakeLists.txt:17); Ubuntu 22.04's apt
# cmake is 3.22 -> take cmake/ninja from pip (lands in /usr/local/bin, ahead of /usr/bin).
CMV=$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
if ! awk -v v="${CMV:-0.0.0}" 'BEGIN{split(v,a,"."); exit !(a[1]>3 || (a[1]==3 && (a[2]>25 || (a[2]==25 && a[3]>=2))))}'; then
  echo "cmake ${CMV:-missing} < 3.25.2 -> pip"
  ( python3 -m pip install -q --break-system-packages 'cmake>=3.28' ninja 2>/dev/null || python3 -m pip install -q 'cmake>=3.28' ninja ) || die cmake-too-old-$CMV
  hash -r; export PATH=/usr/local/bin:$PATH
  cmake --version | head -1
fi
mark toolchain ok

# ---- step 1: arch detect (NEVER hardcode — a hardcoded arch was an original
# porting bug; L40S is sm_89) -------------------------------------------------
CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr -d '.' | paste -sd ';' -)"
[ -n "$CUDA_ARCH" ] || die arch-detect
echo "DETECTED CUDA_ARCH=$CUDA_ARCH"
mark arch ok

# ---- step 2: clone + record + C10 patch ------------------------------------
if [ ! -d "$WORK/FIDESlib" ]; then
  git clone --branch "$REF" --depth 1 https://github.com/CAPS-UMU/FIDESlib.git "$WORK/FIDESlib" \
    || die clone
fi
# 2026-09-18 (long-run prep): PIN the commit our patches were authored against (upstream `main` moves; last session built
# fa972864) and apply the repo's own FIDESlib patches BEFORE anything is compiled. Both are env-driven so the callers
# (s31_bootstrap.sh, 10_bringup.sh) need no new arguments:
#   FIDESLIB_PIN_COMMIT=<sha>            fetch + checkout exactly this commit (die if it cannot be fetched)
#   FIDESLIB_EXTRA_PATCH_DIR=<dir>       git-apply every *.patch in lexical order; already-applied patches are skipped; any
#                                        patch that neither applies nor is already applied is FATAL (never build a half-patched tree)
if [ -n "${FIDESLIB_PIN_COMMIT:-}" ]; then
  if [ "$(git -C "$WORK/FIDESlib" rev-parse HEAD)" != "$FIDESLIB_PIN_COMMIT" ]; then
    git -C "$WORK/FIDESlib" fetch --depth 1 origin "$FIDESLIB_PIN_COMMIT" && git -C "$WORK/FIDESlib" checkout -q "$FIDESLIB_PIN_COMMIT" || die pin-commit
  fi
  [ "$(git -C "$WORK/FIDESlib" rev-parse HEAD)" = "$FIDESLIB_PIN_COMMIT" ] || die pin-commit-mismatch
  mark pin-commit ok
fi
if [ -n "${FIDESLIB_EXTRA_PATCH_DIR:-}" ]; then
  for pf in "$FIDESLIB_EXTRA_PATCH_DIR"/*.patch; do
    [ -f "$pf" ] || continue
    if git -C "$WORK/FIDESlib" apply --check "$pf" 2>/dev/null; then
      git -C "$WORK/FIDESlib" apply "$pf" || die "extra-patch-$(basename "$pf")"; echo "EXTRA-PATCH applied: $(basename "$pf") sha256 $(sha256sum "$pf" | cut -c1-16)"
    elif git -C "$WORK/FIDESlib" apply -R --check "$pf" 2>/dev/null; then echo "EXTRA-PATCH already applied: $(basename "$pf")"
    else die "extra-patch-does-not-apply-$(basename "$pf")"; fi
  done
  git -C "$WORK/FIDESlib" diff --stat | tail -5
  mark extra-patches ok
fi
git -C "$WORK/FIDESlib" rev-parse HEAD > "$WORK/FIDESLIB_COMMIT.txt"
python3 "$REPO/hpc_gpu_port/patches/patch_peerutils.py" "$WORK/FIDESlib/src/PeerUtils.cu" \
  || die peerutils-patch
# 2026-09-03: multi-GPU key memory -- skip the replicated, never-read DECOMP
# limbs of key-switching keys under GPUid.size()>1 (12 GB/card at 2^17). The
# applier is idempotent and REFUSES (exit 1) if FIDESlib drifted past its anchor.
python3 "$REPO/hpc_gpu_port/patches/patch_mgpu_key_decomp.py" "$WORK/FIDESlib/src/CKKS/LimbPartition.cu" \
  || die mgpu-key-decomp-patch
mark clone ok

# ---- step 3: deps/build.sh fixups. Patch lineage differs between v2.1.2
# (deps/openfhe-1.5.1.patch) and main (deps/fideslib-ref-1.5.1.1.patch,
# repointed submodule) — probe, log, and DO NOT improvise if neither is found
# (PHASE_A_REPORT.md:119-146). ------------------------------------------------
DEPS="$WORK/FIDESlib/deps"
if [ -f "$DEPS/openfhe-1.5.1.patch" ]; then
  echo "PATCH-LINEAGE: v2.1.2-style (openfhe-1.5.1.patch)"
elif ls "$DEPS"/fideslib-ref-*.patch >/dev/null 2>&1; then
  echo "PATCH-LINEAGE: main-style ($(ls "$DEPS"/fideslib-ref-*.patch))"
else
  die unknown-patch-lineage
fi
sed -i 's/ --remote//' "$DEPS/build.sh"
sed -i 's/^make -j12$/make -j'"$JOBS"'/; s/^make install -j12$/make install -j'"$JOBS"'/' "$DEPS/build.sh"
# 2026-09-18 (long-run prep): OPENFHE_EXTRA_PATCH_DIR=<dir> -- deps/build.sh wipes and re-checks-out openfhe-src, so the repo's own
# OpenFHE patches (*.patch directly in <dir>, lexical order) are applied INSIDE it, right after FIDESlib's own OpenFHE patch.
# A patch that does not apply is SKIPPED with a loud line and the build continues: the only such patch (O1, scaleEnc = pre/3 under
# FIDESLIB_BOOT_UNIFORM_EXT=split) is an option whose absence FIDESlib's K1 detects at compile time (macro) and falls back from.
# A build.sh without the expected anchor line is FATAL here, before anything compiles (the FIDESlib commit is pinned: it cannot move).
if [ -n "${OPENFHE_EXTRA_PATCH_DIR:-}" ]; then
  python3 - "$DEPS/build.sh" "$OPENFHE_EXTRA_PATCH_DIR" <<'PY' || die openfhe-extra-patch-hook
import sys
path, pdir = sys.argv[1], sys.argv[2]
lines = open(path).read().split("\n")
if any("OPENFHE-EXTRA-PATCH" in l for l in lines): sys.exit(0)            # idempotent
idx = [i for i, l in enumerate(lines) if l.startswith("git apply ../")]
if len(idx) != 1: sys.exit("build.sh: expected exactly one active 'git apply ../<patch>' line, found %d" % len(idx))
hook = ('for pf in "%s"/*.patch; do [ -f "$pf" ] || continue; '
        'if git apply --check "$pf"; then git apply "$pf"; echo "OPENFHE-EXTRA-PATCH applied: $(basename "$pf") sha256 $(sha256sum "$pf" | cut -c1-16)"; '
        'else echo "OPENFHE-EXTRA-PATCH DOES NOT APPLY, SKIPPED (build continues without it): $(basename "$pf")"; fi; done') % pdir
lines.insert(idx[0] + 1, hook)
open(path, "w").write("\n".join(lines))
PY
  grep -n "OPENFHE-EXTRA-PATCH" "$DEPS/build.sh" | head -2
  mark openfhe-extra-patch-hook ok
fi
mark deps-fixups ok

# ---- step 4: patched OpenFHE via deps/build.sh (the 01b route) --------------
( cd "$DEPS" && ./build.sh "$WORK/install" ) || die openfhe-build
mark openfhe ok
if [ -n "${OPENFHE_EXTRA_PATCH_DIR:-}" ]; then
  if grep -rqs "OPENFHE_FHESSM_BOOT_UNIFORM_SPLIT_ENV" "$WORK/install/include"; then echo "OPENFHE-O1: present in the installed headers (FIDESLIB_BOOT_UNIFORM_EXT=split is available)"
  else echo "OPENFHE-O1: ABSENT from the installed headers (FIDESLIB_BOOT_UNIFORM_EXT=split will fall back to =1 and say so)"; fi
fi

# ---- step 5: FIDESlib against the installed OpenFHE -------------------------
rm -rf "$WORK/FIDESlib/build"
cmake -S "$WORK/FIDESlib" -B "$WORK/FIDESlib/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFIDESLIB_INSTALL_OPENFHE=OFF \
  -DFIDESLIB_ARCH="$CUDA_ARCH" \
  -DFIDESLIB_INSTALL_PREFIX="$WORK/install" \
  -DOPENFHE_INSTALL_PREFIX="$WORK/install" \
  -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
  || die fideslib-configure
cmake --build "$WORK/FIDESlib/build" --target install -j"$JOBS" || die fideslib-build
mark fideslib ok

# ---- step 6: harness --------------------------------------------------------
cmake -S "$REPO/hpc_gpu_port" -B "$WORK/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_PREFIX_PATH="$WORK/install" \
  || die harness-configure
cmake --build "$WORK/build" -j"$JOBS" || die harness-build
[ -x "$WORK/build/gpu_real_model" ] || die harness-binary-missing
mark harness ok

# OpenMP is load-bearing and fails SILENTLY if -Xcompiler=-fopenmp was dropped
# (2.93x on the diagonal encode): the runtime check is "ompMaxThreads" in the
# harness header line of the first real run — the reproducer manifests' logs
# show it; verify there.
echo "=== RIDER BUILD DONE ($REF, arch $CUDA_ARCH) ==="
echo "binary: $WORK/build/gpu_real_model   (runtime: LD_LIBRARY_PATH=$WORK/install/lib)"
