#!/usr/bin/env bash
# ON-RUN-AUTHORED 2026-08-22T17:5xZ (S2.9 BW pod) — diagnosis: x_est_setup
# attempt 1 failed rc=1 at step `sage-create`, console
# /root/campaign/state/runs/x_est_setup/console.log:
#   "error libmamba Could not solve for environment specs / The following
#    package could not be installed / sagemath =* * does not exist"
#
# DELTA vs estimator/setup_estimator_env.sh (sha256
# 5869173ce91357d617b86e13306dbea64ea54c99679b20cb0bca03520e4f8c73), which is
# sha-pinned by manifest x_est_setup and is therefore NOT edited in place:
#
#   ONE token: `micromamba create ... -c conda-forge sagemath`
#           -> `micromamba create ... -c conda-forge sage`
#
# WHY: on linux-64 conda-forge publishes the metapackage as `sage`, not
# `sagemath`. Verified on the pod, not assumed:
#     micromamba search -c conda-forge sagemath -> No entries matching "sagemath" found
#     micromamba search -c conda-forge sage     -> 9.7 ... 10.0, 10.1 (hd8ed1ab_0)
# The original name resolves on osx-arm64, which is where the Mac env in
# estimator/.micromamba was built (its pkgs/ tree is osx-arm64), so the script
# was correct for the box it was written on and wrong for this one.
#
# NET BEHAVIOURAL EFFECT: the environment is created from the same channel with
# the same solver; only the metapackage NAME changes. INVALIDATES NOTHING — no
# prior run used this script on linux-64 (it is marked "UNRUN on a fresh box as
# of 2026-08-22" in its own header), so there is no earlier record to revisit.
# Everything downstream (env name sage-estimator, MAMBA_ROOT_PREFIX, the pinned
# lattice-estimator commit 3e48ef4, the smoke test, the ESTENV markers) is
# byte-identical to the original.
set -uxo pipefail
cd "$(dirname "$0")"
mark() { echo "ESTENV:$1:$2"; }
die() { mark "$1" fail; exit 1; }
if [ ! -x bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba || die micromamba-download
fi
mark micromamba ok
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
if ! ./bin/micromamba env list 2>/dev/null | grep -q "sage-estimator"; then
  ./bin/micromamba create -y -n sage-estimator -c conda-forge sage || die sage-create
fi
mark sage ok
if [ ! -d lattice-estimator-src ]; then
  git clone https://github.com/malb/lattice-estimator.git lattice-estimator-src || die estimator-clone
  git -C lattice-estimator-src checkout 3e48ef4 || die estimator-checkout
fi
mark estimator ok
./bin/micromamba run -n sage-estimator sage -c "import sys; sys.path.insert(0,'lattice-estimator-src'); from estimator import *; print('estimator import OK')" || die smoke
mark smoke ok
echo "=== ESTIMATOR ENV READY. Hybrid battery:  MAMBA_ROOT_PREFIX=\$PWD/.micromamba ./bin/micromamba run -n sage-estimator sage run_fest_hybrid.py  (N=2^17 sets need >> 16 GB RAM; this box has 117 GB) ==="
