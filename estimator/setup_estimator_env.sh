#!/usr/bin/env bash
# Self-contained sage + lattice-estimator environment for the F-est/hybrid
# batteries on a box that has never had one (S2.9 Block E, gpus:0 rider).
# Recipe = the one the Mac env was built from (the original pod setup notes, not included
# §9a) + the pinned lattice-estimator commit (3e48ef4, estimator/run_lattice_estimator.py
# header). Writes into estimator/ so the existing run_*.py invocations work
# unchanged:  MAMBA_ROOT_PREFIX=$PWD/.micromamba ./bin/micromamba run -n sage-estimator sage run_fest_hybrid.py
# UNRUN on a fresh box as of 2026-08-22. Markers: ESTENV:<step>:<ok|fail>.
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
  ./bin/micromamba create -y -n sage-estimator -c conda-forge sagemath || die sage-create
fi
mark sage ok
if [ ! -d lattice-estimator-src ]; then
  git clone https://github.com/malb/lattice-estimator.git lattice-estimator-src || die estimator-clone
  git -C lattice-estimator-src checkout 3e48ef4 || die estimator-checkout
fi
mark estimator ok
./bin/micromamba run -n sage-estimator sage -c "import sys; sys.path.insert(0,'lattice-estimator-src'); from estimator import *; print('estimator import OK')" || die smoke
mark smoke ok
echo "=== ESTIMATOR ENV READY. Hybrid battery:  MAMBA_ROOT_PREFIX=\$PWD/.micromamba ./bin/micromamba run -n sage-estimator sage run_fest_hybrid.py  (N=2^17 sets need >> 16 GB RAM) ==="
