#!/bin/sh
# Second OpenFHE prefix: a NAMED COMMIT of openfhe-development, built into its
# own src/build/install so `vendor/build` + `vendor/install` (the pinned v1.5.1
# every other probe links against) are never touched.
#
# Usage:  harness/build_openfhe_dev.sh <branch-or-tag> <slug>
#   e.g.  harness/build_openfhe_dev.sh dev devhead
# Produces: vendor/openfhe-<slug>/ (source, shallow clone)
#           vendor/build-<slug>/   (ninja build tree)
#           vendor/install-<slug>/ (CONFIG package for find_package(OpenFHE))
# Flags match build_openfhe.sh exactly (Release, OpenMP OFF, no tests/examples,
# NATIVE_SIZE default = 64) so the two prefixes differ only in source revision.
set -eu
cd "$(dirname "$0")/.."

REF="${1:-dev}"
SLUG="${2:-devhead}"
SRC="vendor/openfhe-$SLUG"

python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install --quiet cmake ninja

mkdir -p vendor
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$REF" --recurse-submodules \
    https://github.com/openfheorg/openfhe-development.git "$SRC"
fi
# cereal is a submodule; the shallow clone may leave it empty (as the v1.5.1
# release tarball does) — same fallback as build_openfhe.sh.
if [ ! -f "$SRC/third-party/cereal/include/cereal/cereal.hpp" ]; then
  curl -sL -o /tmp/cereal.tar.gz https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz
  tar -xzf /tmp/cereal.tar.gz -C /tmp
  cp -R /tmp/cereal-1.3.2/include "$SRC/third-party/cereal/"
fi

git -C "$SRC" rev-parse HEAD > "vendor/HEAD-$SLUG.txt"
echo "== $SLUG at $(cat "vendor/HEAD-$SLUG.txt")"

PATH="$PWD/.venv/bin:$PATH" cmake -S "$SRC" -B "vendor/build-$SLUG" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/vendor/install-$SLUG" \
  -DWITH_OPENMP=OFF -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_EXTRAS=OFF
PATH="$PWD/.venv/bin:$PATH" ninja -C "vendor/build-$SLUG" install
echo "OpenFHE ($REF @ $(cat "vendor/HEAD-$SLUG.txt")) installed to vendor/install-$SLUG"
