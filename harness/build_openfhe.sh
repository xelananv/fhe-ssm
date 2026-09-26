#!/bin/sh
# Reproducible OpenFHE toolchain bootstrap for this repo (macOS, no Homebrew).
# cmake/ninja come from pip wheels; OpenFHE v1.5.1 from the release tarball
# (which lacks the cereal submodule — fetched separately).
set -eu
cd "$(dirname "$0")/.."

python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install --quiet cmake ninja

mkdir -p vendor
if [ ! -d vendor/openfhe-development ]; then
  curl -sL -o vendor/openfhe.tar.gz \
    https://github.com/openfheorg/openfhe-development/archive/refs/tags/v1.5.1.tar.gz
  tar -xzf vendor/openfhe.tar.gz -C vendor
  mv vendor/openfhe-development-1.5.1 vendor/openfhe-development
  rm vendor/openfhe.tar.gz
fi
if [ ! -f vendor/openfhe-development/third-party/cereal/include/cereal/cereal.hpp ]; then
  curl -sL -o /tmp/cereal.tar.gz https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz
  tar -xzf /tmp/cereal.tar.gz -C /tmp
  cp -R /tmp/cereal-1.3.2/include vendor/openfhe-development/third-party/cereal/
fi

PATH="$PWD/.venv/bin:$PATH" cmake -S vendor/openfhe-development -B vendor/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/vendor/install" \
  -DWITH_OPENMP=OFF -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_EXTRAS=OFF
PATH="$PWD/.venv/bin:$PATH" ninja -C vendor/build install
echo "OpenFHE installed to vendor/install"
