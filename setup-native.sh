#!/bin/bash
# One-shot setup, run on a Leonardo login node from the directory holding this
# script and run-native.sh (use shared $SCRATCH so GPU nodes can read it).
#   bash setup-native.sh
# Downloads everything into the script's directory and leaves it ready for run-native.sh.
set -euo pipefail
trial=$(cd "$(dirname "$0")" && pwd)
cd "$trial"

[[ -x uv ]] || cp "$(command -v uv)" uv
./uv python install 3.12

fetch() {  # fetch URL OUTPUT [SHA256]
    if [[ ! -f $2 ]]; then
        wget -q -O "$2.part" "$1"
        mv "$2.part" "$2"
    fi
    if [[ -n ${3:-} ]]; then
        echo "$3  $2" | sha256sum -c --quiet
    fi
}

# NVIDIA's sample and BACE input (pinned tag)
if [[ ! -d cuPotential-0.3.5 ]]; then
    fetch https://codeload.github.com/NVIDIA/cuPotential/tar.gz/refs/tags/v0.3.5 source.tar.gz
    tar -xzf source.tar.gz
fi

# OpenMM 8.4.0 stock benchmark inputs
mkdir -p inputs
base=https://raw.githubusercontent.com/openmm/openmm/8.4.0/examples/benchmarks
fetch $base/benchmark.py                 inputs/benchmark.py                 942d21b57d50000499bda4df222cfbd146d32b5da7a5356311f0da3fda4ce25a
fetch $base/apoa1.pdb                    inputs/apoa1.pdb                    900cdb7a2d9a90f4cbd1de43e9562566cb281a92e8696ce5a87f95a4bb2cb88e
fetch $base/5dfr_solv-cube_equil.pdb     inputs/5dfr_solv-cube_equil.pdb     fc9759761d9b33c58b09a3d0bf1dbdb76f23dbe0e89a8bd92f43ebcf0f2f2fdf

# Runtime libraries, extracted once (no .deb handling at job time).
# The cuPotential wheel needs glibc >= 2.38 (Leonardo has 2.28; no site module provides a newer one),
# so Python is started through a private glibc loader. Nothing on the host is changed.
if [[ ! -d runtime/glibc ]]; then
    fetch https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8_amd64.deb libc6.deb \
          af36c7ac770770fe3d3c10e85d6bc538e76e57570ba7db7d397fb9f654783ef3
    mkdir -p runtime/glibc
    ar p libc6.deb data.tar.zst | zstd -dc | tar -xf - -C runtime/glibc
fi
# CUDA 13.4 forward-compatibility libraries, needed because Leonardo's driver is R535
if [[ ! -d runtime/compat ]]; then
    fetch https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-13-4_615.71.10-1ubuntu1_amd64.deb cuda-compat.deb \
          f62ed0d30d8939bba3a7b55727e3fd70d657ae4973fa2875e775068a2c305f30
    mkdir -p runtime/compat
    ar p cuda-compat.deb data.tar.xz | tar -xJf - -C runtime/compat
fi
rm -f libc6.deb cuda-compat.deb source.tar.gz

# Python wheels, staged as a tarball (unpacked to node-local storage by the runner)
store=$(mktemp -d "${CUPOTENTIAL_STAGING_DIR:-${TMPDIR:-/tmp}}/cupotential-wheels-XXXXXX")
trap 'rm -rf -- "$store"' EXIT
export UV_CACHE_DIR="$store/cache"
./uv pip install --python 3.12 --target "$store/packages" --python-platform x86_64-manylinux_2_38 \
    nv-openmm-plugin==0.3.5 'openmm[cuda13]==8.4.0.post2'
tar -cf "$store/packages.tar" -C "$store" packages
cp "$store/packages.tar" packages.tar.new
mv packages.tar.new packages.tar
echo "Setup complete in $trial"
