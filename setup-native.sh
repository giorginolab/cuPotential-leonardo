#!/bin/bash
set -euo pipefail
trial=$(cd "$(dirname "$0")" && pwd)
store=$(mktemp -d "${CUPOTENTIAL_STAGING_DIR:-/scratch_local}/cupotential-wheels-XXXXXX")
trap 'rm -rf -- "$store"' EXIT
export UV_CACHE_DIR="$store/cache"
"$trial/uv" pip install --python 3.12 --target "$store/packages" --python-platform x86_64-manylinux_2_38 \
    nv-openmm-plugin==0.3.5 'openmm[cuda13]==8.4.0.post2'
tar -cf "$store/packages.tar" -C "$store" packages
cp "$store/packages.tar" "$trial/packages.tar.new"
mv "$trial/packages.tar.new" "$trial/packages.tar"
