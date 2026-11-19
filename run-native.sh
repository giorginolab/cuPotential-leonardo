#!/bin/bash -l
set -eo pipefail
cd "$(dirname "$0")"
trial=$PWD
module add cuda/12.6
module add gcc/12.2.0
local_trial=$(mktemp -d "${TMPDIR:-/dev/shm}/cupotential-XXXXXX")
trap 'rm -rf -- "$local_trial"' EXIT
mkdir "$local_trial/glibc" "$local_trial/compat"
ar p "$trial/libc6.deb" data.tar.zst | zstd -dc | tar -xf - -C "$local_trial/glibc"
ar p "$trial/cuda-compat-13-4.deb" data.tar.xz | tar -xJf - -C "$local_trial/compat"
tar -xf "$trial/packages.tar" -C "$local_trial"
glibc="$local_trial/glibc/usr/lib/x86_64-linux-gnu"
compat="$local_trial/compat/usr/local/cuda-13.4/compat"
libstdcpp=$(dirname "$(g++ -print-file-name=libstdc++.so.6)")
python=$("$trial/uv" python find 3.12)
export PYTHONPATH="$local_trial/packages"
export CUPO_OPENMM_BENCHMARKS_DIR="$trial/inputs"
nvidia_libs=$(find "$local_trial/packages/nvidia" -type d -name lib -printf '%p:')
libraries="$glibc:$compat:$nvidia_libs$libstdcpp"
if (( $# )); then
    script=$(realpath "$1")
    shift
    cd "$(dirname "$script")"
else
    script="$trial/cuPotential-0.3.5/plugins/openmm/samples/solvated_systems/sample.py"
fi
"$glibc/ld-linux-x86-64.so.2" --library-path "$libraries" "$python" -u "$script" "$@"
