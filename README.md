# cuPotential on Leonardo

Run NVIDIA’s cuPotential 0.3.5 sample and OpenMM 8.4’s stock benchmarks on one Leonardo A100. The setup uses `uv` and Python 3.12, with no Conda or Singularity.

## Install from scratch

On a Leonardo login node, start in the directory containing this README and its two shell scripts. Use shared scratch so the GPU node can read the installation.

```bash
set -e
trial=$(mktemp -d "$SCRATCH/cupotential-XXXXXX")
printf 'Installation directory: %s\n' "$trial"
cp setup-native.sh run-native.sh "$trial/"
cp "$(command -v uv)" "$trial/uv"
"$trial/uv" python install 3.12
cd "$trial"

wget -O source.tar.gz https://codeload.github.com/NVIDIA/cuPotential/tar.gz/refs/tags/v0.3.5
tar -xzf source.tar.gz

wget -O libc6.deb \
  https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8_amd64.deb
wget -O cuda-compat-13-4.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-13-4_615.71.10-1ubuntu1_amd64.deb

mkdir -p inputs
openmm_url=https://raw.githubusercontent.com/openmm/openmm/8.4.0/examples/benchmarks
for file in benchmark.py apoa1.pdb 5dfr_solv-cube_equil.pdb; do
  wget -O "inputs/$file" "$openmm_url/$file"
done

bash setup-native.sh
```

The setup script downloads the Python wheels with `uv` and stages them in `packages.tar`. The runner loads Leonardo’s CUDA and GCC modules and uses the downloaded glibc and CUDA compatibility libraries for the benchmark process.

## Run

Keep `trial` set to the installation directory printed above. Run NVIDIA’s full benchmark, which compares cuPotential with stock OpenMM and ends with `PASS` on success.

```bash
srun -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --chdir="$trial" \
  /bin/bash "$trial/run-native.sh"
```

Run OpenMM’s own DHFR and APOA1 benchmarks in mixed precision. Use `--ensemble=NPT` for constant pressure.

```bash
srun -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --chdir="$trial" \
  /bin/bash "$trial/run-native.sh" "$trial/inputs/benchmark.py" \
  --platform=CUDA --precision=mixed --test=pme,apoa1pme --ensemble=NVT \
  --seconds=10 --style=table
```

See the [Leonardo how-to](cuPotential-leonardo-howto.md) for setup details and results.
