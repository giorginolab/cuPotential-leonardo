# cuPotential on Leonardo

Run NVIDIA’s cuPotential 0.3.5 sample and OpenMM 8.4’s stock benchmarks on one Leonardo A100. The setup uses `uv` and Python 3.12, with no Conda or Singularity.

See https://developer.nvidia.com/blog/accelerate-molecular-dynamics-simulations-with-accurate-flexible-primitives-using-nvidia-cupotential/

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

## Verified results

Both suites ran on one NVIDIA A100 with CUDA mixed precision. They used explicit TIP3P water, a 0.9 nm real-space cutoff, and Langevin middle integration at 300 K with a 4 fs timestep and 1 ps⁻¹ friction. Bonds to hydrogen were constrained, water was rigid, and solute hydrogen masses were repartitioned to 1.5 amu. The integrator constraint tolerance was 10⁻⁵.

### NVIDIA cuPotential comparison

All three systems used AMBER14. The stock arm used PME with an Ewald error tolerance of 5 × 10⁻⁴, and the cuPotential arm used the plugin’s nonbonded force and integrator. For NPT, each arm used its own Monte Carlo barostat at 1 bar, with a volume-move attempt every 25 steps.

Slurm job `64873726` ran 500 warm-up steps and 3,000 timed steps per implementation and ensemble. All three systems passed the sample’s energy and force checks against double-precision references.

| System | Ensemble | Stock ms/step | cuPotential ms/step | Stock ns/day | cuPotential ns/day | Speedup |
|---|---|---:|---:|---:|---:|---:|
| APOA1 | NVT | 0.750 | 0.285 | 460.7 | 1213.9 | 2.63× |
| APOA1 | NPT | 0.897 | 0.326 | 385.4 | 1061.2 | 2.75× |
| BACE | NVT | 0.548 | 0.181 | 630.3 | 1911.7 | 3.03× |
| BACE | NPT | 0.699 | 0.223 | 494.3 | 1546.6 | 3.13× |
| DHFR | NVT | 0.284 | 0.088 | 1216.9 | 3941.3 | 3.24× |
| DHFR | NPT | 0.341 | 0.099 | 1013.2 | 3492.0 | 3.45× |

See [benchmark.log](benchmark.log) for the full output.

### Stock OpenMM benchmark

Slurm job `64874161` ran OpenMM’s own benchmarks in NVT with CUDA mixed precision and a 10-second target per test. Both tests used PME. DHFR used AMBER99SB, and APOA1 used AMBER ff14SB with Lipid17.

| System | OpenMM test | Throughput (ns/day) |
|---|---|---:|
| DHFR | `pme` | 1269.07 |
| APOA1 | `apoa1pme` | 486.321 |
