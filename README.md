# cuPotential on Leonardo

Run NVIDIA’s cuPotential 0.3.5 sample and OpenMM 8.4’s stock benchmarks on one Leonardo A100. The setup uses `uv` and Python 3.12, with no Conda or Singularity.

See https://developer.nvidia.com/blog/accelerate-molecular-dynamics-simulations-with-accurate-flexible-primitives-using-nvidia-cupotential/

## Install from scratch

On a Leonardo login node, start in the directory containing this README and its shell scripts. Use shared scratch so the GPU node can read the installation. The setup takes about 30 seconds.

```bash
set -e
trial=$(mktemp -d "$SCRATCH/cupotential-XXXXXX")
printf 'Installation directory: %s\n' "$trial"
cp setup-native.sh run-native.sh "$trial/"
bash "$trial/setup-native.sh"
```

`setup-native.sh` does all the downloading: the pinned cuPotential 0.3.5 source (for the sample), the OpenMM 8.4.0 benchmark inputs, and the Python wheels (via `uv`, staged in `packages.tar`). It also fetches the two runtime libraries below. Both downloads and the OpenMM inputs are SHA-256-checked and unpacked once into `$trial/runtime`.

- **glibc 2.39** (Ubuntu `libc6`). The cuPotential wheel needs glibc 2.38 or newer, and Leonardo has 2.28 with no newer site module. The runner starts Python through this private loader, so the host is untouched. The wheel's native library cannot load on glibc 2.28, so this download has no wheel-only substitute.
- **`cuda-compat-13-4`**. Leonardo's R535 driver is older than the R615 that CUDA 13.4 needs. NVIDIA's [forward-compatibility package](https://docs.nvidia.com/cuda/cupotential/install) covers the gap.

The runner loads Leonardo's CUDA and GCC modules and unpacks the wheels to node-local storage for each job.

## Run

Keep `trial` set to the installation directory printed above. Run NVIDIA’s full benchmark, which compares cuPotential with stock OpenMM and ends with `PASS` on success.

```bash
srun -A $SLURM_ACCOUNT -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --chdir="$trial" \
  /bin/bash "$trial/run-native.sh"
```

Run OpenMM’s own DHFR and APOA1 benchmarks in mixed precision. Use `--ensemble=NPT` for constant pressure.

```bash
srun -A $SLURM_ACCOUNT -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --chdir="$trial" \
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

### CHARMM36 DHFR

CHARMM36 with explicit CHARMM TIP3P water also passed on one Leonardo A100 in Slurm job `64882374`. The run used the same 23,558-atom DHFR coordinates and timing protocol as the AMBER comparison above.

| Ensemble | Stock ms/step | cuPotential ms/step | Stock ns/day | cuPotential ns/day | Speedup |
|---|---:|---:|---:|---:|---:|
| NVT | 0.307 | 0.301 | 1125.4 | 1150.1 | 1.02× |
| NPT | 0.367 | 0.330 | 941.1 | 1048.2 | 1.11× |

cuPotential replaced the PME electrostatics force. OpenMM retained CHARMM’s separate Lennard-Jones and 1–4 terms. The adapted van der Waals check kept those terms in the double-precision reference and both comparison arms. The electrostatics and van der Waals checks passed. These timings come from one run, and the measured NVT gain was about 2%.

From the repository directory after setup, copy the wrapper to shared scratch alongside the extracted NVIDIA sample.

```bash
cp charmm-dhfr.py "$trial/"
srun -A $SLURM_ACCOUNT -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --chdir="$trial" \
  /bin/bash "$trial/run-native.sh" "$trial/charmm-dhfr.py"
```

See [the CHARMM36 run log](charmm-dhfr.log) for the full output. The upstream sample’s progress line still says `Amber14` because the label is hardcoded.
