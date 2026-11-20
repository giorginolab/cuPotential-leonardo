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

### CHARMM36 and CHARMM36m DHFR

CHARMM36 (Slurm job `64944038`) and CHARMM36m (job `64945303`) with explicit CHARMM TIP3P water passed on one Leonardo A100. The runs used the same 23,558-atom DHFR coordinates and timing protocol as the AMBER comparison above.

| Force field | Ensemble | Stock ms/step | cuPotential ms/step | Stock ns/day | cuPotential ns/day | Speedup |
|---|---|---:|---:|---:|---:|---:|
| CHARMM36 | NVT | 0.286 | 0.097 | 1207.0 | 3568.0 | 2.96× |
| CHARMM36 | NPT | 0.345 | 0.111 | 1001.4 | 3109.4 | 3.11× |
| CHARMM36m | NVT | 0.295 | 0.112 | 1173.3 | 3087.9 | 2.63× |
| CHARMM36m | NPT | 0.352 | 0.126 | 982.8 | 2750.0 | 2.80× |

CHARMM36m is OpenMM 8.4’s `charmm36_2024.xml` (protein parameters from `par_all36m_prot.prm`). Both arms use the same folded system, and all accuracy checks passed for both force fields.

**Why a naive CHARMM36 run shows no speedup.** OpenMM’s CHARMM36 force field puts Lennard-Jones in a separate `CustomNonbondedForce` with tabulated per-type-pair coefficients. cuPotential only replaces the `NonbondedForce` (the plugin rejects tabulated functions), so a stock `createSystem` run leaves LJ on OpenMM’s kernel. Stock OpenMM overlaps PME and that LJ kernel on separate streams. With the plugin, PME gets 2.2× faster (0.184 to 0.084 ms per force evaluation) but the unaccelerated LJ force (about 0.15 ms) becomes the critical path. Stock ran 0.307 ms/step and the plugin 0.301 (NVT), a 1.02× speedup, and 1.11× under NPT.

**The fix.** [charmm-dhfr.py](charmm-dhfr.py) moves LJ into the `NonbondedForce` as per-atom sigma and epsilon, which the plugin accelerates natively. The 1–4 LJ stays in OpenMM’s small `LennardJones14` bond force (about 0.02 ms). The folded CHARMM36 system reproduces the original stock energy to 0.01 kJ/mol, and the sample’s electrostatics and van der Waals accuracy checks pass. The stock arm also runs faster after the fold (0.286 versus 0.307 ms/step), so the table’s speedup is against the faster baseline. Against the original unfolded stock system, the NVT speedup is 3.2×.

**NBFIX (CHARMM36m).** CHARMM36 has no NBFIX pair overrides for this system, so the fold is exact. CHARMM36m does: 9 atom types (81 atoms of this DHFR system) deviate from Lorentz-Berthelot mixing. The script puts the Lorentz-Berthelot LJ in the `NonbondedForce` and the deviation, only for those type pairs, in a small interaction-group `CustomNonbondedForce` that OpenMM evaluates. The total LJ is unchanged: the folded CHARMM36m system matches the original unfolded energy to 0.003 kJ/mol, with a maximum force difference of 0.0035 kJ/mol/nm. The extra force costs about 0.015 ms/step, which is why CHARMM36m is somewhat slower than CHARMM36. The sample’s accuracy checks isolate the `NonbondedForce`, so the NBFIX correction is not part of those checks.

**Dispersion correction.** The script passes `useDispersionCorrection=False` for both force fields, because `charmm36.xml` disables it and `charmm36_2024.xml` leaves it to OpenMM’s default of on. It refuses a system that has both NBFIX and a dispersion correction.

From the repository directory after setup, copy the wrapper to shared scratch alongside the extracted NVIDIA sample.

```bash
cp charmm-dhfr.py "$trial/"
srun -A $SLURM_ACCOUNT -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --chdir="$trial" \
  /bin/bash "$trial/run-native.sh" "$trial/charmm-dhfr.py"
```

For CHARMM36m, add `CHARMM_FF=charmm36_2024` to the environment of that command.

See the [CHARMM36](charmm-dhfr.log) and [CHARMM36m](charmm36m-dhfr.log) run logs for the full output. The upstream sample’s progress line still says `Amber14` because the label is hardcoded.
