# Running cuPotential on Leonardo

NVIDIA’s full solvated-systems benchmark passed on one A100 in Slurm job `64873726`. The native run used Python 3.12, cuPotential 0.3.5, and OpenMM 8.4.0.post2. The stock and accelerated runs used the same GPU.

## Rerun the benchmark

For a fresh installation, follow [Install from scratch](README.md#install-from-scratch) on a Leonardo login node. Keep the `trial` variable from setup. To reuse the original installation, set `trial="$SCRATCH/cupotential-BMLcMk"` if you have access to it.

```bash
set -o pipefail
srun -A $SLURM_ACCOUNT --partition=boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 \
  --chdir="$trial" /bin/bash "$trial/run-native.sh" \
  2>&1 | tee "$trial/benchmark-rerun.log"
```

The explicit partition selects A100 nodes. The runner unpacks its dependencies into a temporary directory on the allocated node, launches NVIDIA’s sample, and cleans up when the job exits. A successful run ends in `PASS`.

## Run OpenMM’s own benchmarks

From a Leonardo login node, run DHFR and APOA1 with stock OpenMM on one A100. Use the same `trial` directory. Its `inputs` directory contains OpenMM 8.4’s benchmark script and both PDB files.

```bash
srun -A $SLURM_ACCOUNT --partition=boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 \
  --chdir="$trial" /bin/bash "$trial/run-native.sh" "$trial/inputs/benchmark.py" \
  --platform=CUDA --precision=mixed --test=pme,apoa1pme --ensemble=NVT \
  --seconds=10 --style=table
```

Use `--ensemble=NPT` for constant pressure, or `--test=pme` for DHFR alone. The table reports throughput in ns/day. OpenMM’s benchmark uses its own force-field and timing settings. For the stock-versus-cuPotential comparison, use NVIDIA’s benchmark above. [OpenMM 8.4 benchmark script](https://github.com/openmm/openmm/blob/8.4.0/examples/benchmarks/benchmark.py)

## Native setup

- The runner loads `module add cuda/12.6` and `module add gcc/12.2.0`. CUDA 12.6 is the newest site module available on the tested node. The release wheel’s dependencies supply the CUDA 13.4 runtime libraries.
- `setup-native.sh` stages the wheels with `uv`, targeting `x86_64-manylinux_2_38`, and writes `packages.tar` to shared scratch. Run it from the login node if the package archive needs rebuilding. It stages in `$TMPDIR` (or `/tmp`) by default; set `CUPOTENTIAL_STAGING_DIR` to override.
- Leonardo’s host glibc was 2.28. `libOpenMMNvidiaCUDA.so` in the wheel needs `GLIBC_2.38` (`__isoc23_strtol`, `__isoc23_sscanf`) and `GLIBC_2.34` (`dlopen`), so the runner starts Python through a private glibc 2.39 loader that `setup-native.sh` extracted to `runtime/glibc`.
- The tested A100 node ran driver 535.274.02. The runner puts the libraries unpacked from NVIDIA’s `cuda-compat-13-4` package on the private loader’s library path ahead of the system driver libraries.
- The job runs without Conda or Singularity. The pinned cuPotential source archive supplies the sample and BACE input. The APOA1 and DHFR inputs were fetched from OpenMM 8.4.0 and verified against the checksums expected by the sample.

See [NVIDIA’s installation requirements](https://docs.nvidia.com/cuda/cupotential/install) and [the cuPotential 0.3.5 release](https://github.com/NVIDIA/cuPotential/releases/tag/v0.3.5).

## Troubleshooting: no speedup on a new force field

cuPotential replaces only OpenMM's `NonbondedForce`. List the System's forces first (`[(type(f).__name__, f.getName()) for f in system.getForces()]`). If Lennard-Jones sits in a separate `CustomNonbondedForce`, as in OpenMM's CHARMM36 files, it stays on stock OpenMM and sets the step time. Stock OpenMM hides that cost by overlapping PME and the custom force on separate streams, so a faster PME alone gives about 1.0×. The plugin rejects `CustomNonbondedForce` with tabulated functions ("tabulated functions are not supported").

To diagnose, put each force in its own force group and time them with coordinates frozen, using a `CustomIntegrator` with `setIntegrationForceGroups`. For CHARMM36 DHFR, PME fell from 0.184 to 0.084 ms with the plugin while the LJ force stayed at about 0.15 ms.

The fix is to express the LJ as per-atom sigma and epsilon on the `NonbondedForce`. `charmm-dhfr.py` does this and moves any NBFIX deviation from Lorentz-Berthelot mixing into a small interaction-group `CustomNonbondedForce`. See the [README](README.md#charmm36-and-charmm36m-dhfr) for results. When you check such a change, also compare energy and forces with the original system. Mind the dispersion correction: `charmm36.xml` disables it and `charmm36_2024.xml` does not.

## Benchmark results

See the [verified results in the README](README.md#verified-results) for timings, throughput, and speedups.
