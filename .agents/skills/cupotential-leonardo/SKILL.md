---
name: cupotential-leonardo
description: Set up, run, or troubleshoot NVIDIA cuPotential and stock OpenMM benchmarks on the Leonardo HPC cluster using uv and Slurm.
---

# cuPotential benchmarks on Leonardo

Read the repository’s [README](../../../README.md) for the installation commands and runnable examples. Use [the how-to](../../../cuPotential-leonardo-howto.md) for the verified configuration and prior results.

- Use the native `uv` workflow on shared scratch. Create a fresh setup when the prepared installation is unavailable or belongs to another user. Stage the package archive on the login node and unpack it on the compute node through the supplied runner. Login-node `/scratch_local` was unavailable on the tested GPU node.
- Request one GPU with `srun -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1`. Use the site CUDA module when available. On the verified R535 node, `cuda/12.6` supplied the site environment, the Python wheels supplied the CUDA 13.4 runtime, and `cuda-compat-13-4` supplied the forward-compatible driver libraries. The runner’s private glibc loader handled the wheel’s newer glibc requirement without changing the host. Recheck the driver and available modules if the node image or package versions differ.
- For a cuPotential speedup comparison, run NVIDIA’s solvated-systems sample. Capture the Slurm job ID and log, check for `PASS`, and report NVT and NPT speedups. For a stock OpenMM throughput request, run OpenMM’s `benchmark.py` and report ns/day with the test names, ensemble, and precision. The two suites use different force-field and timing settings.
