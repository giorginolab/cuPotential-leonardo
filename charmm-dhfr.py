"""CHARMM36 DHFR through NVIDIA's solvated-systems sample.

OpenMM's CHARMM36 force field puts Lennard-Jones in a separate CustomNonbondedForce with
tabulated per-type-pair coefficients. cuPotential cannot convert that force ("tabulated
functions are not supported"), so it would run unaccelerated next to the plugin's PME and
cap the speedup at ~1.0x. When no pair deviates from Lorentz-Berthelot mixing (no NBFIX),
the same LJ can be expressed as ordinary per-atom sigma/epsilon on the NonbondedForce, which
the plugin accelerates natively. This script does that. If NBFIX overrides are present (CHARMM36m),
they go into a small interaction-group CustomNonbondedForce that holds only the deviation from
Lorentz-Berthelot mixing, so the total LJ is unchanged.
The 1-4 LJ stays in OpenMM's small LennardJones14 CustomBondForce.
"""
from pathlib import Path
import os
import sys

sample_dir = Path(__file__).parent / "cuPotential-0.3.5/plugins/openmm/samples/solvated_systems"
sys.path.insert(0, str(sample_dir))
import numpy as np
import sample
from openmm import app, unit

mm = sample.mm
# charmm36 (default) or charmm36_2024, which uses the CHARMM36m protein parameters
FORCE_FIELD = os.environ.get("CHARMM_FF", "charmm36")


def fold_lj_into_nonbonded(system):
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    index, lj = next((i, f) for i, f in enumerate(system.getForces())
                     if isinstance(f, mm.CustomNonbondedForce) and f.getName() == "LennardJones")
    acoef_fn, bcoef_fn = lj.getTabulatedFunction(0), lj.getTabulatedFunction(1)
    n = acoef_fn.getFunctionParameters()[0]
    a = np.array(acoef_fn.getFunctionParameters()[2]).reshape(n, n)
    b = np.array(bcoef_fn.getFunctionParameters()[2]).reshape(n, n)
    da, db = np.diag(a), np.diag(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(db > 0, da / db, 0.0) ** (1 / 6)          # acoef/bcoef = sigma^6
        epsilon = np.where(da > 0, db ** 2 / (4 * da), 0.0)         # bcoef^2/(4 acoef) = epsilon
    sigma_ij = 0.5 * (sigma[:, None] + sigma[None, :])
    epsilon_ij = np.sqrt(epsilon[:, None] * epsilon[None, :])
    lorentz_berthelot = 4 * epsilon_ij * sigma_ij ** 12
    delta_a, delta_b = a - lorentz_berthelot, b - 4 * epsilon_ij * sigma_ij ** 6
    keep_a = np.abs(delta_a) > 1e-5 * np.abs(a) + 1e-12
    keep_b = np.abs(delta_b) > 1e-5 * np.abs(b) + 1e-12
    nbfix_types = [t for t in range(n) if keep_a[t].any() or keep_b[t].any()]
    if nbfix_types and lj.getUseLongRangeCorrection():
        raise ValueError("NBFIX with a dispersion correction is not supported by this script")
    if nbfix_types:
        # Sparse correction: only the NBFIX type pairs differ from Lorentz-Berthelot, and they
        # involve a handful of atoms, so OpenMM evaluates this small force cheaply.
        delta_a[~keep_a] = 0.0
        delta_b[~keep_b] = 0.0
        corr = mm.CustomNonbondedForce("dacoef(type1, type2)/r^12 - dbcoef(type1, type2)/r^6")
        corr.addTabulatedFunction("dacoef", mm.Discrete2DFunction(n, n, delta_a.flatten(order="F")))
        corr.addTabulatedFunction("dbcoef", mm.Discrete2DFunction(n, n, delta_b.flatten(order="F")))
        corr.addPerParticleParameter("type")
        corr.setName("NBFIXCorrection")
        corr.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        corr.setCutoffDistance(lj.getCutoffDistance())
        corr.setUseLongRangeCorrection(False)
        members = set()
        for i in range(system.getNumParticles()):
            t = int(lj.getParticleParameters(i)[0])
            corr.addParticle([t])
            if t in nbfix_types:
                members.add(i)
        for k in range(lj.getNumExclusions()):
            corr.addExclusion(*lj.getExclusionParticles(k))
        corr.addInteractionGroup(members, members)
        system.addForce(corr)
        print(f"NBFIX: {len(nbfix_types)} atom types, {len(members)} atoms in the correction force")
    for i in range(system.getNumParticles()):
        t = int(lj.getParticleParameters(i)[0])
        charge, _, _ = nb.getParticleParameters(i)
        nb.setParticleParameters(i, charge, sigma[t], epsilon[t])
    nb.setUseDispersionCorrection(lj.getUseLongRangeCorrection())
    system.removeForce(index)
    return system


def create_charmm_system(topology):
    ff = FORCE_FIELD
    forcefield = app.ForceField(f"{ff}.xml", f"{ff}/water.xml")
    system = forcefield.createSystem(
        topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
        ewaldErrorTolerance=5e-4, constraints=app.HBonds, rigidWater=True,
        hydrogenMass=1.5 * unit.amu,
        # charmm36.xml turns this off and CHARMM36/36m are parameterized without it;
        # charmm36_2024.xml leaves it unset, so OpenMM would enable it.
        useDispersionCorrection=False,
    )
    fold_lj_into_nonbonded(system)
    print("CHARMM36 forces:", [(type(f).__name__, f.getName()) for f in system.getForces()])
    return system


print(f"Using {FORCE_FIELD} with CHARMM TIP3P (LJ folded into NonbondedForce). "
      "The upstream progress label still says Amber14.")
sample.create_system = create_charmm_system
sample.SYSTEMS = ["dhfr"]
raise SystemExit(sample.main())
