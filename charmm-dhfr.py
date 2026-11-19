from pathlib import Path
import sys

sample_dir = Path(__file__).parent / "cuPotential-0.3.5/plugins/openmm/samples/solvated_systems"
sys.path.insert(0, str(sample_dir))
import sample
from openmm import app, unit

def create_charmm_system(topology):
    forcefield = app.ForceField("charmm36.xml", "charmm36/water.xml")
    system = forcefield.createSystem(
        topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
        ewaldErrorTolerance=5e-4, constraints=app.HBonds, rigidWater=True,
        hydrogenMass=1.5 * unit.amu,
    )
    print("CHARMM36 forces:", [(type(f).__name__, f.getName()) for f in system.getForces()])
    return system

original_vdw_only = sample.vdw_only

def charmm_vdw_only(system):
    # The AMBER helper strips the separate forces that carry CHARMM's LJ terms.
    serializer = sample.mm.XmlSerializer
    lj_forces = [serializer.serialize(force) for force in system.getForces()
                 if force.getName() in ("LennardJones", "LennardJones14")]
    if len(lj_forces) != 2:
        raise ValueError("Expected separate CHARMM36 Lennard-Jones and 1–4 terms")
    original_vdw_only(system)
    for force_xml in lj_forces:
        system.addForce(serializer.deserialize(force_xml))
    return system

print("Using CHARMM36 with CHARMM TIP3P. The upstream progress label still says Amber14.")
sample.create_system = create_charmm_system
sample.vdw_only = charmm_vdw_only
sample.SYSTEMS = ["dhfr"]
raise SystemExit(sample.main())
