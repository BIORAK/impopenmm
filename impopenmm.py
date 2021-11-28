# Using OpenMM with IMP

from __future__ import print_function
import IMP.atom
import IMP.container
import sys
#from LangevinDynamicsOpenMM import LangevinIntegratorOpenMM
import sys
import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# Create an IMP model and add a heavy atom-only protein from a PDB file
m = IMP.Model()
prot = IMP.atom.read_pdb('Peptide.pdb', m,
                         IMP.atom.NonWaterNonHydrogenPDBSelector())

# Read in the CHARMM heavy atom topology and parameter files
ff = IMP.atom.get_heavy_atom_CHARMM_parameters()

# Using the CHARMM libraries, determine the ideal topology (atoms and their
# connectivity) for the PDB file's primary sequence
topology = ff.create_topology(prot)

# Typically this modifies the C and N termini of each chain in the protein by
# applying the CHARMM CTER and NTER patches. Patches can also be manually
# applied at this point, e.g. to add disulfide bridges.
topology.apply_default_patches()

# Each atom is mapped to its CHARMM type. These are needed to look up bond
# lengths, Lennard-Jones radii etc. in the CHARMM parameter file. Atom types
# can also be manually assigned at this point using the CHARMMAtom decorator.
topology.add_atom_types(prot)

# Remove any atoms that are in the PDB file but not in the topology, and add
# in any that are in the topology but not the PDB.
IMP.atom.remove_charmm_untyped_atoms(prot)
topology.add_missing_atoms(prot)

# Construct Cartesian coordinates for any atoms that were added
topology.add_coordinates(prot)
topology.add_charges(prot)

# Generate and return lists of bonds, angles, dihedrals and impropers for
# the protein. Each is a Particle in the model which defines the 2, 3 or 4
# atoms that are bonded, and adds parameters such as ideal bond length
# and force constant. Note that bonds and impropers are explicitly listed
# in the CHARMM topology file, while angles and dihedrals are generated
# automatically from an existing set of bonds. These particles only define the
# bonds, but do not score them or exclude them from the nonbonded list.
bonds = topology.add_bonds(prot)
angles = ff.create_angles(bonds)
dihedrals = ff.create_dihedrals(bonds)
impropers = topology.add_impropers(prot)

# Add non-bonded interaction (in this case, Lennard-Jones). This needs to
# know the radii and well depths for each atom, so add them from the forcefield
# (they can also be assigned manually using the XYZR or LennardJones
# decorators):
ff.add_radii(prot)
ff.add_well_depths(prot)

# Get a list of all atoms in the protein, and put it in a container
atoms = IMP.atom.get_by_type(prot, IMP.atom.ATOM_TYPE)
cont = IMP.container.ListSingletonContainer(m, atoms)

# Add a restraint for the Lennard-Jones interaction. Again, this is built from
# a collection of building blocks. First, a ClosePairContainer maintains a list
# of all pairs of Particles that are close. A StereochemistryPairFilter is used
# to exclude atoms from this list that are bonded to each other or are involved
# in an angle or dihedral (1-3 or 1-4 interaction). Then, a
# LennardJonesPairScore scores a pair of atoms with the Lennard-Jones potential.
# Finally, a PairsRestraint is used which simply applies the
# LennardJonesPairScore to each pair in the ClosePairContainer.
nbl = IMP.container.ClosePairContainer(cont, 4.0)
pair_filter = IMP.atom.StereochemistryPairFilter()
pair_filter.set_bonds(bonds)
pair_filter.set_angles(angles)
pair_filter.set_dihedrals(dihedrals)
nbl.add_pair_filter(pair_filter)

########################################## OpenMM Code ################################################### 
# Add topology and paramters from IMP to OpenMM
openmmSys = System()
openmmTop = Topology()

impchain = IMP.atom.Chain(IMP.atom.get_by_type(prot, IMP.atom.CHAIN_TYPE)[0])
IMP.atom.add_radii(prot)

chain = openmmTop.addChain(str(impchain.get_id()))
residues = IMP.atom.get_by_type(prot, IMP.atom.RESIDUE_TYPE)
coord = []
for resi in residues:
    res = openmmTop.addResidue(resi.get_name(), chain)
    for particle in IMP.atom.Selection(resi).get_selected_particles():
      openmmTop.addAtom(IMP.atom.Atom(particle).get_atom_type(), element.Element.getByMass(IMP.atom.Mass(particle).get_mass()), res)
      openmmSys.addParticle(IMP.atom.Mass(particle).get_mass() * dalton)
      coord.append(IMP.core.XYZR(particle).get_coordinates())

# for atom in openmmTop.residues():
#   print(atom)

pos = np.array(coord) * angstrom
box = np.eye(3) * 4 * nanometers
#print(pos)

#Add harmonic bond interactions
harmonicBonds = HarmonicBondForce()
for bondparticle in bonds:
   atomtype = [IMP.atom.Atom(IMP.atom.Bond(bondparticle).get_bonded(i)) for i in range(2)]
   harmonicBonds.addBond(atomtype[0].get_particle_index().get_index(), 
                         atomtype[1].get_particle_index().get_index(), 
                         IMP.atom.Bond(bondparticle).get_length() * angstrom, 
                         IMP.atom.Bond(bondparticle).get_stiffness())
    
openmmSys.addForce(harmonicBonds)

#Add harmonic angle interactions
harmonicAngles = HarmonicAngleForce()
for angleparticle in angles:
  atomtype = [IMP.atom.Atom(IMP.atom.Angle(angleparticle).get_particle(i)) for i in range(3)]
  harmonicAngles.addAngle(atomtype[0].get_particle_index().get_index(), 
                          atomtype[1].get_particle_index().get_index(), 
                          atomtype[2].get_particle_index().get_index(), 
                          IMP.atom.Angle(angleparticle).get_ideal(), 
                          IMP.atom.Angle(angleparticle).get_stiffness())

openmmSys.addForce(harmonicAngles)

#Add dihedral interactions
dihedralTorsions = PeriodicTorsionForce()
for dihedralparticle in dihedrals:
    d = IMP.atom.Dihedral(dihedralparticle)
    atomtype = [IMP.atom.Atom(d.get_particle(i)) for i in range(4)]
    dihedralTorsions.addTorsion(atomtype[0].get_particle_index().get_index(), 
                                atomtype[1].get_particle_index().get_index(), 
                                atomtype[2].get_particle_index().get_index(),
                                atomtype[3].get_particle_index().get_index(),
                                d.get_multiplicity(), 
                                d.get_ideal(), 
                                d.get_stiffness())

openmmSys.addForce(dihedralTorsions)

# #Add nonbonded interactions to OpenMM per particle
nonbonded = NonbondedForce()
for particleindex in nbl.get_all_possible_indexes():
  sigma = ((IMP.core.XYZR(m.get_particle(particleindex)).get_radius()) * 2) / 2 ** (1/6)
  nonbonded.addParticle(IMP.atom.Charged(m.get_particle(particleindex)).get_charge(), sigma, IMP.atom.LennardJones(m.get_particle(particleindex)).get_well_depth()*4.184)

openmmSys.addForce(nonbonded)

# define the simulation
platform_name = 'CPU'
platform = Platform.getPlatformByName(platform_name)
if platform_name == 'CPU':
    properties = {}
else:
    properties = {'CudaPrecision': 'mixed'}
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 1.0*femtosecond)
simulation = Simulation(openmmTop, openmmSys, integrator, platform)
print(simulation)
#simulation.context.setPeriodicBoxVectors(box[0], box[1], box[2])
simulation.context.setPositions(pos)
simulation.context.setVelocities(300*kelvin)

# checkout the energy and force
state = simulation.context.getState(getEnergy=True, getForces=True)
print(state)
print(state.getPotentialEnergy())
for f in state.getForces():
    print(f)
