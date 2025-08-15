import numpy as np
import h5py
from openmm.app import Simulation
from openmm import System, State
import openmm.unit as unit
from openmm import CMMotionRemover
import os
from Utilities import LogManager
from Analyzers import compute_RG
from pathlib import Path
from typing import Union, Optional, Tuple

class SaveStructure:
    def __init__(self, reportFile: Union[str, Path], reportInterval: int = 1000):
        self.filename: str = str(reportFile)
        self.reportInterval: int = reportInterval
        mode: str = self.filename.split('.')[-1].lower()
        if mode not in ['cndb']:
            raise ValueError(f"Unsupported file format: {mode}. Supported formats are 'cndb'.")
        self.mode: str = mode
        self.savestep: int = 0
        self.is_paused: bool = False

        if self.mode == 'cndb':
            if os.path.exists(self.filename):
                backup_name: str = self.filename + ".bkp"
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(self.filename, backup_name)

            self.saveFile: h5py.File = h5py.File(self.filename, "w")
    def close(self) -> None:
        self.saveFile.close()
    
    def pause(self) -> None:
        self.is_paused = True
    
    def resume(self) -> None:
        self.is_paused = False

    def describeNextReport(self, simulation: Simulation) -> tuple[int, bool, bool, bool, bool]:
        """Get information about the next report this object will generate."""
        steps: int = self.reportInterval - simulation.currentStep % self.reportInterval
        return (steps, True, False, False, False)  # positions, velocities, forces, energies

    def report(self, simulation: Simulation, state: State) -> None:
        """Generate a report."""
        if not self.is_paused:
            data: np.ndarray = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            if self.mode == 'cndb':
                self.saveFile[str(self.savestep)] = np.array(data)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            self.savestep += 1
                
class StabilityReporter:
    def __init__(self, 
                 filename: Union[str, Path], 
                 reportInterval: int = 100, 
                 logger: Optional[object] = None,
                 kinetic_threshold: float = 5.0, 
                 potential_threshold: float = 1000.0, 
                 scale: float = 1.0):
        self.saveFile = open(filename, 'w')
        self.interval: int = reportInterval
        self.kinetic_threshold: float = kinetic_threshold
        self.potential_threshold: float = potential_threshold
        self.scale: float = scale
        self.logger = logger or LogManager().get_logger(__name__)
        self.logger.info(f"StabilityReporter initialized with thresholds: K.E. = {self.kinetic_threshold}, P.E. = {self.potential_threshold}")
        
    def describeNextReport(self, simulation: Simulation) -> Tuple[int, bool, bool, bool, bool]:
        """
        Required by OpenMM Reporter interface. Specifies when the next report should occur.
        """
        steps_to_next_report = self.interval - simulation.currentStep % self.interval
        return (steps_to_next_report, False, False, False, True)  # positions, velocities, forces, energies

    def report(self, simulation: Simulation, state: State) -> None:
        """
        Main method called at every reportInterval steps. Checks energies and reinitializes velocities if needed.
        """
        num_particles = simulation.system.getNumParticles()
        e_kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / num_particles
        e_potential = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / num_particles
        
        
        if hasattr(simulation.integrator, 'getTemperature'):
            temperature = simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
        else:
            temperature = 240.0
        
        kBT = 0.008314 * temperature
        e_kinetic_expected = 1.5 * kBT

        if e_kinetic/e_kinetic_expected > self.kinetic_threshold or abs(e_potential)/e_kinetic_expected > self.potential_threshold:
    
            seed = np.random.randint(100_000)
            simulation.context.setVelocitiesToTemperature(temperature, seed)
            self.logger.warning(f"<<INSTABILITY | Reinitialized velocities>> at step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f}")
            self.saveFile.write(f"<<INSTABILITY | Reinitialized velocities>> Step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f}\n")
            self.saveFile.flush()

class EnergyReporter:
    def __init__(
        self,
        filename: Union[str, Path],
        force_field_manager,
        reportInterval: int = 1000,
        reportForceGrp: bool = False
    ):
        self.saveFile = open(filename, 'w')
        self.interval: int = reportInterval
        self.ff_man = force_field_manager
        self.report_force_grp: bool = reportForceGrp
        self.is_initialized: bool = False
        self.is_paused: bool = False
    
    def pause(self) -> None:
        self.is_paused = True
    
    def resume(self) -> None:
        self.is_paused = False
        
    def _make_header(self, simulation: Simulation) -> None:
        system: System = simulation.system
        self.saveFile.write(f"{'Step':<10} {'Temperature':<12} {'Rag Gyr':<10} {'K.E./particle':<15} {'P.E./particle':<15}")
        if self.report_force_grp:
            for i, force in enumerate(system.getForces()):
                group = force.getForceGroup()
                force_name = self.ff_man.force_name_map.get(i, "Unnamed")
                force_name_w_grp = f"{force_name}({group})"
                self.saveFile.write(f"{force_name_w_grp:<20}")
        self.saveFile.write("\n")
        self.saveFile.flush()
    
    def _initialize_constants(self, simulation: Simulation) -> None:
        system: System = simulation.system
        dof: int = 0
        for i in range(system.getNumParticles()):
            if system.getParticleMass(i) > 0 * unit.dalton:
                dof += 3
        for i in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(i)
            if system.getParticleMass(p1) > 0 * unit.dalton or system.getParticleMass(p2) > 0 * unit.dalton:
                dof -= 1
        if any(type(system.getForce(i)) == CMMotionRemover for i in range(system.getNumForces())):
            dof -= 3
        self._dof: int = dof
        self.is_initialized = True
        
    def describeNextReport(self, simulation: Simulation) -> tuple[int, bool, bool, bool, bool]:
        steps_to_next_report: int = self.interval - simulation.currentStep % self.interval
        return (steps_to_next_report, True, False, False, True)

    def report(self, simulation: Simulation, state: State) -> None:
        if not self.is_initialized: 
            self._initialize_constants(simulation)
            self._make_header(simulation)

        if not self.is_paused:            
            system: System = simulation.system
            context = simulation.context
            num_particles: int = system.getNumParticles()
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            Rg: float = compute_RG(positions)

            integrator = simulation.context.getIntegrator()
            if hasattr(integrator, 'computeSystemTemperature'):
                temperature: float = integrator.computeSystemTemperature().value_in_unit(unit.kelvin)
            else:
                temperature = (2 * state.getKineticEnergy() / (self._dof * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)

            ke_per_particle: float = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / num_particles
            pe_per_particle: float = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / num_particles
            self.saveFile.write(f"{simulation.currentStep:<10} {temperature:<12.3f} {Rg:<10.4f} {ke_per_particle:<15.4f} {pe_per_particle:<15.4f}") 
            
            if self.report_force_grp:
                for i, force in enumerate(system.getForces()):
                    group: int = force.getForceGroup()
                    state_grp = context.getState(getEnergy=True, groups={group})
                    pot_energy: float = state_grp.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    pe_grp_per_particle: float = pot_energy / num_particles
                    self.saveFile.write(f"{pe_grp_per_particle:<20.4f}")
            
            self.saveFile.write("\n")
            self.saveFile.flush()

def save_pdb(chrom_dyn_obj, **kwargs):
    filename = kwargs.get(
        'filename',
        os.path.join(
            chrom_dyn_obj.output_dir,
            f"{chrom_dyn_obj.name}_{chrom_dyn_obj.simulation.currentStep}.pdb"
        )
    )

    # Unique residue names for different chains
    residue_names_by_chain = [
        'GLY', 'ALA', 'SER', 'VAL', 'THR', 'LEU', 'ILE', 'ASN', 'GLN', 'ASP',
        'GLU', 'PHE', 'TYR', 'TRP', 'CYS', 'MET', 'HIS', 'ARG', 'LYS', 'PRO'
    ]

    # Get atomic positions
    state = chrom_dyn_obj.simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    topology = chrom_dyn_obj.topology  # OpenMM Topology

    with open(filename, 'w') as pdb_file:
        pdb_file.write(f"TITLE     {chrom_dyn_obj.name}\n")
        pdb_file.write(f"MODEL     {chrom_dyn_obj.simulation.currentStep}\n")

        atom_index = 0
        chain_index = -1
        for chain in topology.chains():
            chain_index += 1
            if chain_index > 9:
                chain_id = '9'  # Reuse chainID
            else:
                chain_id = str(chain_index)

            # Assign unique residue name per chain
            res_name = residue_names_by_chain[chain_index % len(residue_names_by_chain)]

            for residue in chain.residues():
                for atom in residue.atoms():
                    pos = positions[atom_index]
                    atom_serial = atom_index + 1
                    atom_name = 'CA'       # placeholder
                    res_seq = residue.index + 1            # constant or can be residue.index + 1
                    element = 'C'          # consistent with 'CA'

                    pdb_line = (
                        f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:>3s} {chain_id:1s}"
                        f"{res_seq:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  "
                        f"1.00  0.00           {element:>2s}\n"
                    )
                    pdb_file.write(pdb_line)
                    atom_index += 1

        pdb_file.write("ENDMDL\n")