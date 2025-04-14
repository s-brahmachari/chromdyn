import numpy as np
import random
from Logger import LogManager
import h5py
import openmm.unit as unit
from openmm import CMMotionRemover
import os

class SaveStructure:
    def __init__(self, reportFile, reportInterval=1000):
        self.filename = reportFile
        self.reportInterval = reportInterval
        mode = self.filename.split('.')[-1].lower()
        if mode not in ['cndb',]:
            raise ValueError(f"Unsupported file format: {mode}. Supported formats are 'cndb'.")
        self.mode = mode
        self.savestep = 0
        self.is_paused = False

        if self.mode == 'cndb':
            if os.path.exists(self.filename):
                backup_name = self.filename + ".bkp"
                # Remove old backup if it exists (optional, to avoid overwrite error)
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(self.filename, backup_name)

            # Create the new HDF5 file
            self.saveFile = h5py.File(self.filename, "w")

    def close(self):
        self.saveFile.close()
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Returns:
            A tuple containing the number of steps until the next report, and whether
            the positions, velocities, forces, energies, and parameters are needed.
        """
        steps = self.reportInterval - simulation.currentStep % self.reportInterval
        
        return (steps, True, False, False, False) # positions, velocites, forces, energies

    def report(self, simulation, state):
        """Generate a report.

        Args:
            simulation: The Simulation to generate a report for.
            state: The current State of the simulation.
        """
        if not self.is_paused:
            # Get positions as a NumPy array in nanometers
            data = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

            # Save the structure based on the specified mode
            if self.mode == 'cndb':
                self.saveFile[str(self.savestep)] = np.array(data)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            self.savestep += 1
    
class StabilityReporter:
    def __init__(self, filename, reportInterval=100, logger=None,
                 kinetic_threshold=5.0, potential_threshold=1000.0, scale=1.0):
        self.saveFile = open(filename, 'w')
        self.interval = reportInterval
        self.kinetic_threshold = kinetic_threshold
        self.potential_threshold = potential_threshold
        self.scale = scale
        self.logger = logger or LogManager().get_logger(__name__)
        self.logger.info(f"StabilityReporter initialized with thresholds: K.E. = {self.kinetic_threshold}, P.E. = {self.potential_threshold}")
        
    def describeNextReport(self, simulation):
        """
        Required by OpenMM Reporter interface. Specifies when the next report should occur.
        """
        steps_to_next_report = self.interval - simulation.currentStep % self.interval
        return (steps_to_next_report, False, False, False, True)  # positions, velocites, forces, energies

    def report(self, simulation, state):
        """
        Main method called at every reportInterval steps. Checks energies and reinitializes velocities if needed.
        """
        # Retrieve energies
        num_particles = simulation.system.getNumParticles()
        e_kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / num_particles
        e_potential = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / num_particles
        temperature = simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
        if e_kinetic > self.kinetic_threshold or abs(e_potential) > self.potential_threshold:
            self.logger.warning(f"<<INSTABILITY DETECTED>> at step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f} ")
            # If thresholds exceeded, reinitialize velocities
            seed = np.random.randint(100_000)
            simulation.context.setVelocitiesToTemperature(temperature, seed)
            self.saveFile.write(f"Step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f} | Reinitialized velocities.\n")
            self.saveFile.flush()  # Ensure the file is written immediately

class EnergyReporter:
    def __init__(self, filename, force_field_manager, reportInterval=1000, reportForceGrp=False):
        self.saveFile = open(filename, 'w')
        self.interval = reportInterval
        self.ff_man = force_field_manager
        self.report_force_grp = reportForceGrp
        self.is_initialized = False
        self.is_paused = False
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused=False
        
    def _make_header(self, simulation):
        system = simulation.system
        self.saveFile.write(f"{'Step':<10} {'Temperature':<12} {'Rag Gyr':<10} {'K.E./particle':<15} {'P.E./particle':<15}")
        if self.report_force_grp:
            for i, force in enumerate(system.getForces()):
                group = force.getForceGroup()
                force_name = self.ff_man.force_name_map.get(i, "Unnamed")
                force_name_w_grp = f"{force_name}({group})"
                self.saveFile.write(f"{force_name_w_grp:<20}")
        self.saveFile.write("\n")
        self.saveFile.flush()
    
    def _initialize_constants(self, simulation):
        system = simulation.system
        # Compute the number of degrees of freedom.
        dof = 0
        for i in range(system.getNumParticles()):
            if system.getParticleMass(i) > 0*unit.dalton:
                dof += 3
        for i in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(i)
            if system.getParticleMass(p1) > 0*unit.dalton or system.getParticleMass(p2) > 0*unit.dalton:
                dof -= 1
        if any(type(system.getForce(i)) == CMMotionRemover for i in range(system.getNumForces())):
            dof -= 3
        self._dof = dof
        
        self.is_initialized = True
        
    def describeNextReport(self, simulation):
        """
        Required by OpenMM Reporter interface. Specifies when the next report should occur.
        """
        steps_to_next_report = self.interval - simulation.currentStep % self.interval
        return (steps_to_next_report, True, False, False, True)  # positions, velocites, forces, energies

    def report(self, simulation, state):
        """
        Main method called at every reportInterval steps. Checks energies and reinitializes velocities if needed.
        """
        if not self.is_initialized: 
            self._initialize_constants(simulation)
            self._make_header(simulation)

        if not self.is_paused:            
            system = simulation.system
            context = simulation.context
            num_particles = system.getNumParticles()
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            Rg = compute_RG(positions)
            # temperature = simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
            integrator = simulation.context.getIntegrator()
            if hasattr(integrator, 'computeSystemTemperature'):
                temperature = integrator.computeSystemTemperature().value_in_unit(unit.kelvin)
            else:
                temperature = (2*state.getKineticEnergy()/(self._dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)    
            ke_per_particle = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / num_particles
            pe_per_particle = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / num_particles
            self.saveFile.write(f"{simulation.currentStep:<10} {temperature:<12.3f} {Rg:<10.4f} {ke_per_particle:<15.4f} {pe_per_particle:<15.4f}") 
            
            if self.report_force_grp:
                for i, force in enumerate(system.getForces()):
                    group = force.getForceGroup()
                    state_grp = context.getState(getEnergy=True, groups={group})
                    pot_energy = state_grp.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    pe_grp_per_particle = pot_energy / num_particles
                    self.saveFile.write(f"{pe_grp_per_particle:<20.4f}")
            
            self.saveFile.write("\n")
            self.saveFile.flush()
       
def compute_RG(positions):
    center_of_mass = np.mean(positions, axis=0)
    squared_distances = np.sum((positions - center_of_mass) ** 2, axis=1)
    Rg = np.sqrt(np.mean(squared_distances))
    return Rg

def gen_structure(mode, num_steps, **kwargs):
    step_size = kwargs.get('step_size', 1)
    max_restarts = kwargs.get('max_restarts', 5000)
    logger = kwargs.get('logger', LogManager().get_logger(__name__))
    
    if mode.lower()=='saw3d':
        try:
            path, msg = get_pos_3Dsaw(num_steps, step_size, max_restarts)
            logger.info(msg)
        except RuntimeError as e:
            path, msg = get_pos_3Drandom_walk(num_steps, step_size)
            msg = f"3D SAW failed! Returning a random walk instead. Error: {e}"
            logger.warning(msg)
        return path
    elif mode.lower()=='randomwalk':
        # Generate a random walk
        path, msg = get_pos_3Drandom_walk(num_steps, step_size)
        logger.info(msg)
        return path
    else:
        path, _ = get_pos_3Drandom_walk(num_steps, step_size)
        logger.warning(f"Invalid mode '{mode}'. Defaulting to random walk. Position shape: {path.shape}.")
        return path
    
def get_pos_3Dsaw(num_steps, step_size, max_restarts=5000):
        """
        Generate a 3D self-avoiding walk on a cubic lattice with restarts.
        
        Args:
            num_steps (int): Desired length of the walk.
            max_restarts (int): Maximum number of full retries if stuck.
            verbose (bool): Print progress info.

        Returns:
            path (np.ndarray): Array of shape (num_steps, 3) with 3D walk positions.
        """
        directions = [
            np.array([step_size, 0, 0]), np.array([-step_size, 0, 0]),
            np.array([0, step_size, 0]), np.array([0, -step_size, 0]),
            np.array([0, 0, step_size]), np.array([0, 0, -step_size]),
        ]

        for attempt in range(1, max_restarts + 1):
            position = np.array([0, 0, 0])
            visited = set()
            visited.add(tuple(position))
            path = [position.copy()]
            success = True

            for step in range(1, num_steps):
                # List all valid next positions
                valid_moves = []
                for d in directions:
                    candidate = tuple(position + d)
                    if candidate not in visited:
                        valid_moves.append(d)

                if not valid_moves:
                    success = False
                    # if verbose:
                    #     self.logger.info(f"[Attempt {attempt}] Stuck at step {step}, restarting...")
                    break  # restart the whole walk

                move = random.choice(valid_moves)
                position += move
                visited.add(tuple(position))
                path.append(position.copy())

            if success:
                path = np.array(path)
                msg = f"3D SAW created after {attempt} attempt(s). Position shape: {path.shape}"
                return path, msg
        raise RuntimeError(f"Failed to generate a self-avoiding walk after {max_restarts} attempts.")
            
def get_pos_3Drandom_walk(num_steps, step_size):
    # Generate random directions on the unit sphere
    directions = np.random.normal(size=(num_steps, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize
    steps = directions * step_size  # Scale to step size
    positions = np.cumsum(steps, axis=0)  # Cumulative sum for path
    msg = f"Random walk created. Position shape: {positions.shape}"
    return positions, msg  # Shape: (num_steps, 3)

def save_pdb(chrom_dyn_obj, **kwargs):
    
    filename = kwargs.get('filename', 
                          os.path.join(chrom_dyn_obj.output_dir, f"{chrom_dyn_obj.name}_{chrom_dyn_obj.simulation.currentStep}.pdb"))
    state = chrom_dyn_obj.simulation.context.getState(getPositions=True)
    data = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    with open(filename, 'w') as pdb_file:
        pdb_file.write(f"TITLE     {chrom_dyn_obj.name} - chain 0\n")
        pdb_file.write(f"MODEL     {chrom_dyn_obj.simulation.currentStep}\n")
        totalAtom = 1
        for i, line in enumerate(data):
            resName = 'GLY'
            pdb_line = (
                f"ATOM  {totalAtom:5d}  CA  {resName} A{totalAtom:4d}    "
                f"{line[0]:8.3f}{line[1]:8.3f}{line[2]:8.3f}  1.00  0.00           C\n"
            )
            pdb_file.write(pdb_line)
            totalAtom += 1
        pdb_file.write(f"ENDMDL\n")