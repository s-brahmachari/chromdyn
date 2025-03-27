import os
import h5py
import time
import numpy as np
from openmm import System
from openmm.app import Simulation, StateDataReporter
import openmm.unit as unit

from platforms import PlatformManager
from integrators import IntegratorManager
from forcefield import ForceFieldManager
from analyzers import StateAnalyzer, HiCManager
from logger import LogManager

class ChromatinDynamics:
    """
    Chromatin Dynamics simulation using OpenMM with modular managers:
    - System setup, forces, integrators, analysis, and logging.
    """

    def __init__(self, topology, log_file='ChromatinDynamics.log', integrator="langevin", platform_name="CUDA", output_dir="output"):
        
        self.logger = LogManager(log_file=os.path.join(output_dir,log_file)).get_logger(__name__)
        
        self.logger.info('*' * 60)
        self.logger
        self.logger.info(f"{'Chromatin Dynamics':^{60}}")
        self.logger.info('*' * 60)  # Center the title
        # self.logger.info('\n\n')
        self.system = System()
        self.topology = topology
        self.output_dir = output_dir
        self.num_particles = topology.getNumAtoms()

        self.platform_manager = PlatformManager(platform_name, logger=self.logger)
        self.integrator_manager = IntegratorManager(integrator=integrator, temperature=120.0, logger=self.logger)
        self.force_field_manager = ForceFieldManager(self.topology, logger=self.logger)

        self.simulation = None
        self.analyzer = None

    def read_sequence(self, seq_file):
        """Reads sequence file and returns list of types."""
        self.logger.info(f"Reading sequence from file: {seq_file}")
        with open(seq_file, "r") as f:
            return [line.split()[1] for line in f if line.strip()]

    def system_setup(self, mode='default', **kwargs):
        
        type_table = str(kwargs.get('type_table', None))
        k_res = float(kwargs.get('k_res', 1.0))            # Default bond spring constant
        r_rep = float(kwargs.get('r_rep', 1.0))
        chi = float(kwargs.get('chi', 0.0))
        cmm_remove = kwargs.get('cmm_remove', None)
        k_bond = float(kwargs.get('k_bond', 30.0))
        r_bond = float(kwargs.get('r_bond', 1.0))
        k_angle = float(kwargs.get('k_angle', 2.0))
        k_rep = float(kwargs.get('k_rep', 5.0))
        E_rep = float(kwargs.get('E_rep', 4.0))
        theta0 = float(kwargs.get('theta0', 180.0))
        
        """Configures system with appropriate force fields based on mode."""
        # self.logger.info("-"*60)
        self.logger.info(f"Setting up system with mode='{mode}'")

        self.logger.info(f"Adding {self.topology.getNumAtoms()} particles ...")
        self.logger.info("-"*60)
        for _ in range(self.topology.getNumAtoms()):
            self.system.addParticle(1.0)
            
        if cmm_remove: self.force_field_manager.removeCOM(self.system)

        if mode == 'default':
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            type_labels, interaction_matrix = self.force_field_manager._get_type_interaction_matrix('./type_interaction_table.csv')
            self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)
        
        elif mode == 'debug':
            self.force_field_manager.add_harmonic_trap(self.system, kr=k_res)
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            self.force_field_manager.add_self_avoidance(self.system, Ecut=E_rep, k=k_rep, r=r_rep)
            type_labels, interaction_matrix = self.force_field_manager._get_type_interaction_matrix('./type_interaction_table.csv')
            self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)
            
        elif mode == 'harmtrap':
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            self.force_field_manager.add_harmonic_trap(self.system, kr=k_res)

        elif mode == "harmtrap_with_self_avoidance":
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            self.force_field_manager.add_harmonic_trap(self.system, kr=k_res)
            self.force_field_manager.add_self_avoidance(self.system, Ecut=E_rep, k=k_rep, r=r_rep)
        
        elif mode == "saw":
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            self.force_field_manager.add_self_avoidance(self.system, Ecut=E_rep, k=k_rep, r=r_rep)
            
        elif mode=="saw_stiff_backbone":
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            self.force_field_manager.add_self_avoidance(self.system, Ecut=E_rep, k=k_rep, r=r_rep)
            self.force_field_manager.add_harmonic_angles(self.system, theta0=theta0, k_angle=k_angle)
            
        elif mode=="saw_stiff_backbone_bad_solvent":
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            self.force_field_manager.add_self_avoidance(self.system, Ecut=E_rep, k=k_rep, r=r_rep)
            self.force_field_manager.add_harmonic_angles(self.system, theta0=theta0, k_angle=k_angle)
            type_labels = ["A", "B"]
            interaction_matrix = [[chi, 0.0], [0.0, 0.0]]
            self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)
            
        elif mode == "gauss":
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            
        elif mode == "saw_bad_solvent":
            self.force_field_manager.add_harmonic_bonds(self.system, k=k_bond, r0=r_bond)
            self.force_field_manager.add_self_avoidance(self.system, Ecut=E_rep, k=k_rep, r=r_rep)
            type_labels = ["A", "B"]
            interaction_matrix = [[chi, 0.0], [0.0, 0.0]]
            self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)
            
        self.logger.info("System set up complete!")
        self.logger.info("-"*60)
    
    def get_pos_3Drandom_walk(self, num_steps, step_size):
        # Generate random directions on the unit sphere
        directions = np.random.normal(size=(num_steps, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize
        steps = directions * step_size  # Scale to step size
        positions = np.cumsum(steps, axis=0)  # Cumulative sum for path
        return positions  # Shape: (num_steps, 3)
    
    def simulation_setup(self):
        """Sets up the integrator, platform, context, and analyzer."""
        integrator = self.integrator_manager.create_integrator('langevin')
        platform = self.platform_manager.get_platform()
        self.logger.info("-"*60)
        self.simulation = Simulation(self.topology, self.system, integrator, platform)
        self.analyzer = StateAnalyzer(self.simulation, output_dir=self.output_dir)
        self.logger.info("Setting up context...")
        self.logger.info("Random position initialization in context")
        # positions = np.random.random((self.num_particles, 3))
        positions = self.get_pos_3Drandom_walk(self.num_particles, 1.0)
        self.simulation.context.setPositions(positions)
        
        self.logger.info(f"Simulation set up complete!")
        self.logger.info("-"*60)
        self.simulation.reporters.append(
            StateDataReporter(os.path.join(self.output_dir, "energy_report.txt"), 1000,
                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True)
        )
    
    def run(self, n_steps, verbose=True):
        """Runs the simulation and reports performance."""
        if verbose:
            self.logger.info("-"*60)
            self.logger.info(f"Running simulation for {n_steps} steps...")
            

        start_time = time.time()
        self.simulation.step(n_steps)
        self.check_and_reinitialize_velocities()

        elapsed = time.time() - start_time
        steps_per_sec = n_steps / elapsed

        if verbose:
            self.logger.info(f"Completed {n_steps} steps in {elapsed:.2f}s ({steps_per_sec:.0f} steps/s) | Radius of gyration: {self.analyzer.compute_RG():.2f}")
            self.logger.info("-"*60)
        return steps_per_sec

    def check_and_reinitialize_velocities(self, kinetic_threshold=5.0, scale=1.0):
        """Check for NaN or high kinetic energy; reinitialize velocities if unstable."""
        state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        e_kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / self.num_particles

        if np.isnan(positions).any() or np.isnan(e_kinetic) or e_kinetic > kinetic_threshold:
            self.logger.warning(
                f"[Instability] Kinetic energy = {e_kinetic:.2f} kJ/mol exceeds threshold {kinetic_threshold}. "
                f"Reinitializing velocities..."
            )
            self.reinitialize_velocities(scale)

    def reinitialize_velocities(self, scale=1.0):
        """Reinitialize velocities based on integrator temperature and mass."""
        sigma = np.sqrt(self.integrator_manager.temperature / (self.system.getParticleMass(0)/ unit.dalton))
        velocities = np.random.normal(0, sigma, size=(self.num_particles, 3)) * scale
        self.simulation.context.setVelocities(unit.Quantity(velocities, unit.nanometers / unit.picoseconds))
        self.logger.info("Velocities reinitialized.")

    def print_force_info(self):
        """Logs a formatted summary of forces and per-particle energy."""
        system = self.simulation.system
        context = self.simulation.context
        num_particles = system.getNumParticles()

        self.logger.info("-" * 120)
        self.logger.info(f"{'Index':<6} {'Force Class':<30} {'Force Name':<20} {'Group':<8} "
                         f"{'Particles':<12} {'Bonds':<12} {'Exclusions':<12} {'Energy/Particle':<20}")
        self.logger.info("-" * 120)

        for i, force in enumerate(system.getForces()):
            group = force.getForceGroup()
            force_class = force.__class__.__name__
            force_name = self.force_field_manager.force_name_map.get(i, "Unnamed")
            state = context.getState(getEnergy=True, groups={group})
            pot_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            kin_energy = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            per_particle_energy = pot_energy / num_particles if num_particles else 0.0

            num_particles_force = getattr(force, 'getNumParticles', lambda: 'N/A')()
            num_bonds_force = getattr(force, 'getNumBonds', lambda: 'N/A')()
            num_exclusions = getattr(force, 'getNumExclusions', lambda: 'N/A')()

            self.logger.info(
                f"{i:<6} {force_class:<30} {force_name:<20} {group:<8} "
                f"{num_particles_force:<12} {num_bonds_force:<12} {num_exclusions:<12} {per_particle_energy:<20.3f}"
            )

        self.logger.info("-" * 120)
        self.logger.info(f"Total number of particles: {num_particles} | Kinetic Energy per particle: {kin_energy/self.num_particles:.3f}")
        self.logger.info("-" * 120)
        