from logger import LoggerManager
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


class ChromatinDynamics:
    """
    Chromatin Dynamics simulation using OpenMM with modular managers:
    - System setup, forces, integrators, analysis, and logging.
    """

    def __init__(self, topology, integrator="langevin", platform_name="CUDA", output_dir="output"):
        self.logger = LoggerManager().get_logger(__name__)
        
        self.logger.info('\n\n'+'*' * 100 + f"\n\n{'Chromatin Dynamics':^{100}}\n\n" + '*' * 100 +'\n')  # Center the title
        
        self.system = System()
        self.topology = topology
        self.output_dir = output_dir
        self.num_particles = topology.getNumAtoms()

        self.platform_manager = PlatformManager(platform_name, logger=self.logger)
        self.integrator_manager = IntegratorManager(integrator=integrator, logger=self.logger)
        self.force_field_manager = ForceFieldManager(self.topology, logger=self.logger)

        self.simulation = None
        self.analyzer = None

        

    def read_sequence(self, seq_file):
        """Reads sequence file and returns list of types."""
        self.logger.info(f"Reading sequence from file: {seq_file}")
        with open(seq_file, "r") as f:
            return [line.split()[1] for line in f if line.strip()]

    def system_setup(self, mode='default', interaction_matrix=None, k_res=1.0, r_rep=0.5, chi=-0.02):
        """Configures system with appropriate force fields based on mode."""
        self.logger.info("-"*60)
        self.logger.info(f"Setting up system with mode='{mode}'")

        self.logger.info(f"Adding {self.topology.getNumAtoms()} particles ...")
        for _ in range(self.topology.getNumAtoms()):
            self.system.addParticle(1.0)
        self.force_field_manager.removeCOM(self.system)

        if mode == 'default':
            self.force_field_manager.add_harmonic_bonds(self.system)
            type_labels, interaction_matrix = self.force_field_manager._get_type_interaction_matrix('./type_interaction_table.csv')
            self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)

        elif mode == 'harmtrap':
            self.force_field_manager.add_harmonic_bonds(self.system)
            self.force_field_manager.add_harmonic_trap(self.system, kr=k_res)

        elif mode == "harmtrap_with_self_avoidance":
            self.force_field_manager.add_harmonic_bonds(self.system)
            self.force_field_manager.add_harmonic_trap(self.system, kr=k_res)
            self.force_field_manager.add_self_avoidance(self.system, r=r_rep)
        
        elif mode == "saw":
            self.force_field_manager.add_harmonic_bonds(self.system)
            self.force_field_manager.add_self_avoidance(self.system, r=r_rep)
        elif mode == "rouse":
            self.force_field_manager.add_harmonic_bonds(self.system)
        elif mode == "bad_solvent_collapse":
            type_labels = ["A", "B"]
            interaction_matrix = [[chi, 0.0], [0.0, 0.0]]
            self.force_field_manager.add_harmonic_bonds(self.system)
            self.force_field_manager.add_self_avoidance(self.system, r=r_rep)
            self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)
            # self.force_field_manager.add_harmonic_trap(self.system, kr=0.001)
        self.logger.info("System set up complete!")
        self.logger.info("-"*60)

    def simulation_setup(self):
        """Sets up the integrator, platform, context, and analyzer."""
        integrator = self.integrator_manager.create_integrator('langevin')
        platform = self.platform_manager.get_platform()
        self.logger.info("-"*60)
        self.simulation = Simulation(self.topology, self.system, integrator, platform)
        self.analyzer = StateAnalyzer(self.simulation, output_dir=self.output_dir)
        self.logger.info("Setting up context...")
        self.logger.info("Random position initialization in context")
        positions = np.random.random((self.num_particles, 3))
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
            self.logger.info(f"Running simulation for {n_steps} steps...")

        start_time = time.time()
        self.simulation.step(n_steps)
        self.check_and_reinitialize_velocities()

        elapsed = time.time() - start_time
        steps_per_sec = n_steps / elapsed

        if verbose:
            self.logger.info(f"Completed {n_steps} steps in {elapsed:.2f}s | {steps_per_sec:.0f} steps/s.")
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
        self.logger.info(f"{'Index':<6} {'Force Class':<30} {'Force Name':<25} {'Group':<8} "
                         f"{'Particles':<12} {'Bonds':<12} {'Energy/Particle':<20}")
        self.logger.info("-" * 120)

        for i, force in enumerate(system.getForces()):
            group = force.getForceGroup()
            force_class = force.__class__.__name__
            force_name = self.force_field_manager.force_name_map.get(i, "Unnamed")
            state = context.getState(getEnergy=True, groups={group})
            total_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            per_particle_energy = total_energy / num_particles if num_particles else 0.0

            num_particles_force = getattr(force, 'getNumParticles', lambda: 'N/A')()
            num_bonds_force = getattr(force, 'getNumBonds', lambda: 'N/A')()

            self.logger.info(
                f"{i:<6} {force_class:<30} {force_name:<25} {group:<8} "
                f"{num_particles_force:<12} {num_bonds_force:<12} {per_particle_energy:<30.5f}"
            )

        self.logger.info("-" * 120)
        self.logger.info(f"Total number of particles: {num_particles}")
        self.logger.info("-" * 120)
        self.logger.info("=" * 120)