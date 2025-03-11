#!/usr/bin/env python
"""
Chromatin Dynamics Simulation using OpenMM.
- Brownian or Langevin dynamics via IntegratorManager.
- Various forces via ForceFieldManager.
- Platform selection via PlatformManager.
- State output, saving, and analysis via StateAnalyzer.
"""

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
# from optimization import HiCInversion

# -------------------------------------------------------------------
# ChromatinDynamics: Main simulation class
# -------------------------------------------------------------------
class ChromatinDynamics:
    def __init__(self, topology, integrator="langevin", platform_name="CUDA", output_dir="output"):
        self.system = System()
        self.topology = topology
        self.output_dir = output_dir
        self.num_particles = topology.getNumAtoms()

        self.platform_manager = PlatformManager(platform_name)
        self.integrator_manager = IntegratorManager(integrator=integrator)
        self.force_field_manager = ForceFieldManager(self.topology)
        self.simulation = None
        self.analyzer = None
    
    # def optimizer_setup(self,method='hicinv', hicmap=None):
    #     if method.lower() == 'hicinv':
    #         hicman = HiCManager(hicmap)
    #         self.optimizer = HiCInversion(hicman.get_hic_map())
    
    # def optimize_step(self):
    #     pass
            
    def read_sequence(self, seq_file):
        """Reads a sequence file and returns list of types."""
        with open(seq_file, "r") as f:
            return [line.split()[1] for line in f if line.strip()]

    def system_setup(self, mode='default', interaction_matrix=None, k_res=1.0, r_rep=1.0, chi=-0.02):
        """Set up system and force fields."""
        
        for _ in range(self.topology.getNumAtoms()):
                self.system.addParticle(1.0)
        self.force_field_manager.removeCOM(self.system)
        
        if mode == 'default':
            self.force_field_manager.add_harmonic_bonds(self.system)
            type_labels, interaction_matrix = self.force_field_manager._get_type_interaction_matrix('./type_interaction_table.csv')
            self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)
            
        elif mode=='harmtrap':
            self.force_field_manager.add_harmonic_bonds(self.system)
            self.force_field_manager.add_harmonic_trap(self.system, kr=k_res)
        
        elif mode=="harmtrap_with_self_avoidance":
            self.force_field_manager.add_harmonic_bonds(self.system)
            self.force_field_manager.add_harmonic_trap(self.system, k=k_res)
            self.force_field_manager.add_self_avoidance(self.system, r=r_rep)
            
        elif mode=="bad_solvent_collapse":
            
            type_labels = ["A", "B"]
            interaction_matrix = [
                [chi, 0.0],
                [0.0, 0.0],
            ]
            
            self.force_field_manager.add_harmonic_bonds(self.system)
            self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)
            

    def simulation_setup(self):
        """Set up simulation components."""
        integrator = self.integrator_manager.create_integrator()
        platform = self.platform_manager.get_platform()
        self.simulation = Simulation(self.topology, self.system, integrator, platform)
        self.analyzer = StateAnalyzer(self.simulation, output_dir=self.output_dir)
        # self.stability_manager = StabilityReporter(self.simulation,reportInterval=1000)
        # Initial random positions
        positions = np.random.random((self.num_particles, 3))
        self.simulation.context.setPositions(positions)
        print(f"[INFO] Chosen platform: {platform.getName()}")
        # Energy reporting
        self.simulation.reporters.append(
            StateDataReporter(os.path.join(self.output_dir, "energy_report.txt"), 1000,
                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True)
        )

    def run(self, n_steps):
        """Run simulation and check for stability issues."""
        start_time = time.time()
        self.simulation.step(n_steps)
        self.check_and_reinitialize_velocities()
        
        end_time = time.time()
        elapsed = end_time - start_time
        steps_per_sec = n_steps / elapsed
        print(f"[INFO] Step completed.")
        print(f"[INFO] Steps per second: {steps_per_sec:.0f} steps/s over {n_steps} steps (Elapsed time: {elapsed:.2f} s)")
        return steps_per_sec

    def check_and_reinitialize_velocities(self, kinetic_threshold=5.0, scale=1.0):
        """Check stability and reinitialize velocities if necessary."""
        state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        # velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometers/unit.picoseconds)
        e_kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / self.num_particles

        if (np.isnan(positions).any() or np.isnan(e_kinetic) or e_kinetic > kinetic_threshold ):
            print(f"[WARNING] Instability detected (eK={e_kinetic:.2f}). Reinitializing velocities.")
            self.reinitialize_velocities(scale)

    def reinitialize_velocities(self, scale=1.0):
        """Reinitialize random velocities with scaling."""
        sigma = np.sqrt(self.integrator_manager.temperature / self.system.getParticleMass(0))
        velocities = np.random.normal(0, sigma, size=(self.num_particles, 3)) * scale
        self.simulation.context.setVelocities(unit.Quantity(velocities, unit.nanometers/unit.picoseconds))
        print("[INFO] Velocities reinitialized.")
        
    def print_force_info(self):
        """
        Prints a detailed summary of all forces in the system, including:
        - Force index, class, and human-readable name
        - Force group number
        - Number of particles and bonds (if applicable)
        - Energy contribution per particle (kJ/mol)
        """

        system = self.simulation.system
        context = self.simulation.context
        num_particles = system.getNumParticles()

        # Header
        print("\n{:<6} {:<30} {:<25} {:<8} {:<15} {:<12} {:<30}".format(
            "Index", "Force Class", "Force Name", "Group", "Num Particles", "Num Bonds", "Energy per Particle"
        ))
        print("-" * 160)

        # Loop over forces
        for i, force in enumerate(system.getForces()):
            group = force.getForceGroup()
            force_class = force.__class__.__name__

            # Proper way to get force name (use a force -> name map that you maintain)
            force_name = self.force_field_manager.force_name_map.get(i, "Unnamed")  # fallback if not found

            # Get energy for this force group
            state = context.getState(getEnergy=True, groups={group})
            total_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            per_particle_energy = total_energy / num_particles if num_particles > 0 else 0.0

            # Try to get num_particles and num_bonds
            num_particles_force = 'N/A'
            num_bonds_force = 'N/A'

            if hasattr(force, 'getNumParticles'):
                try:
                    num_particles_force = force.getNumParticles()
                except:
                    pass

            if hasattr(force, 'getNumBonds'):
                try:
                    num_bonds_force = force.getNumBonds()
                except:
                    pass

            # Print formatted row
            print(f"{i:<6} {force_class:<30} {force_name:<25} {group:<8} {num_particles_force:<15} {num_bonds_force:<12} {per_particle_energy:<30.5f}")

        print("-" * 160)
        print(f"[INFO] Total number of particles in system: {num_particles}\n")
# # from PolymerSimulation import *
# #!/usr/bin/env python
# """
# Polymer simulation using OpenMM with support for Brownian or Langevin dynamics.
# Force fields (e.g., harmonic bonds) are managed by a separate ForceFieldManager.
# The simulation reads sequence information from a seq.txt file, runs for a number
# of steps, reports energies, and saves positions in an HDF5 file.
# """

# import os
# import h5py
# import numpy as np
# from openmm import System
# from openmm.app import Simulation, StateDataReporter
# import openmm.unit as unit
# # from openmm.unit import picoseconds, nanometers, kelvin, femtosecond, daltons, kilojoules_per_mole

# from platforms import PlatformManager
# from integrators import IntegratorManager
# from forcefield import ForceFieldManager
# # -------------------------------------------------------------------
# # Polymer Simulation: Main simulation class
# # -------------------------------------------------------------------
# class ChromatinDynamics:
#     def __init__(self, topology, dynamics="brownian", platform_name="CUDA", output_dir="output"):
        
#         # OpenMM objects
#         self.system = System()
#         self.topology = topology
#         self.simulation = None
#         self.num_particles = self.topology.getNumAtoms()
#         self.output_dir = output_dir
#         # Managers
#         self.platform_manager = PlatformManager(platform_name)
#         self.integrator_manager = IntegratorManager(dynamics=dynamics)
#         self.force_field_manager = ForceFieldManager(self.topology)
#         # self.topo_manager = TopologyManager(chain_lens)
        
#     def read_sequence(self, seq_file):
#         """
#         Reads a sequence file (seq.txt) with lines like "1 A" or "2 B" and returns a list of types.
#         """
#         types = []
#         with open(seq_file, "r") as f:
#             for line in f:
#                 if line.strip():
#                     parts = line.split()
#                     # Here we simply return the second column; you can use this info to set parameters later.
#                     types.append(parts[1])
#         return types

#     def system_setup(self, mode='default', interaction_matrix=None):
#         """
#         Creates a simple system and topology.
#         """
#         # Create system: add one particle per bead.
#         if mode=='default':
#             for _ in range(self.topology.getNumAtoms()):
#                 self.system.addParticle(1.0)  # default mass = 1.0 dalton

#             type_labels = ["A", "B", "C", "D"]  # Assume matrix was defined for A, B, C, D

#             interaction_matrix = [
#                 [-0.25, 0.0, -0.15, -0.1],  # A interactions
#                 [0.0, -0.3, -0.18, -0.12], # B interactions
#                 [-0.15, -0.18, -0.35, -0.14],# C interactions
#                 [-0.1, -0.12, -0.14, -0.4],  # D interactions (might not be used)
#             ]

#             # Add harmonic bonds using the ForceFieldManager.
#             self.force_field_manager.add_harmonic_bonds(self.system)
#             self.force_field_manager.removeCOM(self.system)
#             # self.force_field_manager.add_self_avoidance(self.system)
#             # self.force_field_manager.add_cylindrical_confinement(self.system)
#             # self.force_field_manager.add_harmonic_trap(self.system, kr=1.0)
#             # self.force_field_manager.add_spherical_confinement(self.system)
#             self.force_field_manager.add_type_to_type_interaction(self.system, interaction_matrix, type_labels)

#     def simulation_setup(self):
#         """
#         Sets up the integrator, platform, simulation object, and energy reporter.
#         """
#         integrator = self.integrator_manager.create_integrator()
#         platform = self.platform_manager.get_platform()
#         self.simulation = Simulation(self.topology, self.system, integrator, platform)

#         # Set initial positions along a straight line (can be changed as needed)
#         positions = np.random.random((self.num_particles,3))
#         self.simulation.context.setPositions(positions)

#         # Add energy reporter for state data
#         energy_reporter = StateDataReporter(
#             os.path.join(self.output_dir, "energy_report.txt"),
#             1000,
#             step=True,
#             potentialEnergy=True,
#             kineticEnergy=True,
#             totalEnergy=True,
#             temperature=True
#         )
#         self.simulation.reporters.append(energy_reporter)

#     def run(self, n_steps):
#         """
#         Runs the simulation for n_steps.
#         """
#         self.simulation.step(n_steps)
#         self.check_and_reinitialize_velocities(mult=1.0)
        

#     def save_positions(self, h5_filename):
#         """
#         Saves the current positions to an HDF5 file.
#         """
#         state = self.simulation.context.getState(getPositions=True)
#         positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        
#         with h5py.File(h5_filename, "w") as hf:
#             hf.create_dataset("positions", data=positions)
#         print(f"Positions saved to {h5_filename}")
        
#     def check_and_reinitialize_velocities(self, kinetic_threshold=5.0, scale=1.0):
#         """
#         Checks for NaN values or excessive kinetic energy and reinitializes velocities if needed.
        
#         Args:
#             kinetic_threshold (float): Threshold for per-particle kinetic energy (default 5.0).
#             mult (float): Scaling factor for velocities upon reinitialization (default 1.0).
#         """
#         state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
#         positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
#         velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometers/unit.picoseconds)
#         e_kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / self.num_particles
        
#         if (np.isnan(positions).any() or np.isnan(velocities).any() or np.isnan(e_kinetic) or e_kinetic > kinetic_threshold):
#             print(f"[WARNING] Detected instability (eK={e_kinetic:.2f}). Reinitializing velocities.")
#             self.reinitialize_velocities(scale)
            
#     def reinitialize_velocities(self, scale=1.0):
#         """
#         Reinitializes random velocities with optional scaling factor.
        
#         Args:
#             mult (float): Scaling factor for velocities.
#         """
#         sigma = np.sqrt(self.integrator_manager.temperature / self.system.getParticleMass(0))  # assuming unit mass particles
#         velocities = np.random.normal(0, sigma, size=(self.num_particles, 3)) * scale
#         velocities_quantity = unit.Quantity(velocities, unit.nanometers/unit.picoseconds)
#         self.simulation.context.setVelocities(velocities_quantity)
#         print("[INFO] Velocities reinitialized.")

#     def print_force_info(self):
#         """
#         Prints information about the forces in the simulation's system:
#         - Force index and class name.
#         - The force group number.
#         - Number of particles (or bonds) for that force, if available.
#         - Energy contribution from that force group, based on the current context.
#         """

#         # Define header columns.
#         header = "{:<6} {:<25} {:<8} {:<15} {:<12} {:<30}".format(
#             "Index", "Force Class", "Group", "Num Particles", "Num Bonds", "Energy"
#         )
#         print(header)
#         print("-" * len(header))

#         for i, force in enumerate(self.system.getForces()):
#             group = force.getForceGroup()
#             # print(force.getEnergyFunction())
#             force_name = force.__class__.__name__
#             # Get energy contribution from this force group.
#             state = self.simulation.context.getState(getEnergy=True, getVelocities=True, getForces=True, getPositions=True,groups={group})
#             energy = state.getPotentialEnergy()/(self.topology.getNumAtoms()*unit.kilojoule_per_mole)
#             forceval = state.getForces(asNumpy=True)/(unit.kilojoule_per_mole/unit.angstrom)
#             vel = state.getVelocities(asNumpy=True)/(self.topology.getNumAtoms()*unit.angstrom/unit.picosecond)
#             pos = state.getPositions(asNumpy=True)/(unit.nanometer)

#             # Try to get number of particles or bonds.
#             num_particles = "N/A"
#             num_bonds = "N/A"
#             if hasattr(force, "getNumParticles"):
#                 try:
#                     num_particles = force.getNumParticles()
#                 except Exception:
#                     pass
#             elif hasattr(force, "getNumBonds"):
#                 try:
#                     num_bonds = force.getNumBonds()
#                 except Exception:
#                     pass            

#             line = "{:<6} {:<25} {:<8} {:<15} {:<12} {:<30} {:<10}".format(
#                 i, force_name, group, num_particles, num_bonds, str(energy),  str(pos[0])
#             )
#             print(line)
