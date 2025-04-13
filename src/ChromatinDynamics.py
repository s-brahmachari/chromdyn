import os
import time
from openmm import System
from openmm.app import Simulation
import openmm.unit as unit

from Platforms import PlatformManager
from Integrators import IntegratorManager
from Forcefield import ForceFieldManager
from Logger import LogManager
from Utilities import gen_structure, SaveStructure, StabilityReporter, EnergyReporter

class ChromatinDynamics:
    """
    Chromatin Dynamics simulation using OpenMM with modular managers:
    - System setup, forces, integrators, analysis, and logging.
    """

    def __init__(self, topology, name='ChromatinDynamics', platform_name="CUDA", output_dir="output", console_stream=True):
        
        self.logger = LogManager(log_file=os.path.join(output_dir,name+'.log')).get_logger(__name__, console=console_stream)
        
        self.logger.info('*' * 60)
        self.logger.info(f"{'Chromatin Dynamics':^{60}}")
        self.logger.info('*' * 60)  # Center the title
        self.name = name
        self.system = System()
        self.topology = topology
        self.output_dir = output_dir
        self.logger.info(f"Storing output in {self.output_dir}")
        self.num_particles = topology.getNumAtoms()
        for _ in range(self.num_particles):
            self.system.addParticle(1.0) # adding mass=1.0, modify for virtual particles
        self.logger.info(f"System initialized with {self.system.getNumParticles()} particles.")
            
        self.platform_manager = PlatformManager(platform_name, logger=self.logger)
        self.force_field_manager = ForceFieldManager(self.topology, self.system, logger=self.logger, Nonbonded_cutoff=5.0)
        self.logger.info('force_field_manager initialized. Use this to add forces to the system before setting up simulation.')
        self.simulation = None
        self.logger.info("-"*60)
    
    def simulation_setup(self, **kwargs):
        """Sets up the integrator, platform, context"""
        
        init_struct=kwargs.get('init_struct', 'randomwalk')
        integrator=kwargs.get('integrator', "langevin")
        temperature=kwargs.get('temperature', 120.0)
        timestep=kwargs.get('timestep', 0.01)
        save_pos = kwargs.get('save_pos', True)
        save_energy = kwargs.get('save_energy', True)
        stability_report_interval = kwargs.get('stability_report_interval', 500)
        energy_report_interval = kwargs.get('energy_report_interval', 1000)
        pos_report_interval = kwargs.get('pos_report_interval', 1000)
        
        self.logger.info("-"*60)
        self.integrator_manager = IntegratorManager(integrator=integrator, temperature=temperature, logger=self.logger, timestep=timestep)
        self.simulation = Simulation(self.topology, 
                                     self.system, 
                                     self.integrator_manager.integrator, 
                                     self.platform_manager.get_platform())

        self.logger.info("Setting up context...")
        positions = gen_structure(mode=init_struct, num_steps=self.num_particles, logger=self.logger)
        # self.logger.info(msg)
        self.simulation.context.setPositions(positions)
        self.logger.info(f"Simulation set up complete!")
        # self.logger.info("-"*60)
        self.print_force_info()
        instability_report_file = os.path.join(self.output_dir, self.name+"_stability_report.txt")
        self.simulation.reporters.append(StabilityReporter(instability_report_file, 
                                                           reportInterval=stability_report_interval, 
                                                           logger=self.logger,
                                                           kinetic_threshold=1e5,
                                                           potential_threshold=1e5)
                                         )
        self.logger.info(f"Creating Instability report at {instability_report_file}.")
            
        if save_energy:
            energy_report_file = os.path.join(self.output_dir, self.name+"_energy_report.txt")
            self.energy_reporter = EnergyReporter(energy_report_file, 
                                                  self.force_field_manager, 
                                                  reportInterval=energy_report_interval, 
                                                  reportForceGrp=True,)
            self.simulation.reporters.append(self.energy_reporter)
            self.logger.info(f"Created Energy reporter at {energy_report_file}.")
        
        if save_pos:
            position_report_file = os.path.join(self.output_dir, self.name+"_positions.cndb")
            self.pos_reporter = SaveStructure(position_report_file, 
                                              reportInterval=pos_report_interval,)
            self.simulation.reporters.append(self.pos_reporter)
            self.logger.info(f"Created Position reporter at {position_report_file}.")
        
    def run(self, n_steps, verbose=True):
        """Runs the simulation and reports performance."""
        if verbose:
            self.logger.info("-"*60)
            self.logger.info(f"Running simulation for {n_steps} steps...")
        start_time = time.time()
        self.simulation.step(n_steps)
        elapsed = time.time() - start_time
        steps_per_sec = n_steps / elapsed
        if verbose:
            self.logger.info(f"Completed {n_steps} steps in {elapsed:.2f}s ({steps_per_sec:.0f} steps/s)")
            self.logger.info("-"*60)
        return steps_per_sec

    def print_force_info(self):
        """Logs a formatted summary of forces and per-particle energy."""
        system = self.simulation.system
        context = self.simulation.context
        num_particles = system.getNumParticles()
        
        self.logger.info("-" * 120)
        self.logger.info(f"{'Index':<6} {'Force Class':<30} {'Force Name':<20} {'Group':<8} "
                         f"{'Particles':<12} {'Bonds':<12} {'Exclusions':<12} {'P.E./Particle':<20}")
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
        
        state = context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        kin_energy = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        
        self.logger.info("-" * 120)
        self.logger.info(f"Total number of particles: {num_particles} | K.E. per particle: {kin_energy/self.num_particles:.3f} | P.E. per particle: {pot_energy/self.num_particles:.3f}")
        self.logger.info("-" * 120)
        