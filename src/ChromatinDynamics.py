import time
from pathlib import Path
from openmm import System
from openmm.app import Simulation, Topology
import openmm.unit as unit

from Platforms import PlatformManager
from Integrators import IntegratorManager
from Forcefield import ForceFieldManager
from Utilities import gen_structure, LogManager
from Reporters import SaveStructure, StabilityReporter, EnergyReporter

class ChromatinDynamics:
    """
    Chromatin Dynamics simulation using OpenMM with modular managers:
    - System setup, forces, integrators, analysis, and logging.
    """

    def __init__(self, topology: Topology, name: str = 'ChromatinDynamics', platform_name: str = "CUDA", output_dir: str = "output", console_stream: bool = True, mass: float = 1.0) -> None:
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = LogManager(log_file=self.output_dir / f"{name}.log").get_logger(__name__, console=console_stream)

        self.logger.info("*" * 60)
        self.logger.info(f"{'Chromatin Dynamics':^60}")
        self.logger.info("*" * 60)

        self.topology = topology
        self.system = System()
        self.num_particles = topology.getNumAtoms()
        for _ in range(self.num_particles):
            self.system.addParticle(mass)

        self.logger.info(f"System initialized with {self.num_particles} particles. Output directory: {self.output_dir}")

        self.platform_manager = PlatformManager(platform_name, logger=self.logger)
        self.force_field_manager = ForceFieldManager(self.topology, self.system, logger=self.logger, Nonbonded_cutoff=5.0)
        self.logger.info("force_field_manager initialized. Use this to add forces before running setup.")

        self.simulation = None
        self.reporters: dict[str, object] = {}

    def simulation_setup(self,
                         init_struct: str = 'randomwalk',
                         integrator: str = 'langevin',
                         temperature: float = 120.0,
                         timestep: float = 0.01,
                         save_pos: bool = True,
                         save_energy: bool = True,
                         stability_report_interval: int = 500,
                         energy_report_interval: int = 1000,
                         pos_report_interval: int = 1000) -> None:
        """Sets up the integrator, platform, context, and attaches reporters."""

        self.integrator_manager = IntegratorManager(
            integrator=integrator,
            temperature=temperature,
            logger=self.logger,
            timestep=timestep
        )

        self.simulation = Simulation(
            self.topology,
            self.system,
            self.integrator_manager.integrator,
            self.platform_manager.get_platform()
        )

        self.logger.info("Setting up simulation context...")
        positions = gen_structure(mode=init_struct, num_steps=self.num_particles, logger=self.logger)
        self.simulation.context.setPositions(positions)
        self.logger.info("Simulation context initialized.")

        self.print_force_info()

        if save_pos:
            path = self.output_dir / f"{self.name}_positions.cndb"
            self.reporters['position'] = SaveStructure(path, reportInterval=pos_report_interval)
            self.simulation.reporters.append(self.reporters['position'])
            self.logger.info(f"Position reporter created: {path}")

        if save_energy:
            path = self.output_dir / f"{self.name}_energy_report.txt"
            self.reporters['energy'] = EnergyReporter(
                path,
                self.force_field_manager,
                reportInterval=energy_report_interval,
                reportForceGrp=True
            )
            self.simulation.reporters.append(self.reporters['energy'])
            self.logger.info(f"Energy reporter created: {path}")

        path = self.output_dir / f"{self.name}_stability_report.txt"
        self.reporters['stability'] = StabilityReporter(
            path,
            reportInterval=stability_report_interval,
            logger=self.logger,
            kinetic_threshold=5.0,
            potential_threshold=5.0
        )
        self.simulation.reporters.append(self.reporters['stability'])
        self.logger.info(f"Stability reporter created: {path}")

    def run(self, n_steps: int, verbose: bool = True, report: bool = True) -> float:
        """Runs the simulation and reports performance."""
        if not self.simulation:
            raise RuntimeError("Simulation not initialized. Call simulation_setup() first.")

        if verbose:
            self.logger.info("-" * 60)
            self.logger.info(f"Running simulation for {n_steps} steps...")

        if not report:
            self.pause_reporters()

        start = time.time()
        try:
            self.simulation.step(n_steps)
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise

        elapsed = time.time() - start
        steps_per_sec = n_steps / elapsed

        if verbose:
            self.logger.info(f"Completed {n_steps} steps in {elapsed:.2f}s ({steps_per_sec:.0f} steps/s)")
            self.logger.info("-" * 60)

        if not report:
            self.resume_reporters()

        return steps_per_sec

    def pause_reporters(self) -> None:
        for name, reporter in self.reporters.items():
            if hasattr(reporter, 'pause'):
                reporter.pause()
                self.logger.info(f"Paused reporter: {name}")

    def resume_reporters(self) -> None:
        for name, reporter in self.reporters.items():
            if hasattr(reporter, 'resume'):
                reporter.resume()
                self.logger.info(f"Resumed reporter: {name}")

    def save_reports(self) -> None:
        if 'position' in self.reporters and hasattr(self.reporters['position'], 'close'):
            self.reporters['position'].close()
            self.logger.info("Closed position reporter file.")

    def print_force_info(self) -> None:
        """Logs a formatted summary of forces and per-particle energy."""
        context = self.simulation.context
        system = self.simulation.system
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
            per_particle_energy = pot_energy / num_particles if num_particles else 0.0

            num_particles_force = getattr(force, 'getNumParticles', lambda: 'N/A')()
            num_bonds_force = getattr(force, 'getNumBonds', lambda: 'N/A')()
            num_exclusions = getattr(force, 'getNumExclusions', lambda: 'N/A')()

            self.logger.info(
                f"{i:<6} {force_class:<30} {force_name:<20} {group:<8} "
                f"{num_particles_force:<12} {num_bonds_force:<12} {num_exclusions:<12} {per_particle_energy:<20.3f}"
            )

        total_state = context.getState(getEnergy=True)
        pot_energy = total_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        kin_energy = total_state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)

        self.logger.info("-" * 120)
        self.logger.info(f"Total particles: {num_particles} | K.E./particle: {kin_energy/num_particles:.3f} | P.E./particle: {pot_energy/num_particles:.3f}")
        self.logger.info("-" * 120)

    def __repr__(self) -> str:
        return f"<ChromatinDynamics(name={self.name}, particles={self.num_particles}, output_dir={self.output_dir})>"