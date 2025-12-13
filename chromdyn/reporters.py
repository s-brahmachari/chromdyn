#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *
import numpy as np
import h5py
import json
from openmm.app import Simulation
from openmm import System, State, app
import openmm.unit as unit
from openmm import CMMotionRemover
import os
from .utilities import LogManager
from pathlib import Path
from typing import Union, Optional, Tuple
from .traj_utils import Analyzer


class SaveStructure:
    def __init__(
        self,
        report_file: Union[str, Path],
        PBC: bool,
        topology: app.Topology,
        reportInterval: int = 1000,
    ):
        self.filename: str = str(report_file)
        self.reportInterval: int = reportInterval
        self.topology = topology
        self.PBC = PBC
        mode: str = self.filename.split(".")[-1].lower()
        if mode not in ["cndb"]:
            raise ValueError(
                f"Unsupported file format: {mode}. Supported formats are 'cndb'."
            )
        self.mode: str = mode
        self.savestep: int = 0
        self.is_paused: bool = False

        if self.mode == "cndb":
            if os.path.exists(self.filename):
                backup_name: str = self.filename + ".bkp"
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(self.filename, backup_name)

            self.saveFile: h5py.File = h5py.File(self.filename, "w")

        # Initialize static datasets
        self._save_types()
        self._save_full_topology()

    def _save_types(self):
        """Extracts atom types (from element) and saves as a dataset."""
        type_list = []
        for atom in self.topology.atoms():
            # Use element symbol as type if it's an Element object, else use string directly
            t_str = (
                atom.element if isinstance(atom.element, str) else atom.element.symbol
            )
            type_list.append(t_str)

        # Save as variable-length string dataset
        dt = h5py.special_dtype(vlen=str)
        self.saveFile.create_dataset("types", data=np.array(type_list, dtype=dt))

    def _save_full_topology(self):
        """
        Serializes the OpenMM Topology into a JSON string and saves it.
        Structure:
        {
            "chains": [
                {"index": 0, "id": "C1", "residues": [...]}
            ],
            "bonds": [[atom_idx1, atom_idx2], ...]
        }
        """
        topology_data = {"chains": [], "bonds": []}

        # Helper to map Atom Object -> Global Index
        atom_to_index = {atom: i for i, atom in enumerate(self.topology.atoms())}

        # 1. Serialize Hierarchy (Chain -> Residue -> Atom)
        for chain in self.topology.chains():
            chain_dict = {"index": chain.index, "id": chain.id, "residues": []}
            for res in chain.residues():
                res_dict = {
                    "index": res.index,
                    "name": res.name,
                    "id": res.id,
                    "atoms": [],
                }
                for atom in res.atoms():
                    atom_dict = {
                        "index": atom.index,
                        "name": atom.name,
                        "type": (
                            atom.element
                            if isinstance(atom.element, str)
                            else atom.element.symbol
                        ),
                    }
                    res_dict["atoms"].append(atom_dict)
                chain_dict["residues"].append(res_dict)
            topology_data["chains"].append(chain_dict)

        # 2. Serialize Bonds
        # Store as list of [index_A, index_B]
        for bond in self.topology.bonds():
            a1_idx = atom_to_index[bond.atom1]
            a2_idx = atom_to_index[bond.atom2]
            topology_data["bonds"].append([a1_idx, a2_idx])

        # 3. Save to HDF5 as JSON string
        json_str = json.dumps(topology_data)
        dt = h5py.special_dtype(vlen=str)
        dset = self.saveFile.create_dataset("topology_json", shape=(1,), dtype=dt)
        dset[0] = json_str

    def close(self) -> None:
        self.saveFile.close()

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def describeNextReport(
        self, simulation: Simulation
    ) -> tuple[int, bool, bool, bool, bool]:
        """Get information about the next report this object will generate."""
        steps: int = self.reportInterval - simulation.currentStep % self.reportInterval
        return (
            steps,
            True,
            False,
            False,
            False,
        )  # positions, velocities, forces, energies

    def report(self, simulation: Simulation, state: State) -> None:
        """Generate a report."""
        if not self.is_paused:
            data: np.ndarray = state.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer
            )
            if self.mode == "cndb":
                self.saveFile[str(self.savestep)] = np.array(data)

                if self.PBC:
                    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                        unit.nanometer
                    )
                    self.saveFile[str(self.savestep)].attrs["box"] = box

            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            self.saveFile.flush()
            self.savestep += 1


class StabilityReporter:
    def __init__(
        self,
        filename: Union[str, Path],
        reportInterval: int = 100,
        logger: Optional[object] = None,
        kinetic_threshold: float = 5.0,
        potential_threshold: float = 1000.0,
        scale: float = 1.0,
    ):
        self.saveFile = open(filename, "w")
        self.interval: int = reportInterval
        self.kinetic_threshold: float = kinetic_threshold
        self.potential_threshold: float = potential_threshold
        self.scale: float = scale
        self.logger = logger or LogManager().get_logger(__name__)
        self.logger.info(
            f"StabilityReporter initialized with thresholds: K.E. = {self.kinetic_threshold}, P.E. = {self.potential_threshold}"
        )

    def describeNextReport(
        self, simulation: Simulation
    ) -> Tuple[int, bool, bool, bool, bool]:
        """
        Required by OpenMM Reporter interface. Specifies when the next report should occur.
        """
        steps_to_next_report = self.interval - simulation.currentStep % self.interval
        return (
            steps_to_next_report,
            False,
            False,
            False,
            True,
        )  # positions, velocities, forces, energies

    def report(self, simulation: Simulation, state: State) -> None:
        """
        Main method called at every reportInterval steps. Checks energies and reinitializes velocities if needed.
        """
        num_particles = simulation.system.getNumParticles()
        e_kinetic = (
            state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            / num_particles
        )
        e_potential = (
            state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            / num_particles
        )

        if hasattr(simulation.integrator, "getTemperature"):
            temperature = simulation.integrator.getTemperature().value_in_unit(
                unit.kelvin
            )
        else:
            temperature = 240.0

        kBT = 0.008314 * temperature
        e_kinetic_expected = 1.5 * kBT

        if (
            e_kinetic / e_kinetic_expected > self.kinetic_threshold
            or abs(e_potential) / e_kinetic_expected > self.potential_threshold
        ):

            seed = np.random.randint(100_000)
            simulation.context.setVelocitiesToTemperature(temperature, seed)
            self.saveFile.write(
                f"<<INSTABILITY | Reinitialized velocities>> Step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f}\n"
            )
            self.saveFile.flush()
            if simulation.currentStep % (self.interval * 100) == 0:
                self.logger.warning(
                    f"<<INSTABILITY | Reinitialized velocities>> at step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f}"
                )


class EnergyReporter:
    def __init__(
        self,
        report_file: Union[str, Path],
        force_field_manager,
        reportInterval: int = 1000,
        reportForceGrp: bool = False,
    ):
        self.filename: str = str(report_file)
        self.saveFile = open(report_file, "w")
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
        self.saveFile.write(
            f"{'Step':<10} {'Temperature':<12} {'RG':<10} {'K.E./particle':<15} {'P.E./particle':<15}"
        )
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
            if (
                system.getParticleMass(p1) > 0 * unit.dalton
                or system.getParticleMass(p2) > 0 * unit.dalton
            ):
                dof -= 1
        if any(
            isinstance(system.getForce(i), CMMotionRemover)
            for i in range(system.getNumForces())
        ):
            dof -= 3
        self._dof: int = dof
        self.is_initialized = True

    def describeNextReport(
        self, simulation: Simulation
    ) -> tuple[int, bool, bool, bool, bool]:
        steps_to_next_report: int = (
            self.interval - simulation.currentStep % self.interval
        )
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
            Rg: float = Analyzer.compute_RG(positions)

            integrator = simulation.context.getIntegrator()
            if hasattr(integrator, "computeSystemTemperature"):
                temperature: float = (
                    integrator.computeSystemTemperature().value_in_unit(unit.kelvin)
                )
            else:
                temperature = (
                    2
                    * state.getKineticEnergy()
                    / (self._dof * unit.MOLAR_GAS_CONSTANT_R)
                ).value_in_unit(unit.kelvin)

            ke_per_particle: float = (
                state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
                / num_particles
            )
            pe_per_particle: float = (
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                / num_particles
            )
            self.saveFile.write(
                f"{simulation.currentStep:<10} {temperature:<12.3f} {Rg:<10.4f} {ke_per_particle:<15.4f} {pe_per_particle:<15.4f}"
            )

            if self.report_force_grp:
                for i, force in enumerate(system.getForces()):
                    group: int = force.getForceGroup()
                    state_grp = context.getState(getEnergy=True, groups={group})
                    pot_energy: float = state_grp.getPotentialEnergy().value_in_unit(
                        unit.kilojoules_per_mole
                    )
                    pe_grp_per_particle: float = pot_energy / num_particles
                    self.saveFile.write(f"{pe_grp_per_particle:<20.4f}")

            self.saveFile.write("\n")
            self.saveFile.flush()
