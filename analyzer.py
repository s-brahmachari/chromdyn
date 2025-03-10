import h5py
import os
import openmm.unit as unit
import numpy as np

class StateAnalyzer:
    def __init__(self, simulation, output_dir="output"):
        self.simulation = simulation
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_particles = simulation.topology.getNumAtoms()

    def _save_array(self, data, filename, dataset_name="data"):
        """Utility to save numpy arrays to HDF5."""
        with h5py.File(os.path.join(self.output_dir, filename), "w") as hf:
            hf.create_dataset(dataset_name, data=data)
        print(f"[INFO] {dataset_name} saved to {filename}")

    def save_positions(self, filename="positions.h5"):
        state = self.simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        self._save_array(pos, filename, "positions")

    def save_velocities(self, filename="velocities.h5"):
        state = self.simulation.context.getState(getVelocities=True)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometers/unit.picoseconds)
        self._save_array(vel, filename, "velocities")

    def save_forces(self, filename="forces.h5"):
        state = self.simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        self._save_array(forces, filename, "forces")

    def save_energies(self, filename="energies.txt"):
        state = self.simulation.context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        kin_energy = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        with open(os.path.join(self.output_dir, filename), "a") as f:
            f.write(f"Potential: {pot_energy:.3f} kJ/mol, Kinetic: {kin_energy:.3f} kJ/mol\n")
        print(f"[INFO] Energies logged to {filename}")

    def print_force_info(self):
        system = self.simulation.system
        context = self.simulation.context
        print("{:<6} {:<30} {:<8} {:<15} {:<12} {:<25}".format(
            "Index", "Force Class", "Group", "Num Particles", "Num Bonds", "Energy (kJ/mol)"
        ))
        print("-" * 110)
        for i, force in enumerate(system.getForces()):
            group = force.getForceGroup()
            force_name = force.__class__.__name__
            state = context.getState(getEnergy=True, groups={group})
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            num_particles = getattr(force, 'getNumParticles', lambda: 'N/A')()
            num_bonds = getattr(force, 'getNumBonds', lambda: 'N/A')()
            print(f"{i:<6} {force_name:<30} {group:<8} {num_particles:<15} {num_bonds:<12} {energy:<25.3f}")
            
    def compute_RG(self):
        """
        Computes the radius of gyration (Rg) of the polymer configuration.

        Args:
            save_to_file (bool): Whether to save the Rg value to a file.
            filename (str): File name to save the Rg value if save_to_file=True.

        Returns:
            float: Radius of gyration in nanometers.
        """

        # Get current positions
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)  # Shape (N, 3)

        # Compute center of mass (mean position)
        center_of_mass = np.mean(positions, axis=0)

        # Compute squared distances from center of mass
        squared_distances = np.sum((positions - center_of_mass) ** 2, axis=1)

        # Compute radius of gyration (sqrt of mean squared distance)
        Rg = np.sqrt(np.mean(squared_distances))

        # Prepare output string
        output_str = f"[INFO] Radius of Gyration (Rg): {Rg:.4f} nm for {self.num_particles} particles."

        # Print to console
        print(output_str)

        return Rg