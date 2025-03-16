import h5py
import os
import openmm.unit as unit
import numpy as np
from topology import Topology

class StabilityReporter:
    """
    A custom reporter that checks system stability during simulation runs.
    If per-particle kinetic or potential energies exceed thresholds, velocities are reinitialized.
    """
    def __init__(self, simulation, reportInterval=1000,logger=None, 
                 kinetic_threshold=5.0, potential_threshold=1000.0, scale=1.0, output_file=None):
        """
        Args:
            simulation (Simulation): OpenMM Simulation object.
            reportInterval (int): Number of steps between checks.
            kinetic_threshold (float): Max kinetic energy per particle [kJ/mol].
            potential_threshold (float): Max absolute potential energy per particle [kJ/mol].
            scale (float): Scaling factor for velocity reinitialization.
            output_file (str): Optional file to log stability events.
        """
        self.simulation = simulation
        self.interval = reportInterval
        self.kinetic_threshold = kinetic_threshold
        self.potential_threshold = potential_threshold
        self.scale = scale
        self.output_file = output_file
        self.num_particles = self.simulation.topology.getNumAtoms()

    def describeNextReport(self, simulation):
        """
        Required by OpenMM Reporter interface. Specifies when the next report should occur.
        """
        return (self.interval, True, True, False, False)  # Need energies and positions

    def report(self, simulation, state):
        """
        Main method called at every reportInterval steps. Checks energies and reinitializes velocities if needed.
        """
        # Retrieve energies
        e_kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / self.num_particles
        e_potential = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / self.num_particles

        # Prepare message
        msg = (f"[Stability Check] Step {simulation.currentStep}: "
               f"Kinetic (per particle) = {e_kinetic:.2f} kJ/mol, "
               f"Potential (per particle) = {e_potential:.2f} kJ/mol")

        if e_kinetic > self.kinetic_threshold or abs(e_potential) > self.potential_threshold:
            # If thresholds exceeded, reinitialize velocities
            self.reinitialize_velocities()
            msg += " --> [WARNING] Reinitialized velocities due to instability."
        else:
            msg += " --> [OK] Stable."

        # Print to console
        print(msg)

        # Optionally log to file
        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(msg + "\n")

    def reinitialize_velocities(self):
        """Reinitialize random velocities with scaling."""
        temperature = self.simulation.integrator.getTemperature()
        sigma = np.sqrt(temperature / self.simulation.system.getParticleMass(0))  # Assuming uniform mass
        velocities = np.random.normal(0, sigma, size=(self.num_particles, 3)) * self.scale
        velocities_quantity = unit.Quantity(velocities, unit.nanometers / unit.picoseconds)
        self.simulation.context.setVelocities(velocities_quantity)
        print("[INFO] Velocities reinitialized.")

class StateAnalyzer:
    def __init__(self, simulation,output_dir="output"):
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

    def get_energies(self,):
        state = self.simulation.context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        kin_energy = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        return {'Ep':pot_energy/self.num_particles, 'Ek': kin_energy/self.num_particles}
        
    def compute_RG(self):
        """
        Computes the radius of gyration (Rg) of the polymer configuration.

        Args:
            save_to_file (bool): Whether to save the Rg value to a file.
            filename (str): File name to save the Rg value if save_to_file=True.

        Returns:
            float: Radius of gyration .
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

        return Rg

class HiCManager:
    def __init__(self, hicmap):
        """
        HiCManager to handle Hi-C contact maps from file, array, or generate random.
        
        Args:
            hicmap (str, np.ndarray, int): Path to .txt file, 2D NumPy array, or integer to generate.
            random_max_value (int): Max value for random Hi-C generation (default: 10).
        """
        self.hic_map = None  # Final Hi-C map

        # Process input via internal loading function
        self._load_hic_map(hicmap)

    def _load_hic_map(self, hicmap):
        """
        Attempts to load or generate Hi-C map based on the type of input.
        
        Args:
            hicmap (str, np.ndarray, int): Hi-C data source.
        """

        # First, try to interpret as a file path if it's a string
        if isinstance(hicmap, str):
            try:
                if os.path.isfile(hicmap):
                    if hicmap.endswith(".txt"):
                        self.hic_map = np.loadtxt(hicmap)
                        print(f"[INFO] Hi-C map loaded from file '{hicmap}' with shape {self.hic_map.shape}.")
                        return  # Success, exit function
                    else:
                        print(f"[WARNING] File '{hicmap}' found but unsupported format. Expecting .txt.")
                else:
                    print(f"[WARNING] Path '{hicmap}' is not a valid file.")
            except Exception as e:
                print(f"[ERROR] Failed to load Hi-C map from file '{hicmap}': {e}")

        # If not a valid file or failed, check if it's a NumPy array
        if isinstance(hicmap, np.ndarray):
            try:
                if hicmap.ndim != 2:
                    raise ValueError(f"Provided NumPy array must be 2D. Got shape: {hicmap.shape}")
                self.hic_map = hicmap
                print(f"[INFO] Hi-C map loaded from NumPy array with shape {hicmap.shape}.")
                return  # Success, exit function
            except Exception as e:
                print(f"[ERROR] Failed to use provided NumPy array: {e}")

        # If not an array, check if it's an integer for random generation
        if isinstance(hicmap, int):
            try:
                if hicmap <= 0:
                    raise ValueError("Integer for Hi-C map generation must be positive.")
                random_map = np.random.random(0, 1, size=(hicmap, hicmap))
                self.hic_map = (random_map + random_map.T) // 2  # Make symmetric
                np.fill_diagonal(self.hic_map, 0)  # Optional: zero diagonal
                print(f"[INFO] Random symmetric Hi-C map generated (size: {hicmap}x{hicmap}, max value: {self.random_max_value}).")
                return  # Success, exit function
            except Exception as e:
                print(f"[ERROR] Failed to generate random Hi-C map: {e}")

        # If none of the above succeeded, raise a clear error
        raise TypeError(f"[FATAL ERROR] Unable to interpret input '{hicmap}'. Must be path to '.txt' file, 2D NumPy array, or positive integer.")

    def get_hic_map(self):
        """Returns the loaded Hi-C map."""
        return self.hic_map

    def get_shape(self):
        """Returns shape of the Hi-C map."""
        return self.hic_map.shape if self.hic_map is not None else None
    
    def gen_top(self,):
        topology = Topology()
        return 