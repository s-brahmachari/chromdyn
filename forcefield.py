from openmm import Discrete2DFunction, HarmonicBondForce, CustomNonbondedForce, CustomExternalForce, CMMotionRemover
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# ForceField Manager: Sets up polymer forces (e.g., harmonic bonds)
# -------------------------------------------------------------------
class ForceFieldManager:
    def __init__(self, topology, 
                 Nonbonded_cutoff = 3.0,
                 Nonbonded_method = 'NonPeriodic',
                 ):
        
        self.topology = topology
        self.num_particles = topology.getNumAtoms()
        
        self.Nonbonded_cutoff = Nonbonded_cutoff
        if Nonbonded_method=='NonPeriodic':
            self.Nonbonded_method = CustomNonbondedForce.CutoffNonPeriodic
        self.forceDict = {}
        self.force_name_map = {}
        
    def register_force(self, system, force_obj, name):
        """Register force with a human-readable name for later reference."""
        force_index = system.addForce(force_obj)
        self.forceDict[name] = force_obj
        self.force_name_map[force_index] = name
    
    def removeCOM(self, system, **kwargs):
        """
        Removes the center-of-mass (COM) motion from the system using CMMotionRemover.
        
        Args:
            system (System): OpenMM System object to which the force will be added.
            kwargs: Optional keyword arguments:
                - frequency (int): Frequency (in steps) at which the COM motion is removed. Default is 10.
                - forcegroup (int): Force group assignment for the CMMotionRemover. Default is 31.
        """
        
        # Extract parameters from kwargs with defaults
        frequency = int(kwargs.get('frequency', 10))       # Default frequency = 10
        forcegroup = int(kwargs.get('forcegroup', 31))    # Default force group = 31

        # Logging to provide clear feedback
        print(f"[INFO] Adding CMMotionRemover with frequency: {frequency} and force group: {forcegroup}")

        # Initialize the CMMotionRemover and set force group
        cmm_remove = CMMotionRemover(frequency)
        cmm_remove.setForceGroup(forcegroup)

        # Add to system and store in force dictionary
        self.register_force(system, cmm_remove, "CMMRemover")
        

        print("[INFO] Center of mass motion remover successfully added.")
        return cmm_remove   

    def add_harmonic_bonds(self, system, **kwargs):
        """
        Adds harmonic bonds between consecutive particles (or defined bonds in topology).
        Bond parameters can be passed via kwargs and are stored in self for later access.
        
        Args:
            system (System): OpenMM System object to which the force will be added.
            kwargs: Optional keyword arguments:
                - bond_length (float): Equilibrium bond length. Default is 1.0.
                - bond_k (float): Bond spring constant. Default is 1.0.
                - forcegroup (int): Force group assignment. Default is 0.
        """
        
        # Extract parameters from kwargs with defaults
        bond_r = float(kwargs.get('r0', 1.0))  # Default bond length
        bond_k = float(kwargs.get('k', 10.0))            # Default bond spring constant
        forcegroup = int(kwargs.get('forcegroup', 0))             # Default force group
        
        # Logging for transparency
        print(f"[INFO] Adding harmonic bonds with parameters:")
        print(f"       Bond length: {bond_r}")
        print(f"       Bond spring constant (k): {bond_k}")
        print(f"       Force group: {forcegroup}")

        # Create HarmonicBondForce and assign to force group
        bond_force = HarmonicBondForce()
        bond_force.setForceGroup(forcegroup)
        
        # Add bonds defined in the topology
        for bond in self.topology.bonds():
            bond_force.addBond(int(bond[0].id), int(bond[1].id), bond_r, bond_k)
        
        # Add the force to the system and record in force dictionary
        self.register_force(system, bond_force,"HarmonicBonds")
        print(f"[INFO] HarmonicBondForce successfully added with {bond_force.getNumBonds()} bonds.")
        return bond_force

    def add_harmonic_trap(self, system, **kwargs):
        """
        Adds a harmonic trap (restraint) to the system to confine particles within a spherical region.
        
        Args:
            system (System): OpenMM System object to which the force will be added.
            kwargs: Optional keyword arguments:
                - kr (float): Spring constant for the trap (default 0.1).
                - center (tuple): Center of the trap as (x, y, z) coordinates (default (0.0, 0.0, 0.0)).
                - forcegroup (int): Force group to which the trap will belong (default 1).
        """

        # Extract parameters with defaults
        kr = float(kwargs.get('kr', 0.1))                           # Store kr in self
        center = kwargs.get('r0', (0.0, 0.0, 0.0))                   # Default center
        forcegroup = int(kwargs.get('forcegroup', 1))                   # Default forcegroup

        # Log the selected parameters
        print(f"[INFO] Adding Harmonic Trap with kr={kr}, center={center}, force group={forcegroup}")

        # Define the external force expression (harmonic trap potential)
        restraintForce = CustomExternalForce(
            "0.5 * k_harm_trap * r * r; "
            "r = sqrt( (x - x0_trap)^2 + (y - y0_trap)^2 + (z - z0_trap)^2 )"
        )

        # Add global parameters for strength and center
        restraintForce.addGlobalParameter('k_harm_trap', kr)
        restraintForce.addGlobalParameter('x0_trap', center[0])
        restraintForce.addGlobalParameter('y0_trap', center[1])
        restraintForce.addGlobalParameter('z0_trap', center[2])

        # Add all particles to this external force
        for i in range(self.topology.getNumAtoms()):
            restraintForce.addParticle(i, ())

        # Assign the force to the appropriate force group
        restraintForce.setForceGroup(forcegroup)

        # Add force to system and store it for reference
        self.register_force(system, restraintForce, "HarmonicTrap")
        
        print("[INFO] Harmonic trap successfully added to the system.")
        return restraintForce
    
    def add_self_avoidance(self, system, verbose=True, **kwargs):
        """
        Adds soft-core self-avoidance with flexible parameters passed via kwargs.
        
        Args:
            system (System): OpenMM system.
            verbose (bool): Print setup info.
            **kwargs: Flexible parameters like Ecut, kSA, rSA, forcegroup.
                    Example: {'Ecut': 4.0, 'kSA': 5.0, 'rSA': 1.0, 'forcegroup': 2}
        """

        # Extract parameters with defaults using kwargs.get
        Ecut = kwargs.get('Ecut', 4.0)
        kSA = kwargs.get('k', 5.0)
        rSA = kwargs.get('r', 1.0)
        forcegroup = kwargs.get('forcegroup', 2)

        # Define force
        repul_energy = "0.5 * Ecut * (1.0 + tanh((k_rep * (r_rep - r))))"
        avoidance_force = CustomNonbondedForce(repul_energy)
        avoidance_force.setForceGroup(forcegroup)
        avoidance_force.setCutoffDistance(self.Nonbonded_cutoff)
        avoidance_force.setNonbondedMethod(self.Nonbonded_method)

        # Add global parameters
        avoidance_force.addGlobalParameter('Ecut', Ecut)
        avoidance_force.addGlobalParameter('r_rep', rSA)
        avoidance_force.addGlobalParameter('k_rep', kSA)

        # Add particles
        num_particles = getattr(self, 'num_particles', system.getNumParticles())
        for _ in range(num_particles):
            avoidance_force.addParticle(())

        # Add force
        self.register_force(system, avoidance_force, "SelfAvoidance")
        
        if verbose:
            print(f"[INFO] Self-avoidance force added with parameters: "
                f"Ecut={Ecut}, k_rep={kSA}, r_rep={rSA}, "
                f"cutoff={self.Nonbonded_cutoff}, group={forcegroup}")
        
        return avoidance_force
    
    def add_type_to_type_interaction(self, system, interaction_matrix, type_labels, verbose=True, **kwargs):
        """
        Adds type-to-type nonbonded interactions with flexible parameter passing via kwargs.

        Args:
            system (System): OpenMM system object.
            interaction_matrix (2D array): Interaction matrix aligned with type_labels.
            type_labels (list of str): List of type labels corresponding to interaction_matrix.
            verbose (bool): Whether to print setup info.
            **kwargs: Optional parameters (mu, rc, force_group).

        Returns:
            CustomNonbondedForce: The constructed OpenMM force object.
        """

        # ---- Extract parameters directly from kwargs with defaults ---- #
        self.mu = kwargs.get('mu', 5.0)                   # Steepness parameter for tanh
        self.rc = kwargs.get('rc', 1.5)                  # Reference distance for tanh
        force_group = kwargs.get('force_group', 5)       # Force group index

        # ---- Extract type list from topology ---- #
        type_list = [atom.element for atom in self.topology.atoms()]
        used_types = sorted(set(type_list))  # Unique types used in polymer

        if verbose:
            print(f"[INFO] Types detected in polymer: {used_types}")

        # ---- Check type consistency with interaction matrix ---- #
        missing_types = [t for t in used_types if t not in type_labels]
        if missing_types:
            raise ValueError(f"[ERROR] Types found in topology but missing in interaction matrix: {missing_types}")

        unused_types = [t for t in type_labels if t not in used_types]
        if unused_types and verbose:
            print(f"[WARNING] Types defined in interaction matrix but not used in topology: {unused_types}")

        # ---- Map types and subset interaction matrix ---- #
        type_to_idx = {label: idx for idx, label in enumerate(type_labels)}
        subset_indices = [type_to_idx[t] for t in used_types]
        reduced_matrix = np.array(interaction_matrix)[np.ix_(subset_indices, subset_indices)]

        num_types = len(used_types)
        flat_interaction = reduced_matrix.flatten().tolist()

        # ---- Define force energy expression ---- #
        energy_expr = "interaction_map(t1, t2) * 0.5 * (1 + tanh(mu * (rc - r)));"

        # ---- Create and configure CustomNonbondedForce ---- #
        type_force = CustomNonbondedForce(energy_expr)
        type_force.setForceGroup(force_group)
        type_force.setCutoffDistance(self.Nonbonded_cutoff)
        type_force.setNonbondedMethod(self.Nonbonded_method)
        type_force.addGlobalParameter('mu', self.mu)
        type_force.addGlobalParameter('rc', self.rc)

        # ---- Add tabulated function from interaction matrix ---- #
        interaction_func = Discrete2DFunction(num_types, num_types, flat_interaction)
        type_force.addTabulatedFunction("interaction_map", interaction_func)

        # ---- Assign per-particle type indices ---- #
        type_force.addPerParticleParameter("t")
        type_idx_map = {label: idx for idx, label in enumerate(used_types)}  # Mapping to reduced matrix
        for type_label in type_list:
            type_force.addParticle([float(type_idx_map[type_label])])  # Map original type to reduced index

        # ---- Add force to system ---- #
        self.register_force(system, type_force,"TypeToType")
        # ---- Logging ---- #
        if verbose:
            print(f"[INFO] Type-to-Type interaction added (force group {force_group}, {num_types} types).")
            print(f"[INFO] Parameters -> mu: {self.mu}, rc: {self.rc},cutoff: {self.Nonbonded_cutoff}")

        return type_force

