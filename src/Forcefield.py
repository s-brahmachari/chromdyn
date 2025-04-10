from openmm import Discrete2DFunction, HarmonicBondForce, CustomNonbondedForce, CustomExternalForce, CMMotionRemover, CustomBondForce, HarmonicAngleForce
import numpy as np
import pandas as pd
from Logger import LogManager

# -------------------------------------------------------------------
# ForceField Manager: Sets up polymer forces (e.g., harmonic bonds)
# -------------------------------------------------------------------
class ForceFieldManager:
    def __init__(self, topology, system, logger=None,
                 Nonbonded_cutoff = 3.0,
                 Nonbonded_method = 'NonPeriodic',
                 ):
        self.logger = logger or LogManager().get_logger(__name__)
        self.topology = topology
        self.num_particles = topology.getNumAtoms()
        self.system = system
        self.Nonbonded_cutoff = Nonbonded_cutoff
        if Nonbonded_method=='NonPeriodic':
            self.Nonbonded_method = CustomNonbondedForce.CutoffNonPeriodic
        self.forceDict = {}
        self.force_name_map = {}
            
    def register_force(self, force_obj, name):
        """Register force with a human-readable name for later reference."""
        # if force_obj.__class__.__name__=="CustomNonbondedForce":
        #     force_obj.createExclusionsFromBonds([[int(bond[0].id),int(bond[1].id)] for bond in self.topology.bonds()], 1) 
        #     self.logger.info("Added exclusions from bonded monomers.")
        force_index = self.system.addForce(force_obj)
        self.forceDict[name] = force_obj
        self.force_name_map[force_index] = name
        self.logger.info(f"{name} force successfully added to system.")
        self.logger.info('-'*50)
    
    def add_exclusions_from_bonds(self, force):
        for bond in self.topology.bonds():
            force.addExclusion(int(bond[0].id), int(bond[1].id))
            
    def removeCOM(self, **kwargs):
        """
        Removes the center-of-mass (COM) motion from the system using CMMotionRemover.
        
        Args:
            system (System): OpenMM System object to which the force will be added.
            kwargs: Optional keyword arguments:
                - frequency (int): Frequency (in steps) at which the COM motion is removed. Default is 10.
                - forcegroup (int): Force group assignment for the CMMotionRemover. Default is 31.
        """
        
        # Extract parameters from kwargs with defaults
        frequency = int(kwargs.get('frequency', 100))       # Default frequency = 10
        forcegroup = int(kwargs.get('group', 31))    # Default force group = 31

        # Logging to provide clear feedback
        # self.logger.info('-'*50)
        self.logger.info(f"Adding CMMotionRemover with frequency: {frequency} and force group: {forcegroup}")

        # Initialize the CMMotionRemover and set force group
        cmm_remove = CMMotionRemover(frequency)
        cmm_remove.setForceGroup(forcegroup)

        # Add to system and store in force dictionary
        self.register_force(cmm_remove, "CMMRemover")

    def add_harmonic_bonds(self, **kwargs):
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
        forcegroup = int(kwargs.get('group', 0))             # Default force group
        
        # Logging for transparency
        # self.logger.info('-'*50)
        # Create HarmonicBondForce and assign to force group
        bond_force = HarmonicBondForce()
        bond_force.setForceGroup(forcegroup)
        
        # Add bonds defined in the topology
        for i, bond in enumerate(self.topology.bonds()):
            bond_force.addBond(int(bond[0].id), int(bond[1].id), bond_r, bond_k)
        
        self.logger.info(f"Adding {i+1} harmonic bonds with parameters:")
        self.logger.info(f"length: {bond_r}, spring constant (k): {bond_k}, group: {forcegroup}")
        
        # Add the force to the system and record in force dictionary
        self.register_force( bond_force,"HarmonicBonds")

    def add_harmonic_angles(self, **kwargs):
        """
        Adds harmonic angle forces between triplets of bonded particles.
        Angle parameters are passed via kwargs and stored in self for later access.

        Args:
            system (System): OpenMM System object to which the force will be added.
            kwargs: Optional keyword arguments:
                - theta0 (float): Equilibrium bond angle in degrees. Default is 120.0.
                - k (float): Angle force constant (kJ/mol/rad²). Default is 10.0.
                - forcegroup (int): Force group assignment. Default is 0.
        """
        
        # Extract parameters from kwargs with defaults
        theta0 = float(kwargs.get('theta0', 180.0))    # Default equilibrium angle in degrees
        k_angle = float(kwargs.get('k_angle', 2.0))               # Default angle force constant (kJ/mol/rad²)
        forcegroup = int(kwargs.get('group', 4))  # Default force group
        
        # Convert theta0 to radians (OpenMM expects radians)
        theta0_rad = theta0 * (np.pi / 180)
        
        # Create the HarmonicAngleForce object
        angle_force = HarmonicAngleForce()
        angle_force.setForceGroup(forcegroup)
        
        # Step 1: Build a bonded neighbor list
        bonded_neighbors = {atom.index: [] for atom in self.topology.atoms()}
        
        for bond in self.topology.bonds():
            a, b = int(bond[0].id), int(bond[1].id)
            bonded_neighbors[a].append(b)
            bonded_neighbors[b].append(a)
        
        # Step 2: Find valid angle triplets (i - j - k)
        num_angles = 0
        for j, neighbors in bonded_neighbors.items():
            if len(neighbors) < 2:
                continue  # Need at least 2 bonded neighbors to form an angle
            if len(neighbors)>3:
                self.logger.error("More than two bonded neighbors. Not clear which monomers should be included in angle restraint. It has to be handled properly.")
                raise ValueError
            
            for i in neighbors:
                for k in neighbors:
                    if i >= k:
                        continue  # Avoid duplicate angles
                    
                    # Add angle force (i - j - k)
                    angle_force.addAngle(i, j, k, theta0_rad, k_angle)
                    num_angles += 1
        
        # Log angle force addition
        self.logger.info(f"Adding {num_angles} harmonic angles with parameters:")
        self.logger.info(f"θ₀: {theta0}° ({theta0_rad:.4f} rad), k: {k_angle}, force group: {forcegroup}")
        
        # Add the force to the system and store in the force dictionary
        self.register_force( angle_force, "HarmonicAngles")
    
    def add_fene_bonds(self, **kwargs):
        """
        Adds FENE (Finite Extensible Nonlinear Elastic) bonds to an OpenMM system.

        Args:
            system (System): OpenMM System object to which the FENE bonds will be added.
            bond_list (list of tuple): List of (atom1, atom2) index pairs to bond.
            kwargs:
                - k (float): Spring constant (default 30.0).
                - R0 (float): Maximum extension (default 1.5).
                - forcegroup (int): Force group for the FENE bonds (default 2).
        """
        k = float(kwargs.get('k', 30.0))             # FENE spring constant
        R0 = float(kwargs.get('R0', 1.5))            # Maximum extension
        forcegroup = int(kwargs.get('group', 0))

        self.logger.info(f"Adding FENE bonds with k={k}, R0={R0}, force group={forcegroup}")

        # FENE bond potential (no units here; handled by OpenMM engine)
        fene_force = CustomBondForce(
            "-0.5 * k_fene * R0_fene^2 * log(1 - (r/R0_fene)^2)"
        )
        fene_force.addGlobalParameter("k_fene", k)
        fene_force.addGlobalParameter("R0_fene", R0)

        # Add all FENE bonds
        for i, bond in enumerate(self.topology.bonds()):
            fene_force.addBond(int(bond[0].id), int(bond[1].id), ())
        
        fene_force.setForceGroup(forcegroup)
        self.register_force( fene_force, "FENEBonds")
    
    def add_harmonic_trap(self,  **kwargs):
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
        forcegroup = int(kwargs.get('group', 1))                   # Default forcegroup

        # Log the selected parameters
        # self.logger.info('-'*50)
        self.logger.info(f"Adding Harmonic Trap with kr={kr}, center={center}, force group={forcegroup}")

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
        self.register_force( restraintForce, "HarmonicTrap")

    def add_self_avoidance(self, **kwargs):
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
        forcegroup = kwargs.get('group', 2)

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
        num_particles = getattr(self, 'num_particles', self.system.getNumParticles())
        for _ in range(num_particles):
            avoidance_force.addParticle(())
        # self.logger.info('-'*50)
        self.logger.info(f"Adding Self-avoidance force with parameters:")
        self.logger.info(f"Ecut={Ecut}, k_rep={kSA}, r_rep={rSA}, cutoff={self.Nonbonded_cutoff}, group={forcegroup}")
        # self.add_exceptions_from_bonds(avoidance_force)
        # avoidance_force.createExclusionsFromBonds([[int(bond[0].id),int(bond[1].id)] for bond in self.topology.bonds()], 1) 
        self.register_force(avoidance_force, "SelfAvoidance")

    def add_LJ_repulsion(self, **kwargs):
        """
        Adds hard-core self-avoidance with flexible parameters passed via kwargs.
        
        Args:
            system (System): OpenMM system.
            verbose (bool): Print setup info.
            **kwargs: Flexible parameters like Ecut, kSA, rSA, forcegroup.
                    Example: {'Ecut': 4.0, 'kSA': 5.0, 'rSA': 1.0, 'forcegroup': 2}
        """

        # Extract parameters with defaults using kwargs.get
        sigma = kwargs.get('sigma', 1.0)
        forcegroup = kwargs.get('group', 6)

        # Define force
        repul_energy = "(sigma / r) ^ 12"
        hard_repel_force = CustomNonbondedForce(repul_energy)
        hard_repel_force.setForceGroup(forcegroup)
        hard_repel_force.setCutoffDistance(self.Nonbonded_cutoff)
        hard_repel_force.setNonbondedMethod(self.Nonbonded_method)

        # Add global parameters
        hard_repel_force.addGlobalParameter('sigma', sigma)
        
        # Add particles
        num_particles = getattr(self, 'num_particles', self.system.getNumParticles())
        for _ in range(num_particles):
            hard_repel_force.addParticle(())
        # self.logger.info('-'*50)
        self.logger.info(f"Adding Hard-core repulsion force with parameters:")
        self.logger.info(f"sigma={sigma}, cutoff={self.Nonbonded_cutoff}, group={forcegroup}")
        # self.add_exceptions_from_bonds(avoidance_force)
        # avoidance_force.createExclusionsFromBonds([[int(bond[0].id),int(bond[1].id)] for bond in self.topology.bonds()], 1) 
        self.register_force(hard_repel_force, "HardCoreLJ")

    def add_type_to_type_interaction(self, interaction_matrix, type_labels, verbose=True, **kwargs):
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
        self.logger.info('-'*50)
        self.logger.info(f"Types detected in polymer: {used_types}")

        # ---- Check type consistency with interaction matrix ---- #
        missing_types = [t for t in used_types if t not in type_labels]
        if missing_types:
            raise ValueError(f"Types found in topology but missing in interaction matrix: {missing_types}")

        unused_types = [t for t in type_labels if t not in used_types]
        if unused_types and verbose:
            self.logger.warning(f"Types defined in interaction matrix but not used in topology: {unused_types}")

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

        self.logger.info(f"Adding Type-to-Type interaction (force group {force_group}, {num_types} types).")
        self.logger.info(f"Parameters -> mu: {self.mu}, rc: {self.rc},cutoff: {self.Nonbonded_cutoff}")
        # type_force.createExclusionsFromBonds([[int(bond[0].id),int(bond[1].id)] for bond in self.topology.bonds()], 1) 
        self.register_force(type_force,"TypeToType")

    def _get_type_interaction_matrix(self, file_path):
        """
        Loads type-to-type interaction matrix from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing interaction matrix.

        Returns:
            type_labels (list of str): List of type labels.
            interaction_matrix (np.ndarray): Interaction matrix (2D array).
        """

        df = pd.read_csv(file_path)  # Assume first column and row are type labels
        type_labels = list(df.columns)  # Extract type names from header
        interaction_matrix = df.values.astype(float)  # Convert DataFrame to NumPy array (float)

        if self.logger:
            self.logger.info(f"Loaded interaction matrix from {file_path}")
            self.logger.info(f"Interaction matrix shape: {interaction_matrix.shape}, Type labels: {type_labels}")
            # self.logger.info(f"")

        return type_labels, interaction_matrix
    
    def add_default_forces(self, mode='default', **kwargs): 
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
        rc = float(kwargs.get('rc', 1.5))
        
        """Configures system with appropriate force fields based on mode."""
        # self.logger.info("-"*60)
        self.logger.info(f"Setting up forces with mode='{mode}'")
            
        if cmm_remove: self.removeCOM(self.system)

        if mode == 'default':
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            type_labels, interaction_matrix = self._get_type_interaction_matrix('./type_interaction_table.csv')
            self.add_type_to_type_interaction(interaction_matrix, type_labels)
        
        elif mode == 'debug':
            self.add_harmonic_trap(kr=k_res)
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_self_avoidance(Ecut=E_rep, k=k_rep, r=r_rep)
            type_labels, interaction_matrix = self._get_type_interaction_matrix('./type_interaction_table.csv')
            self.add_type_to_type_interaction(interaction_matrix, type_labels)
            
        elif mode == 'harmtrap_gauss':
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_harmonic_trap(kr=k_res)

        elif mode == "harmtrap_saw":
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_harmonic_trap(kr=k_res)
            self.add_self_avoidance(Ecut=E_rep, k=k_rep, r=r_rep)
        
        elif mode == "saw":
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_self_avoidance(Ecut=E_rep, k=k_rep, r=r_rep)
        
        elif mode == "saw_LJ":
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_LJ_repulsion(sigma=r_rep)
        
        elif mode == "saw_LJ_fene":
            self.add_fene_bonds(k=k_bond)
            self.add_LJ_repulsion(sigma=r_rep)
            
        elif mode=="saw_stiff_backbone":
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_self_avoidance(Ecut=E_rep, k=k_rep, r=r_rep)
            self.add_harmonic_angles(theta0=theta0, k_angle=k_angle)
            
        elif mode=="saw_stiff_backbone_bad_solvent":
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_self_avoidance(Ecut=E_rep, k=k_rep, r=r_rep)
            self.add_harmonic_angles(theta0=theta0, k_angle=k_angle)
            type_labels = ["A", "B"]
            interaction_matrix = [[chi, 0.0], [0.0, 0.0]]
            self.add_type_to_type_interaction(interaction_matrix, type_labels, rc=rc)
            
        elif mode == "gauss":
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
        
        elif mode == "fene":
            self.add_fene_bonds(k=k_bond)
            
        elif mode == "saw_bad_solvent":
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_self_avoidance(Ecut=E_rep, k=k_rep, r=r_rep)
            type_labels = ["A", "B"]
            interaction_matrix = [[chi, 0.0], [0.0, 0.0]]
            self.add_type_to_type_interaction(interaction_matrix, type_labels, rc=rc)
        
        elif mode == "saw_LJ_bad_solvent":
            self.add_harmonic_bonds(k=k_bond, r0=r_bond)
            self.add_LJ_repulsion(sigma=r_rep)
            type_labels = ["A", "B"]
            interaction_matrix = [[chi, 0.0], [0.0, 0.0]]
            self.add_type_to_type_interaction(interaction_matrix, type_labels, rc=rc)
            
        self.logger.info("Force set up complete!")
        self.logger.info("-"*60)
