from openmm import Discrete2DFunction, HarmonicBondForce, CustomNonbondedForce, CustomExternalForce, CMMotionRemover
import numpy as np
# -------------------------------------------------------------------
# ForceField Manager: Sets up polymer forces (e.g., harmonic bonds)
# -------------------------------------------------------------------
class ForceFieldManager:
    def __init__(self, topology, 
                 bond_length=1.0, 
                 bond_k=1.0,
                 epsilon=1.0):
        self.topology = topology
        self.num_particles = topology.getNumAtoms()
        self.bond_length = bond_length
        self.bond_k = bond_k
        self.Epsilon = epsilon
        self.forceDict = {}
    
    def removeCOM(self, system, frequency=10, forcegroup=31):
        cmm_remove = CMMotionRemover(frequency)
        cmm_remove.setForceGroup(forcegroup)
        system.addForce(cmm_remove)
        self.forceDict["CMMRemover"] = cmm_remove
        return cmm_remove

    def add_harmonic_bonds(self, system, forcegroup=0):
        """
        Adds harmonic bonds between consecutive particles.
        Assigned to force group 0.
        """
        bond_force = HarmonicBondForce()
        bond_force.setForceGroup(forcegroup)
        for bond in self.topology.bonds():
        # for i in range(self.num_particles - 1):
            bond_force.addBond(int(bond[0].id), int(bond[1].id), self.bond_length, self.bond_k)
        system.addForce(bond_force)
        self.forceDict["HarmonicBonds"] = bond_force
        return bond_force
    
    def add_harmonic_trap(self, system, 
                                        kr=0.1, center=(0.0,0.0,0.0), forcegroup=1):
        
        R"""
        Sets a Flat-Bottom Harmonic potential to collapse the chromosome chain inside the nucleus wall. The potential is defined as: :math:`step(r-r0) * (kr/2)*(r-r0)^2`.

        Args:

            kr (float, required):
                Spring constant. (Default value = 5e-3). 
            n_rad (float, required):
                Nucleus wall radius in units of :math:`\sigma`. (Default value = 10.0).  
        """

        restraintForce = CustomExternalForce("0.5 * k_harm_trap * r * r; r=sqrt( (x-x0_trap)^2 + (y-y0_trap)^2 + (z-z0_trap)^2)")
        restraintForce.addGlobalParameter('k_harm_trap', kr)
        restraintForce.addGlobalParameter('x0_trap', center[0])
        restraintForce.addGlobalParameter('y0_trap', center[1])
        restraintForce.addGlobalParameter('z0_trap', center[2])
        
        for i in range(self.topology.getNumAtoms()):
            restraintForce.addParticle(i, ())
        
        restraintForce.setForceGroup(forcegroup)
        system.addForce(restraintForce)
        self.forceDict["HarmonicTrap"] = restraintForce
        return restraintForce
    
    def add_self_avoidance(self, system, 
                           Ecut=4.0, k_rep=5.0, r0=1.0, forcegroup=2):
        """
        Adds soft-core self-avoidance between all non-bonded particles.
        Assigned to force group 2.
        
        Args:
            Ecut (float): Energy associated with full overlap.
            k_rep (float): Steepness of the repulsive potential.
            r0 (float): Distance where the potential is half as strong.
        """
        scaled_Ecut = Ecut * self.Epsilon
        repul_energy = "0.5 * Ecut * (1.0 + tanh((k_rep * (r_rep - r))))"
        avoidance_force = CustomNonbondedForce(repul_energy)
        avoidance_force.setForceGroup(forcegroup)
        avoidance_force.addGlobalParameter('Ecut', scaled_Ecut)
        avoidance_force.addGlobalParameter('r_rep', r0)
        avoidance_force.addGlobalParameter('k_rep', k_rep)
        avoidance_force.setCutoffDistance(3.0)
        for _ in range(self.num_particles):
            avoidance_force.addParticle(())
        system.addForce(avoidance_force)
        self.forceDict["SelfAvoidance"] = avoidance_force
        return avoidance_force

    def add_cylindrical_confinement(self, system, 
                                conf_radius=10.0, 
                                conf_height=100.0, 
                                conf_k=1.0, forcegroup=3):
        """
        Adds a cylindrical confinement force to all particles using a CustomExternalForce.
        
        The force penalizes particles when:
        - Their radial distance (r_xy = sqrt(x^2+y^2)) exceeds conf_radius, or 
        - The absolute value of their z-coordinate exceeds half of conf_height.
        
        The energy expression is defined as:
        
            U = 0.5 * conf_k * [ step(r_xy - r_cyn) * (r_xy - r_cyn)^2 +
                                step(|z| - z_thres) * (|z| - z_thres)^2 ]
                                
        where:
        - r_cyn is the radial cutoff (set to conf_radius),
        - z_thres is the vertical threshold (conf_height/2).
        
        This force is assigned to force group 1.
        
        Parameters:
        system (openmm.System): The system to which the force will be added.
        conf_radius (openmm.unit.Quantity): Radial confinement threshold (default 10 nm).
        conf_height (openmm.unit.Quantity): Total cylinder height (default 100 nm; vertical threshold is half).
        conf_k (openmm.unit.Quantity): Force constant for confinement (default 0.0001 kJ/mol/nm²).
        
        Returns:
        The CustomExternalForce object for cylindrical confinement.
        """
        # Compute the vertical threshold.
        z_threshold = conf_height / 2.0

        # Define the energy expression.
        energy_expr = (
            "step(r_xy - r_cyn) * 0.5 * k_cyn * (r_xy - r_cyn)^2 + "
            "step(abs(z) - z_thres) * 0.5 * k_cyn * (abs(z) - z_thres)^2; "
            "r_xy = sqrt(x*x + y*y)"
        )
        
        # Create the CustomExternalForce and assign it to force group 1.
        conf_force = CustomExternalForce(energy_expr)
        conf_force.setForceGroup(forcegroup)
        
        # Add global parameters: radial cutoff, vertical threshold, and force constant.
        conf_force.addGlobalParameter("r_cyn", conf_radius)
        conf_force.addGlobalParameter("k_cyn", conf_k)
        conf_force.addGlobalParameter("z_thres", z_threshold)
        
        # Add each particle to the force.
        for i in range(self.topology.getNumAtoms()):
            conf_force.addParticle(i, [])
        
        system.addForce(conf_force)
        
        self.forceDict["CylindricalConfinement"] = conf_force
        return conf_force
    
    def add_spherical_confinement(self, system, R_conf=50.0, k=0.001, forcegroup=4):
        """
        Adds a spherical confinement force using a CustomExternalForce.
        
        The energy expression penalizes a particle when its radial distance (r) 
        exceeds R_conf:
        
            U = 0.5 * k * (r - R_conf)^2   for r > R_conf,
            U = 0                         for r <= R_conf,
        
        where r = sqrt(x*x + y*y + z*z).
        
        Assigned to force group 3.
        
        Parameters:
        system (openmm.System): The system to which the force will be added.
        R_conf (openmm.unit.Quantity): Confinement radius (default 50 nm).
        k (openmm.unit.Quantity): Force constant (default 0.001 kJ/mol/nm²).
        
        Returns:
        The CustomExternalForce object for spherical confinement.
        """
        # Define the energy expression.
        energy_expr = ("0.5 * k * (r - R_conf)^2; "
                    "r = sqrt(x*x + y*y + z*z)")
        
        # Create the force and assign it to force group 3.
        sph_conf_force = CustomExternalForce(energy_expr)
        sph_conf_force.setForceGroup(forcegroup)
        
        # Add global parameters for the cutoff radius and force constant.
        sph_conf_force.addGlobalParameter("R_conf", R_conf)
        sph_conf_force.addGlobalParameter("k", k)
        
        # Add every particle to the force.
        for i in range(self.num_particles):
            sph_conf_force.addParticle(i, [])
        
        # Add the force to the system and store it.
        system.addForce(sph_conf_force)
        self.forceDict["SphericalConfinement"] = sph_conf_force
        return sph_conf_force

    def add_type_to_type_interaction(self, system, interaction_matrix, type_labels,
                                 mu=3.22, rc=1.78, cutoff=3.0, epsilon=1.0, force_group=5):
        """
        Adds type-to-type nonbonded interactions, automatically handling type consistency.
        
        Args:
            system (System): OpenMM system.
            type_list (list of str): List of type labels for each atom (e.g., ["A", "B", "C"]).
            interaction_matrix (2D list/array): Full interaction matrix aligned with type_labels.
            type_labels (list of str): List of all type labels corresponding to matrix rows/cols.
            mu (float): Steepness parameter.
            rc (float): Reference distance for tanh potential.
            cutoff (float): Cutoff distance for interactions.
            epsilon (float): Energy scaling factor.
            force_group (int): Force group index.
        """
        
        type_list = [atom.element for atom in self.topology.atoms()]
        print(type_list)
        # Unique types used in this polymer system
        used_types = sorted(set(type_list))
        print(f"[INFO] Types used in polymer: {used_types}")

        # Check that all used types are present in interaction matrix
        missing_types = [t for t in used_types if t not in type_labels]
        if missing_types:
            raise ValueError(f"[ERROR] Types found in topology but missing in interaction matrix: {missing_types}")

        # Types in matrix but not in topology (warn, but ignore them)
        unused_types = [t for t in type_labels if t not in used_types]
        if unused_types:
            print(f"[WARNING] Types defined in interaction matrix but not used in topology: {unused_types}")

        # Build mapping from type label to index in matrix
        type_to_idx = {label: idx for idx, label in enumerate(type_labels)}

        # Subset interaction matrix to only relevant types (reordered as used_types)
        subset_indices = [type_to_idx[t] for t in used_types]
        reduced_matrix = np.array(interaction_matrix)[np.ix_(subset_indices, subset_indices)]

        num_types = len(used_types)
        flat_interaction = reduced_matrix.flatten().tolist()

        # Energy expression
        energy_expr = "epsilon * interaction_map(t1, t2) * 0.5 * (1 + tanh(mu * (rc - r)));"

        # Define force
        type_force = CustomNonbondedForce(energy_expr)
        type_force.setForceGroup(force_group)
        type_force.setCutoffDistance(cutoff)
        type_force.addGlobalParameter('mu', mu)
        type_force.addGlobalParameter('rc', rc)
        type_force.addGlobalParameter('epsilon', epsilon)

        # Tabulated function for type interaction
        interaction_func = Discrete2DFunction(num_types, num_types, flat_interaction)
        type_force.addTabulatedFunction("interaction_map", interaction_func)

        type_force.addPerParticleParameter("t") 
        
        # Add per-particle type indices mapped to reduced matrix
        type_idx_map = {label: idx for idx, label in enumerate(used_types)}
        for type_label in type_list:
            type_force.addParticle([float(type_idx_map[type_label])])  # map original label to reduced index

        system.addForce(type_force)
        self.forceDict["TypeToType"] = type_force
        print(f"[INFO] Type-to-Type interaction added (force group {force_group}, {num_types} types).")
        return type_force
    
    def _get_type_list_from_topology(self,):
        """
        Extracts a list of atom types (as integers) from the topology's element symbols.
        
        Args:
            topology (Topology): OpenMM topology object.
        
        Returns:
            type_list (list of int): List of integer type indices.
            type_map (dict): Mapping from type symbol to index.
        """
        type_symbols = [atom.element for atom in self.topology.atoms()]
        unique_types = sorted(set(type_symbols))  # e.g., ['A', 'B', 'C']
        type_map = {symbol: idx for idx, symbol in enumerate(unique_types)}  # {'A':0, 'B':1, 'C':2}
        
        type_list = [type_map[symbol] for symbol in type_symbols]
        return type_list
