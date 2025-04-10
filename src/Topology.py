from openmm.app import Topology

class TopologyGenerator:
    def __init__(self):
        self.topology = Topology()
    
    def gen_top(self, chain_list):
        """
        Generates the molecular topology, adding chains, residues, atoms, and bonds.
        """
        kk=0
        for cid, chain_len in enumerate(chain_list):
            cid+=1
            chain = self.topology.addChain(f"C{cid}")  # Chain ID formatted to 5 digits
            previous_atom = None
            num_residues = chain_len//3
            small_residue_atoms = chain_len % 3
            for resid in range(1, num_residues+1):
                residue = self.topology.addResidue(f"L{resid}", chain)  # Residue ID formatted to 5 digits
                
                atom1 = self.topology.addAtom(f"C{cid}-L{resid}-1", f"A", residue, f"{kk}")
                atom2 = self.topology.addAtom(f"C{cid}-L{resid}-2", f"A", residue,f"{kk+1}")
                atom3 = self.topology.addAtom(f"C{cid}-L{resid}-3", f"A", residue, f"{kk+2}")
                
                # Add intra-residue bonds sequentially
                if previous_atom is not None:
                    self.topology.addBond(previous_atom, atom1)
                
                self.topology.addBond(atom1, atom2)
                self.topology.addBond(atom2, atom3)
                
                # Store last atom for the next connection
                previous_atom = atom3  # Last atom in the residue
                kk+=3
                
            for ii in range(small_residue_atoms):
                
                residue = self.topology.addResidue(f"L{resid+1}", chain) 
                atom = self.topology.addAtom(f"C{cid}-L{resid}-1", f"A", residue, f"{kk}")
                self.topology.addBond(previous_atom, atom)
                previous_atom = atom
                kk += 1
                
    def print_top(self, ):
        """
        Writes the generated topology to a formatted output file, with an atom count at the beginning.
        """
        atom_count = sum(1 for _ in self.topology.atoms())  # Count total atoms
        
        print(f"{atom_count}\n")  # Write total number of atoms as the first line
        print(f"{'Index':<20} {'Atom':<20} {'Element':<10} {'Residue':<10} {'Chain':<10}\n")  # Header
        ii=1
        for chain in self.topology.chains():
            for residue in chain.residues():
                for atom in residue.atoms():
                    print(f"{ii:<20} {atom.name:<20} {atom.element:<10} {residue.name:<10} {chain.id:<10}\n")
                    ii+=1
                        
    def save_top(self, filename="topology_info.txt"):
        """
        Writes the generated topology to a formatted output file, with an atom count at the beginning.
        """
        atom_count = sum(1 for _ in self.topology.atoms())  # Count total atoms
        
        with open(filename, "w") as f:
            f.write(f"{atom_count}\n")  # Write total number of atoms as the first line
            f.write(f"{'Atom':<20} {'Residue':<10} {'Chain':<10}\n")  # Header
            ii=1
            for chain in self.topology.chains():
                for residue in chain.residues():
                    for atom in residue.atoms():
                        f.write(f"{ii:<5} {atom.name:<20} {residue.name:<10} {chain.id:<10}\n")
                        ii+=1


