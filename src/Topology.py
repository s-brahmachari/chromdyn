from openmm.app import Topology
from pathlib import Path
import numpy as np

class TopologyGenerator:
    def __init__(self):
        self.topology = Topology()
    
    def gen_top(self, chain_lens, types="A", **kwargs):
        """
        Generates the molecular topology, adding chains, residues, atoms, and bonds.
        """
        atoms_per_residue = int(kwargs.get('atoms_per_residue', 1))
        chain_names = kwargs.get('chain_names', [f'C{xx}' for xx in range(1,len(chain_lens)+1)])
        isRing = kwargs.get('isRing',[0] * len(chain_lens))
        
        if type(types)==str:
            # check for a filename
            type_file = Path(types)
            if type_file.exists():
                atom_types = np.loadtxt(type_file, usecols=[1], dtype=str)
            elif types=='unique':
                # set unique names for each atom or monomer: M1, M2, ...
                atom_types = [f"M{xx}" for xx in range(1, sum(chain_lens)+1)]    
            elif len(types)<4:
                #assume the str is a name; use for all atoms
                atom_types = [types] * sum(chain_lens)
        elif type(types)==list:
            atom_types = types
        
        assert len(chain_names) == len(chain_lens), 'chain_names do not match chain_lens'
        assert len(atom_types) == sum(chain_lens), 'types file does not match chain_lens'
        
        kk=0
        for cid, chain_len in enumerate(chain_lens):
            chain = self.topology.addChain(chain_names[cid])  # Chain ID formatted to 5 digits
            previous_atom = None
            num_residues = chain_len//atoms_per_residue
            small_residue_atoms = chain_len % atoms_per_residue
            for resid in range(1, num_residues+1):
                residue = self.topology.addResidue(f"L{resid}", chain)  # Residue ID formatted to 5 digits
                for aid in range(atoms_per_residue):
                    atom = self.topology.addAtom(f"{chain_names[cid]}-L{resid}-{aid+1}", f"{atom_types[kk]}", residue, f"{kk}")
                
                    # Add intra-residue bonds sequentially
                    if previous_atom is not None:
                        self.topology.addBond(previous_atom, atom)
                    kk+=1
                    # Store atom for the next connection
                    previous_atom = atom  
                    
                
            for ii in range(small_residue_atoms):
                residue = self.topology.addResidue(f"L{resid+1}", chain) 
                atom = self.topology.addAtom(f"{chain_names[cid]}-L{resid}-{ii+1}", f"{atom_types[kk]}", residue, f"{kk}")
                self.topology.addBond(previous_atom, atom)
                previous_atom = atom
                kk += 1
        
        for idx, chain in enumerate(self.topology.chains()):
            if isRing[idx]>0:
                first_atom=list(chain.atoms())[0]
                last_atom = list(chain.atoms())[-1]
                self.topology.addBond(first_atom, last_atom)
            
        
        
    def print_top(self, ):
        """
        Writes the generated topology to a formatted output file, with an atom count at the beginning.
        """
        atom_count = sum(1 for _ in self.topology.atoms())  # Count total atoms
        
        print(f"{atom_count}\n")  # Write total number of atoms as the first line
        print(f"{'ID':<10} {'Name':<20} {'Type':<10} {'Loci':<10} {'Chain':<10}\n")  # Header
        for chain in self.topology.chains():
            for residue in chain.residues():
                for atom in residue.atoms():
                    print(f"{atom.id:<10} {atom.name:<20} {atom.element:<10} {residue.name:<10} {chain.id:<10}\n")
                        
    def save_top(self, filename='topology.txt'):
        """
        Writes the generated topology to a formatted output file, with an atom count at the beginning.
        """
        atom_count = sum(1 for _ in self.topology.atoms())  # Count total atoms
        
        with open(filename, "w") as f:
            f.write(f"{atom_count}\n")  # Write total number of atoms as the first line
            f.write(f"{'ID':<10} {'Name':<20} {'Type':<10} {'Loci':<10} {'Chain':<10}\n")  # Header
            for chain in self.topology.chains():
                for residue in chain.residues():
                    for atom in residue.atoms():
                        f.write(f"{atom.id:<10} {atom.name:<20} {atom.element:<10} {residue.name:<10} {chain.id:<10}\n")
            


