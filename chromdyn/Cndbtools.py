import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial.distance import pdist
import json
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
from Analyzers import compute_RG


class TopologyData:
    """
    Helper class to access topology data via attributes.
    Parses the nested JSON dictionary into flat statistics.
    """
    def __init__(self, data_dict):
        # Raw data
        self.chains = data_dict.get("chains", [])
        self.bonds = data_dict.get("bonds", [])
        
        # --- Pre-calculated Statistics (What you asked for) ---
        
        # 1. Number of Chains
        self.n_chains = len(self.chains)
        
        # 2. Number of Bonds
        self.n_bonds = len(self.bonds)
        
        # 3. Total Number of Atoms (Beads)
        self.n_atoms = 0
        self._calculate_atoms()

    def _calculate_atoms(self):
        """Internal helper to count atoms."""
        count = 0
        for chain in self.chains:
            for residue in chain['residues']:
                count += len(residue['atoms'])
        self.n_atoms = count

    @property
    def chain_info(self):
        """
        Returns a summary list of tuples: [(ChainID, NumAtoms), ...]
        Example: [('C1', 100), ('C2', 50)]
        """
        info = []
        for chain in self.chains:
            # Count atoms in this chain
            n_atoms_in_chain = sum(len(res['atoms']) for res in chain['residues'])
            info.append((chain['id'], n_atoms_in_chain))
        return info

    def __repr__(self):
            return (f"<TopologyData: {self.n_chains} chains, "
                    f"{self.n_atoms} beads, {self.n_bonds} bonds>")

        
class ChromatinTrajectory:
    def __init__(self, filename: str = None):
        # initialize attributes
        self.cndb = None
        self.filename = filename
        self.Nbeads = 0
        self.Nframes = 0
        self.ChromSeq = []
        self.uniqueChromSeq = set()
        self.dictChromSeq = {}
        self.topology = None
        self.box_vectors = None

        # if filename is provided, load the trajectory
        if filename:
            self.load(filename)

    def load(self, filename: str):
        """call external load_trajectory function"""
        return load_trajectory(self, filename)

    def xyz(self, frames=[0, None, 1], beadSelection=None, XYZ=[0, 1, 2]):
        """call external get_xyz function"""
        return get_xyz(self, frames, beadSelection, XYZ)
    
    def close(self):
        """call external close_trajectory function"""
        close_trajectory(self)

    def __del__(self):
        """destruct the object and try to close the file"""
        self.close()


# For using as independent functions
# self should be the object of the class ChromatinTrajectory
def ndb2cndb(self, filename):
        R"""
        Converts an **ndb** file format to **cndb**.
        
        Args:
            filename (path, required):
                 Path to the ndb file to be converted to cndb.
        """
        Main_chrom      = ['ChrA','ChrB','ChrU'] # Type A B and Unknow
        Chrom_types     = ['ZA','OA','FB','SB','TB','LB','UN']
        Chrom_types_NDB = ['P ','S ','B1','B2','B3','B4','UN'] #P-A1, S-A2 to make it compatible with or systems
        Res_types_PDB   = ['ASP', 'GLU', 'ARG', 'LYS', 'HIS', 'HIS', 'GLY']
        Type_conversion = {'P ': 0,'S ' : 1,'B1' : 2,'B2' : 3,'B3' : 4,'B4' : 5,'UN' : 6}
        title_options = ['HEADER','OBSLTE','TITLE ','SPLT  ','CAVEAT','COMPND','SOURCE','KEYWDS','EXPDTA','NUMMDL','MDLTYP','AUTHOR','REVDAT','SPRSDE','JRNL  ','REMARK']
        model          = "MODEL     {0:4d}"
        atom           = "ATOM  {0:5d} {1:^4s}{2:1s}{3:3s} {4:1s}{5:4d}{6:1s}   {7:8.3f}{8:8.3f}{9:8.3f}{10:6.2f}{11:6.2f}          {12:>2s}{13:2s}"
        ter            = "TER   {0:5d}      {1:3s} {2:1s}{3:4d}{4:1s}"

        file_ndb = filename + str(".ndb")
        name     = filename + str(".cndb")

        if os.path.exists(name):
            print(f"File '{name}' already exists. Skipping conversion.")
            return name  # File already exists

        cndbf = h5py.File(name, 'w')
        
        ndbfile = open(file_ndb, "r")
        
        loop = 0
        types = []
        types_bool = True
        loop_list = []
        x = []
        y = [] 
        z = []

        frame = 0

        for line in ndbfile:
    
            entry = line[0:6]

            info = line.split()


            if 'MODEL' in entry:
                frame += 1

                inModel = True

            elif 'CHROM' in entry:

                subtype = line[16:18]
                #print(subtype)

                types.append(subtype)
                x.append(float(line[40:48]))
                y.append(float(line[49:57]))
                z.append(float(line[58:66]))

            elif 'END' in entry:
                if types_bool:
                    typelist = [Type_conversion[x] for x in types]
                    #print(typelist)
                    cndbf['types'] = typelist
                    types_bool = False

                positions = np.vstack([x,y,z]).T
                cndbf[str(frame)] = positions
                x = []
                y = []
                z = []

            elif 'LOOPS' in entry:
                loop_list.append([int(info[1]), int(info[2])])
                loop += 1
        
        if loop > 0:
            cndbf['loops'] = loop_list

        cndbf.close()
        return(name)

def load_trajectory(self, filename):
    R"""
    Loads cndb file, including types, topology, and PBC box vectors.
    """
    self.filename = filename
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    self.cndb = h5py.File(filename, 'r')

    frame_keys = sorted([k for k in self.cndb.keys() if k.isdigit()], key=int)
    self.Nframes = len(frame_keys)
    
    if self.Nframes == 0:
        print("Warning: No frames found in file.")
        return self

    # --- 1. Load Bead Number ---
    first_frame_data = self.cndb[frame_keys[0]]
    self.Nbeads = first_frame_data.shape[0]
    
    # --- 2. Load Types ---
    if 'types' in self.cndb:
        raw_types = self.cndb['types']
        self.ChromSeq = [t.decode('utf-8') if isinstance(t, bytes) else t for t in raw_types]
    else:
        print("  Warning: 'types' dataset not found. Assuming uniform bead types.")
        first_key = next(iter(self.cndb.keys()))
        n_beads = self.cndb[first_key].shape[0]
        self.ChromSeq = ['U'] * n_beads

    self.uniqueChromSeq = set(self.ChromSeq)
    self.dictChromSeq = {tt: [i for i, e in enumerate(self.ChromSeq) if e == tt] for tt in self.uniqueChromSeq}

    # --- 3. Load Topology JSON---
    self.topology = None
    if 'topology_json' in self.cndb:
        try:
            json_str = self.cndb['topology_json'][0]
            if isinstance(json_str, bytes): json_str = json_str.decode('utf-8')
            self.topology = TopologyData(json.loads(json_str))
        except Exception as e:
            print(f"  Warning: Failed to load topology data: {e}")

    # --- 4. Load Box Vectors ---
    # We check the first frame to see if box attribute exists
    if 'box' in first_frame_data.attrs:
        self.box_vectors = np.zeros((self.Nframes, 3, 3))
        for i, key in enumerate(frame_keys):
            if 'box' in self.cndb[key].attrs:
                self.box_vectors[i] = self.cndb[key].attrs['box']
            else:
                # If one frame is missing, use the previous frame or raise an error
                if i > 0: self.box_vectors[i] = self.box_vectors[i-1]
    else:
        self.box_vectors = None

    print(f"Loaded {self.filename}: {self.Nframes} frames, {self.Nbeads} beads.")
    if self.topology: print(f"Topology: {self.topology}")
    if self.box_vectors is not None: print(f"Box vectors loaded. Shape: {self.box_vectors.shape}")
    
    return self
    
def get_xyz(self, frames=[0, None, 1], beadSelection=None, XYZ=[0, 1, 2]):
    R"""
    Get the selected beads' 3D position from a **cndb** or **ndb** for multiple frames.
    """
    if self.cndb is None:
        raise RuntimeError("No file loaded. Call load() first.")
    # initialize frame list
    frame_list = []
    # check beadSelection 
    if beadSelection is None:
        selection = np.arange(self.Nbeads)
    else:
        selection = np.array(beadSelection)
    #print(f"Choosing Beads ID: {selection}")

    # check frames number
    start, end, step = frames
    if end is None:
        end = self.Nframes #+ 1 I'm not sure if I need this, in OpenMiChroM one'll need that.

    # simple range check
    if start < 0: start = 0
    if end > self.Nframes: end = self.Nframes

    for i in range(start, end, step):
        try:
            key = str(i)
            if key not in self.cndb:
                continue
            frame_data = np.array(self.cndb[key])
            #print(f"Data structure of frame {i}: {frame_data.shape}")
            selected_data = np.take(np.take(frame_data, selection, axis=0), XYZ, axis=1)
            #coords = frame_data[selection][:, XYZ]
            frame_list.append(selected_data)
        except KeyError:
            print(f"Warning: Frame {i} doesn't exit, skip this frame")
        except Exception as e:
            print(f"Error occurs: {e} when extract data from frame {i}.")

    # Return the extracted data
    return np.array(frame_list)

def close_trajectory(traj_instance):
    if hasattr(traj_instance, 'cndb') and traj_instance.cndb:
        traj_instance.cndb.close()


#Check if cupy is available
try:
    import cupy as cp
    print("Cupy is available")
    import cupy as cp
    #from cupy import fft
    from typing import Dict, Tuple, Union, List
    # velocity autocorrelation function
    def _autocorrFFT_gpu(x_multi_dim: cp.ndarray) -> cp.ndarray:
        """
        (Vectorized, GPU) using cupy FFT to calculate autocorrelation of multi-dimensional array.
        """
        N = x_multi_dim.shape[0]
        
        # RATIONALE: excecute FFT on GPU
        F = cp.fft.fft(x_multi_dim, n=2*N, axis=0)
        res = cp.fft.ifft(F * F.conjugate(), axis=0)
        res = res[:N, ...].real # 在 GPU 上切片
        
        # RATIONALE: create normalization vector on GPU
        norm_shape = [N] + [1] * (x_multi_dim.ndim - 1)
        norm = (N - cp.arange(0, N)).reshape(norm_shape)
        
        return res / norm

    def calculate_vacf_gpu(
        velocities: np.ndarray,
        bead_types: np.ndarray,
        sampling_step: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        (Vectorized, GPU) using cupy to calculate velocity autocorrelation function (VACF) C(t).
        """
        type_indices = {utype: np.where(bead_types == utype)[0] for utype in bead_types}   

        # 1. sample on CPU to save VRAM
        sampled_vels_np = velocities[::sampling_step, :, :]
        
        # 2. transfer data to GPU
        sampled_vels_cp = cp.asarray(sampled_vels_np)
        
        # 3. calculate VACF on GPU
        vacf_components_cp = _autocorrFFT_gpu(sampled_vels_cp)
        
        # 4. sum (Cx + Cy + Cz) and transpose
        vacf_all_beads_cp = cp.sum(vacf_components_cp, axis=2).T
                
        # 5. calculate average on GPU by type
        results_cp = {}
        results_cp['general'] = cp.mean(vacf_all_beads_cp, axis=0)
        
        # transfer index to GPU for fancy indexing
        type_indices_cp = {k: cp.asarray(v) for k, v in type_indices.items()}
        
        for btype, indices_cp in type_indices_cp.items():
            if len(indices_cp) > 0:
                results_cp[btype] = cp.mean(vacf_all_beads_cp[indices_cp, :], axis=0)
                
        # 6. transfer final small result array back to CPU
        results_np = {k: cp.asnumpy(v) for k, v in results_cp.items()}
        
        # 7. clean GPU memory
        del sampled_vels_cp, vacf_components_cp, vacf_all_beads_cp, results_cp, type_indices_cp
        
        return results_np

    def calculate_spatial_vel_corr_gpu(
        coords: np.ndarray,
        velocities: np.ndarray,
        bead_types: np.ndarray,
        dist_range: float,
        sampling_step: int = 1,
        num_bins: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        (Vectorized, GPU) calculate spatial velocity correlation C(r) = <v_i · v_j>.
        
        In CPU, loop over frames, but calculate all O(N^2) on GPU.
        """
        n_frames, n_beads, _ = coords.shape
        
        # 1. set Bins and type pairs (lightweight on CPU)
        bins = np.linspace(0, dist_range, num_bins + 1, dtype=np.float32)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        
        unique_types = np.unique(bead_types)
        type_pairs = ["-".join(sorted(pair)) for pair in np.array(np.meshgrid(unique_types, unique_types)).T.reshape(-1, 2)]
        type_pairs = sorted(list(set(type_pairs)))
        
        # --- 2. RATIONALE: pre-calculate masks and *once* transfer to GPU ---
        rows, cols = np.triu_indices(n_beads, k=1)
        rows_cp = cp.asarray(rows)
        cols_cp = cp.asarray(cols)
        
        bead_types_i = bead_types[rows]
        bead_types_j = bead_types[cols]
        
        pair_masks_cp = {}
        for key in type_pairs:
            t1, t2 = key.split('-')
            if t1 == t2:
                mask = (bead_types_i == t1) & (bead_types_j == t2)
            else:
                mask = ((bead_types_i == t1) & (bead_types_j == t2)) | \
                    ((bead_types_i == t2) & (bead_types_j == t1))
            pair_masks_cp[key] = cp.asarray(mask)

        bins_cp = cp.asarray(bins)

        # 3. RATIONALE: initialize accumulators on GPU
        total_corr_cp = {key: cp.zeros(num_bins) for key in ['general'] + type_pairs}
        counts_cp = {key: cp.zeros(num_bins) for key in ['general'] + type_pairs}

        # 4. loop over sampled frames on CPU
        for frame_idx in range(0, n_frames, sampling_step):
            
            # 5. RATIONALE: transfer *only the current frame* to GPU
            frame_coords_cp = cp.asarray(coords[frame_idx], dtype=cp.float32)
            frame_vels_cp = cp.asarray(velocities[frame_idx], dtype=cp.float32)
            
            # 6. RATIONALE: execute all O(N^2) calculations on GPU
            
            # a. distance matrix: use broadcast (N, 1, 3) - (1, N, 3) -> (N, N, 3) -> (N, N)
            dist_matrix_cp = cp.linalg.norm(frame_coords_cp[:, None, :] - frame_coords_cp[None, :, :], axis=2)
            
            # b. dot product matrix: (N, 3) @ (3, N) -> (N, N)
            v_dot_v_cp = frame_vels_cp @ frame_vels_cp.T
            
            # c. extract upper triangle
            all_dists_cp = dist_matrix_cp[rows_cp, cols_cp]
            all_dots_cp = v_dot_v_cp[rows_cp, cols_cp]
            
            # d. digitize
            all_bin_indices_cp = cp.digitize(all_dists_cp, bins_cp[1:])
            
            # 7. RATIONALE: use bincount on GPU for vectorized accumulation
            valid_mask_cp = (all_bin_indices_cp < num_bins)
            
            # accumulate 'general'
            valid_bins_cp = all_bin_indices_cp[valid_mask_cp]
            valid_dots_cp = all_dots_cp[valid_mask_cp]
            total_corr_cp['general'] += cp.bincount(valid_bins_cp, weights=valid_dots_cp, minlength=num_bins)
            counts_cp['general'] += cp.bincount(valid_bins_cp, minlength=num_bins)
            
            # accumulate by type
            for key, type_mask_cp in pair_masks_cp.items():
                final_mask_cp = valid_mask_cp & type_mask_cp
                if cp.any(final_mask_cp):
                    type_bins_cp = all_bin_indices_cp[final_mask_cp]
                    type_dots_cp = all_dots_cp[final_mask_cp]
                    total_corr_cp[key] += cp.bincount(type_bins_cp, weights=type_dots_cp, minlength=num_bins)
                    counts_cp[key] += cp.bincount(type_bins_cp, minlength=num_bins)

        # 8. calculate final average on GPU
        results_cp = {}
        for key in total_corr_cp:
            
            # replace where to be compatible with old version of cupy

            # 1. copy counts to avoid modifying original accumulators
            counts_safe_cp = counts_cp[key].copy()
            
            # 2. create mask for all bins with count 0
            zero_mask_cp = (counts_safe_cp == 0)
            
            # 3. replace these 0s with 1.0. This doesn't affect the result,
            #    because we will set these positions to NaN later.
            counts_safe_cp[zero_mask_cp] = 1.0 
            
            # 4. perform regular division (now safe)
            corr_cp = total_corr_cp[key] / counts_safe_cp
            
            # 5. set positions marked as 0 to NaN
            corr_cp[zero_mask_cp] = cp.nan
            
            results_cp[key] = corr_cp
                
        # 9. transfer final small result array back to CPU
        results_np = {k: cp.asnumpy(v) for k, v in results_cp.items()}
        results_np['bin_centers'] = bin_centers
                
        return results_np
except ImportError:
    print("Cupy is not available, cannot load velocity analysis functoin.")
        
# Plotting

def _draw_pbc_box(ax, box_a, center=np.array([0,0,0]), color="gray", linestyle="--", linewidth=1, alpha=0.7):
    """Helper function to draw a cubic PBC box centered at `center`."""
    half_a = box_a / 2.0
    min_coords = center - half_a
    max_coords = center + half_a

    corners = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]]
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    

def recenter_coordinates_v3(polymer_coords_list, box_vectors):
    """
    Recenter a list of polymer chain coordinates based on generic box vectors.
    The system's COM is moved to the geometric center of the box defined by box_vectors.

    Args:
        polymer_coords_list (list of np.ndarray): Coordinates.
        box_vectors (np.ndarray): 3x3 array [[ax, ay, az], [bx, by, bz], [cx, cy, cz]].

    Returns:
        list of np.ndarray: Recentered coordinates.
    """
    if not polymer_coords_list or not any(chain.size > 0 for chain in polymer_coords_list):
        return polymer_coords_list

    # 1. Calculate System COM
    all_coords_flat = np.vstack([chain for chain in polymer_coords_list 
                                 if chain.ndim == 2 and chain.shape[1] == 3 and chain.size > 0])
    if all_coords_flat.size == 0:
        return polymer_coords_list

    current_com = np.mean(all_coords_flat, axis=0)

    # 2. Calculate Box Geometric Center
    # Center = Origin + 0.5 * (vec_a + vec_b + vec_c)
    # Assuming origin is (0,0,0)
    box_center = 0.5 * np.sum(box_vectors, axis=0)

    # 3. Shift
    shift_vector = box_center - current_com

    recentered_coords_list = []
    for chain_coords in polymer_coords_list:
        if chain_coords.ndim == 2 and chain_coords.shape[1] == 3 and chain_coords.size > 0:
            recentered_coords_list.append(chain_coords + shift_vector)
        else:
            recentered_coords_list.append(chain_coords)

    return recentered_coords_list

def _draw_generic_box(ax, box_vectors, color='k', alpha=0.5, linewidth=1.0):
    """
    Draws a wireframe parallelepiped defined by box_vectors starting at (0,0,0).
    """
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = box_vectors[0] # a
    v2 = box_vectors[1] # b
    v3 = box_vectors[2] # c

    # The 8 corners of the box
    # 0: origin
    # 1: a
    # 2: b
    # 3: c
    # 4: a+b
    # 5: a+c
    # 6: b+c
    # 7: a+b+c
    corners = np.array([
        v0,
        v1,
        v2,
        v3,
        v1 + v2,
        v1 + v3,
        v2 + v3,
        v1 + v2 + v3
    ])

    # Define the 12 edges by connecting corner indices
    edges = [
        [0, 1], [0, 2], [0, 3], # Origin to axes
        [1, 4], [1, 5],         # From a
        [2, 4], [2, 6],         # From b
        [3, 5], [3, 6],         # From c
        [4, 7], [5, 7], [6, 7]  # To far corner
    ]

    lines = []
    for start_idx, end_idx in edges:
        lines.append([corners[start_idx], corners[end_idx]])

    # Plot using Line3DCollection for efficiency
    lc = Line3DCollection(lines, colors=color, alpha=alpha, linewidths=linewidth)
    ax.add_collection3d(lc)
    
    return corners # Return corners to help set axis limits



# Universal visualization function with optional PBC(draw box)
def visualize(traj, select_frame=0, axis_limits=None,
                  colors=None, outputName=None, isring=False, r=None,
                  recenter=False, color_mode='chain', types = None, PBC=False):
    """
    Universal visualization function for polymer chains with optional PBC box.
    
    Args:
        traj: Trajectory object containing coordinate and topology data.
        select_frame (int): Frame index to visualize (default: 0).
        axis_limits (tuple): Manual axis limits as (x_min, x_max, y_min, y_max, z_min, z_max).
        colors (list): Custom colors for chains.
        outputName (str): If provided, save the plot to this filename instead of displaying.
        isring (bool): If True, connect the last bead to the first bead (default: False).
        r (float): Physical radius of beads in nm for size calculation.
        recenter (bool): If True, recenter coordinates (default: True).
        color_mode (str): 'chain' (default) or 'type' for coloring scheme.
        types (list): Custom type sequence to override trajectory types.
        PBC (bool): If True, draw periodic boundary box and use box-based recentering (default: False).
    """

    if PBC:
        recenter = True
        print("PBC is enabled, automatically recentering.")

    # --- 1. Data & Topology Check ---
    if not hasattr(traj, 'topology') or traj.topology is None:
        print("Error: Topology not found in trajectory.")
        return

    chain_info_list = traj.topology.chain_info
    n_chains = len(chain_info_list)
    bead_counts = [count for _, count in chain_info_list]
    
    # Calculate indices
    cumulative_indices = np.cumsum([0] + bead_counts)
    chain_selections = [
        np.arange(start, end) 
        for start, end in zip(cumulative_indices[:-1], cumulative_indices[1:])
    ]

    # --- 2. Load Coordinates (With Fix) ---
    polymer_coords_orig = []
    try:
        for sel in chain_selections:
            # [CRITICAL FIX]: Convert numpy array 'sel' to list using .tolist()
            # This prevents "truth value of an array" errors inside traj.xyz
            sel_list = sel.tolist()
            
            # xyz returns (n_frames, n_beads, 3), take [0] for single frame
            data = traj.xyz(frames=[select_frame, select_frame + 1, 1], beadSelection=sel_list)
            
            if data.shape[0] > 0:
                polymer_coords_orig.append(np.nan_to_num(data[0]))
            else:
                polymer_coords_orig.append(np.array([]))
                
    except Exception as e:
        print(f"Error loading coordinates at frame {select_frame}: {e}")
        return

    # Check if we got data
    if not polymer_coords_orig or all(c.size == 0 for c in polymer_coords_orig):
        print(f"Error: No valid coordinate data found for frame {select_frame}.")
        return

    if PBC:
        # --- 3. Load Box Vectors ---
        box_vectors = None
        if hasattr(traj, 'box_vectors') and (traj.box_vectors is not None):
            try:
                box_vectors = traj.box_vectors[select_frame]
            except:
                pass # Fallback below
        
        if box_vectors is None:
            print("Warning: No box vectors found. Assuming 100nm cube.")
            box_vectors = np.eye(3) * 100.0

    # --- 4. Recenter ---
    if recenter:
        print(f"Recentering coordinates...")
        polymer_coords = recenter_coordinates_v3(polymer_coords_orig, box_vectors)
    else:
        polymer_coords = polymer_coords_orig

    # --- 5. Color Setup ---
    chain_colors_list = [] # For 'chain' mode
    bead_colors_list = []  # For 'type' mode (list of color arrays)
    type_legend_handles = {}

    # Mode: Type
    if color_mode == 'type':
        # Get all types
        types_seq = getattr(traj, 'ChromSeq', getattr(traj, 'types', None))
        # if types are provided, use them
        if types is not None:
            types_seq = types   
            print(f"Using provided types: {types_seq}")
        if types_seq is None:
            print("Warning: 'type' mode requested but no types found. Switching to 'chain'.")
            color_mode = 'chain'
        else:
            print(f"Using types from trajectory: {types_seq}")
            unique_types = sorted(list(set(types_seq)))
            cmap = plt.get_cmap('tab10')
            type_map = {t: cmap(i % 10) for i, t in enumerate(unique_types)}
            
            # Convert all types to colors once
            all_bead_colors = np.array([type_map[t] for t in types_seq])
            
            # Slice colors for each chain
            for sel in chain_selections:
                bead_colors_list.append(all_bead_colors[sel])
            
            # Create Legend Handles
            for t, c in type_map.items():
                type_legend_handles[t] = plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=c, markersize=10, label=f"Type {t}")

    # Mode: Chain
    if color_mode == 'chain':
        if colors is None:
            cmap = plt.get_cmap('tab10')
            chain_colors_list = [cmap(i % 10) for i in range(n_chains)]
        else:
            chain_colors_list = colors

    # --- 6. Plotting ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter_collections = []
    
    for i, chain in enumerate(polymer_coords):
        if chain.size == 0: continue
        
        xs, ys, zs = chain[:, 0], chain[:, 1], chain[:, 2]
        
        # Determine Colors and Plot arguments
        if color_mode == 'chain':
            c_val = chain_colors_list[i] # Single RGBA tuple
            line_c = c_val
            line_a = 0.7
            label_val = f'Chain {i+1}' if n_chains <= 10 else ""
            
            # [FIX]: Use 'color' for single color to avoid warning
            kwargs_scatter = {'color': c_val} 
        else:
            c_val = bead_colors_list[i] # Array of RGBA tuples
            line_c = 'gray' 
            line_a = 0.3
            label_val = "" 
            
            # [FIX]: Use 'c' for color array
            kwargs_scatter = {'c': c_val}

        # Draw Lines
        if isring:
            ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), np.append(zs, zs[0]),
                    color=line_c, alpha=line_a, linewidth=2.0)
        else:
            ax.plot(xs, ys, zs, color=line_c, alpha=line_a, linewidth=2.0)
        
        # Draw Beads
        initial_s = 1 if r is not None else 20
        sc = ax.scatter(xs, ys, zs, alpha=0.8, s=initial_s, label=label_val, **kwargs_scatter)
        scatter_collections.append(sc)

    if PBC:
        # --- 7. Draw PBC Box ---
        box_corners = _draw_generic_box(ax, box_vectors)

        # Labels & Legend
        ax.set_xlabel(r"X ($\sigma$)"); ax.set_ylabel(r"Y ($\sigma$)"); ax.set_zlabel(r"Z ($\sigma$)")
        
        if color_mode == 'type':
            ax.legend(handles=type_legend_handles.values(), loc='best')
        elif color_mode == 'chain' and n_chains <= 10:
            ax.legend(loc='best')

        title_str = f"Frame {select_frame} (c:{color_mode})"
        if recenter: title_str += " (Recentered)"
        ax.set_title(title_str)

    # --- 8. View & Limits ---
    if r is not None:
        ax.set_proj_type('ortho'); ax.view_init(elev=30, azim=-45)
    else:
        ax.set_proj_type('persp')

    if axis_limits:
        x_min, x_max, y_min, y_max, z_min, z_max = axis_limits
    elif PBC:
        all_poly = np.vstack([c for c in polymer_coords if c.size > 0])
        all_points = np.vstack([all_poly, box_corners])
        min_ext = np.min(all_points, axis=0)
        max_ext = np.max(all_points, axis=0)
        center = (min_ext + max_ext) / 2.0
        span = max_ext - min_ext
        max_span = np.max(span) if np.max(span) > 0 else 10.0
        buffer = max_span * 0.55
        x_min, x_max = center[0]-buffer, center[0]+buffer
        y_min, y_max = center[1]-buffer, center[1]+buffer
        z_min, z_max = center[2]-buffer, center[2]+buffer
    else:
        all_poly = np.vstack([c for c in polymer_coords if c.size > 0])
        min_ext = np.min(all_poly, axis=0)
        max_ext = np.max(all_poly, axis=0)
        center = (min_ext + max_ext) / 2.0
        span = max_ext - min_ext
        max_span = np.max(span) if np.max(span) > 0 else 10.0
        buffer = max_span * 0.55
        x_min, x_max = center[0]-buffer, center[0]+buffer
        y_min, y_max = center[1]-buffer, center[1]+buffer
        z_min, z_max = center[2]-buffer, center[2]+buffer

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([1, 1, 1])

    # --- 9. Update Scatter Size ---
    if r is not None:
        fig.canvas.draw()
        data_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        bbox = ax.get_window_extent()
        if bbox and data_range > 0:
            points_per_unit = (bbox.width * 72 / fig.get_dpi()) / data_range
            s_phys = np.clip(np.pi * ((r * points_per_unit) ** 2), 0.01, 50000)
            for sc in scatter_collections:
                sc.set_sizes(np.full(len(sc.get_offsets()), s_phys))

    # --- 10. Output & Display ---
    if outputName:
        plt.savefig(outputName, dpi=150)
        print(f"Plot saved to {outputName}")
        plt.close(fig)
    else:
        plt.show() 


def visualize_animation(traj, start_frame=0, end_frame=None, fps=20,
                            axis_limits=None, colors=None, outputName=None,
                            isring=False, r=None, recenter=True, 
                            color_mode='chain', types = None, PBC = False):
    """
    Universal animation visualization function for polymer chains with optional PBC box.
    
    Args:
        traj: Trajectory object containing coordinate and topology data.
        start_frame (int): Starting frame index (default: 0).
        end_frame (int): Ending frame index (default: None, uses all frames).
        fps (int): Frames per second for animation (default: 20).
        axis_limits (tuple): Manual axis limits as (x_min, x_max, y_min, y_max, z_min, z_max).
        colors (list): Custom colors for chains.
        outputName (str): If provided, save animation to this filename.
        isring (bool): If True, connect the last bead to the first bead (default: False).
        r (float): Physical radius of beads in nm for size calculation.
        recenter (bool): If True, recenter coordinates (default: True).
        color_mode (str): 'chain' (default) or 'type' for coloring scheme.
        types (list): Custom type sequence to override trajectory types.
        PBC (bool): If True, draw periodic boundary box and use box-based recentering (default: False).
    
    Returns:
        Animation object if no outputName is provided.
    """
    if PBC:
        recenter = True
        print("PBC is True, recentering coordinates automatically.")

    # --- 1. Data Loading ---
    if not hasattr(traj, 'topology') or traj.topology is None:
        print("Error: Topology not found.")
        return

    chain_info = traj.topology.chain_info
    n_chains = len(chain_info)
    bead_counts = [c[1] for c in chain_info]
    cumulative_indices = np.cumsum([0] + bead_counts)
    chain_selections = [np.arange(s, e) for s, e in zip(cumulative_indices[:-1], cumulative_indices[1:])]

    total_frames = traj.Nframes
    if end_frame is None or end_frame > total_frames: end_frame = total_frames
    if start_frame < 0: start_frame = 0
    
    print(f"Loading frames {start_frame} to {end_frame}...")

    # --- 2. Load Coordinates (Defensive Fix) ---
    try:
        polymer_coords_all_chains_orig = []
        for sel in chain_selections:
            # [FIX]: Convert to list to prevent 'truth value of array' error
            sel_list = sel.tolist() 
            data = traj.xyz(frames=[start_frame, end_frame, 1], beadSelection=sel_list)
            polymer_coords_all_chains_orig.append(np.nan_to_num(data))
    except Exception as e:
        print(f"Error calling traj.xyz: {e}")
        return

    if not polymer_coords_all_chains_orig: return
    num_anim_frames = end_frame - start_frame

    # --- 3. Load Box Vectors (if PBC)---
    if PBC:
        if hasattr(traj, 'box_vectors') and (traj.box_vectors is not None):
            box_vectors_range = traj.box_vectors[start_frame:end_frame]
        else:
            box_vectors_range = np.tile(np.eye(3) * 100.0, (num_anim_frames, 1, 1))

    # --- 4. Color Setup ---
    chain_colors_list = [] 
    bead_colors_list = []  
    type_legend_handles = {} 

    if color_mode == 'type':
        types_seq = getattr(traj, 'ChromSeq', getattr(traj, 'types', None))
        # if types are provided, use them
        if types is not None:
            types_seq = types   
        if types_seq is None:
            color_mode = 'chain'
        else:
            unique_types = sorted(list(set(types_seq)))
            cmap = plt.get_cmap('tab10')
            type_map = {t: cmap(i % 10) for i, t in enumerate(unique_types)}
            all_bead_colors = np.array([type_map[t] for t in types_seq])
            for sel in chain_selections:
                bead_colors_list.append(all_bead_colors[sel])
            for t, c in type_map.items():
                type_legend_handles[t] = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=f"Type {t}")

    if color_mode == 'chain':
        if colors is None:
            cmap = plt.get_cmap('tab10')
            chain_colors_list = [cmap(i % 10) for i in range(n_chains)]
        else:
            chain_colors_list = colors

    # --- 5. Process Frames ---
    all_frames_processed = []
    for i in range(num_anim_frames):
        frame_coords = [chain_data[i] for chain_data in polymer_coords_all_chains_orig]
        
        if PBC and recenter:
            box = box_vectors_range[i]
            processed = recenter_coordinates_v3(frame_coords, box)
            all_frames_processed.append(processed)
        elif recenter:
            # Simple recentering - move COM to origin
            all_coords_flat = np.vstack([chain for chain in frame_coords 
                                        if chain.ndim == 2 and chain.shape[1] == 3 and chain.size > 0])
            if all_coords_flat.size > 0:
                current_com = np.mean(all_coords_flat, axis=0)
                processed = [chain - current_com if chain.size > 0 else chain for chain in frame_coords]
                all_frames_processed.append(processed)
            else:
                all_frames_processed.append(frame_coords)
        else:
            all_frames_processed.append(frame_coords)

    # --- 6. Limits ---
    if axis_limits:
        x_min, x_max, y_min, y_max, z_min, z_max = axis_limits
    elif PBC:
        # Fast auto-limit using Frame 0 Box
        v = box_vectors_range[0]
        center = 0.5 * np.sum(v, axis=0)
        max_span = max(np.linalg.norm(v[0]), np.linalg.norm(v[1]), np.linalg.norm(v[2]))
        buffer = max_span * 0.6
        x_min, x_max = center[0]-buffer, center[0]+buffer
        y_min, y_max = center[1]-buffer, center[1]+buffer
        z_min, z_max = center[2]-buffer, center[2]+buffer
    else:
        # Auto-limit using all frames
        all_points = []
        for frame_coords in all_frames_processed:
            for chain in frame_coords:
                if chain.size > 0:
                    all_points.append(chain)
        
        if all_points:
            all_points = np.vstack(all_points)
            min_ext = np.min(all_points, axis=0)
            max_ext = np.max(all_points, axis=0)
            center = (min_ext + max_ext) / 2.0
            span = max_ext - min_ext
            max_span = np.max(span) if np.max(span) > 0 else 10.0
            buffer = max_span * 0.6
            x_min, x_max = center[0]-buffer, center[0]+buffer
            y_min, y_max = center[1]-buffer, center[1]+buffer
            z_min, z_max = center[2]-buffer, center[2]+buffer
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = -50, 50, -50, 50, -50, 50

    # --- 7. Setup Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    if r is not None:
        ax.set_proj_type('ortho'); ax.view_init(elev=30, azim=-45)
    else:
        ax.set_proj_type('persp')
    ax.set_box_aspect([1, 1, 1])

    scatters = []
    lines = []
    initial_s = 1 if r is not None else 20

    for i in range(n_chains):
        if color_mode == 'chain':
            c_val = chain_colors_list[i]
            line_c = c_val
            line_a = 0.5
            # [FIX] Use 'color' for single RGBA
            sc = ax.scatter([], [], [], color=c_val, alpha=0.7, s=initial_s)
        else:
            # Type mode: colors set in update
            line_c = 'gray'
            line_a = 0.3
            # [FIX] Initialize with empty, color set later
            sc = ax.scatter([], [], [], alpha=0.7, s=initial_s)
            
        ln, = ax.plot([], [], [], color=line_c, alpha=line_a, linewidth=2.0)
        scatters.append(sc)
        lines.append(ln)

    if PBC:
        _draw_generic_box(ax, box_vectors_range[0])

    if color_mode == 'type':
        ax.legend(handles=type_legend_handles.values(), loc='best')

    # Size calculation
    if r is not None:
        fig.canvas.draw()
        data_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        bbox = ax.get_window_extent()
        if bbox and data_range > 0:
            ppt = (bbox.width * 72 / fig.get_dpi()) / data_range
            s_phys = np.clip(np.pi * ((r * ppt) ** 2), 0.01, 50000)
            for sc in scatters: sc._sizes = np.array([s_phys])

    # --- 8. Update ---
    def update(frame_idx):
        coords = all_frames_processed[frame_idx]
        artists = []
        for i, (sc, ln) in enumerate(zip(scatters, lines)):
            chain = coords[i]
            if chain.size > 0:
                sc._offsets3d = (chain[:,0], chain[:,1], chain[:,2])
                
                # [FIX] Update colors for Type mode
                if color_mode == 'type':
                    sc.set_facecolors(bead_colors_list[i])
                    sc.set_edgecolors(bead_colors_list[i])
                
                xs, ys, zs = chain[:,0], chain[:,1], chain[:,2]
                if isring:
                    xs = np.append(xs, xs[0]); ys = np.append(ys, ys[0]); zs = np.append(zs, zs[0])
                ln.set_data(xs, ys)
                ln.set_3d_properties(zs)
            else:
                sc._offsets3d = ([],[],[])
                ln.set_data([],[])
                ln.set_3d_properties([])
            artists.extend([sc, ln])
        ax.set_title(f"Frame {start_frame + frame_idx} ({color_mode})")
        return artists

    print(f"Generating animation...")
    anim = FuncAnimation(fig, update, frames=num_anim_frames, interval=1000/fps, blit=False)

    if outputName:
        writer = FFMpegWriter(fps=fps) if outputName.endswith('.mp4') else PillowWriter(fps=fps)
        anim.save(outputName, writer=writer, dpi=150)
        print(f"Saved to {outputName}")
        plt.close(fig)
    else:
        try:
            from IPython.display import display
            display(fig)
        except:
            plt.show()
        return anim



def compute_RG_type(traj):
    """
    Function to compute Radius of Gyration (Rg) classified by particle type.
    
    Parameters:
        traj: Trajectory object, which must contain the following attributes:
              - traj.xyz: Coordinate array with shape (T, N, 3)
              - traj.ChromSeq: A list or array of length N, containing bead types (e.g., 'A', 'B')
              
    Returns:
        results (dict): A dictionary containing Rg data.
                        key: 'general' and each type name (e.g., 'A', 'B')
                        value: Corresponding Rg numpy array (of length T)
    """
    # 1. Get coordinates and sequence
    # Ensure xyz is numpy array
    all_positions = np.asarray(traj.xyz(frames=[0, None, 1], beadSelection=None))
    # Ensure ChromSeq is numpy array for boolean indexing
    bead_types = np.asarray(traj.ChromSeq)
    
    # 2. Initialize result dictionary
    results = {}
    
    # 3. Calculate 'general' Rg (all beads)
    # Always calculate this
    results['general'] = compute_RG(all_positions)
    
    # 4. Check if system is Homogeneous (Homogeneous)
    # np.unique returns sorted unique type list
    unique_types = np.unique(bead_types)
    
    # If type count is greater than 1, it's a Heterogeneous (Heterogeneous) system, need to calculate by type
    if len(unique_types) > 1:
        for t_type in unique_types:
            # Create boolean mask (Boolean Mask)
            # mask length equals beads count, True means current type
            mask = (bead_types == t_type)
            
            # Slice
            # Dimension meaning: [all frames, mask filtered column(beads), all coordinates]
            subset_positions = all_positions[:, mask, :]
            
            # Ensure type name is string format as key
            key_name = str(t_type)
            
            # Calculate Rg for this type and store in dictionary
            results[key_name] = compute_RG(subset_positions)
            
    # If len(unique_types) == 1, loop is skipped, only returns general, avoid duplicate calculation
            
    return results