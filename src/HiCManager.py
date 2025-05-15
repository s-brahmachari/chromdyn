import numpy as np
import os
from Utilities import LogManager
from scipy.ndimage import median_filter, uniform_filter
# from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
import h5py

class HiCManager:
    def __init__(self, logger=None,):
        """
        HiCManager to handle Hi-C contact maps from file, array, or generate random.
        
        Args:
            hicmap (str, np.ndarray, int): Path to .txt file, 2D NumPy array, or integer to generate.
            random_max_value (int): Max value for random Hi-C generation (default: 10).
        """
        self.logger = logger or LogManager().get_logger(__name__)
        
    def _remove_neighbor(self, hicmap, k):
        if isinstance(k, int):
            res = hicmap - np.diagflat(np.diag(hicmap, k=k), k=k)
        
        elif isinstance(k, list):
            for idx in k:
                res = hicmap - np.diagflat(np.diag(hicmap, k=idx), k=idx)
        elif isinstance(k, float):
            self.logger.error("k should be an int or list of ints!")
        
        res = np.triu(res) + np.triu(res,k=1).T
        return res

    def filter_matrix(self, matrix, window_size=3, method='median', padding='wrap'):
        """
        Apply a sliding window filter (median or mean) to a matrix, keeping the same dimensions.

        Args:
            matrix (np.ndarray): Input 2D matrix (array).
            window_size (int or tuple): Size of the filtering window (e.g., 3 or (3, 3)).
            method (str): 'median' or 'mean' (average).
            padding (str): Padding mode for boundaries ('reflect', 'constant', 'nearest', 'mirror', 'wrap').

        Returns:
            np.ndarray: Filtered matrix with same dimensions.
        """
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        
        if method == 'median':
            result = median_filter(matrix, size=window_size, mode=padding)
        elif method == 'mean' or method == 'average':
            result = uniform_filter(matrix, size=window_size, mode=padding)
        else:
            raise ValueError("Unsupported method. Choose 'median' or 'mean'.")
        
        return result
    
    def symmetric_l1_normalization(self, A):
        """
        Perform symmetric L1 normalization on a square symmetric matrix.
        
        Args:
            A (numpy.ndarray): Input symmetric matrix (NxN).
        
        Returns:
            numpy.ndarray: Symmetrically normalized matrix.
        """
        if not np.allclose(A, A.T):
            raise ValueError("Input matrix must be symmetric.")
        
        # Compute absolute row and column sums (same since A is symmetric)
        row_sums = np.nansum(np.abs(A), axis=1)  # Sum along rows
        col_sums = np.nansum(np.abs(A), axis=0)  # Sum along columns

        # Compute the normalization factor using sqrt of row and column sums product
        normalization_factors = np.sqrt(np.outer(row_sums, col_sums))

        # Normalize each element
        A_normalized = A / normalization_factors

        return A_normalized
    
    def get_flat_matrix(self, mat):
        """
        Get the averaged Hi-C map.
        Normalize pairwise contact frequencies by the average frequency between similarly distant genomic loci.
        
        Args:
            mat (np.ndarray): Input Hi-C contact matrix.
            tol (float): Tolerance to avoid division by zero, default is 1e-8.
        
        Returns:
            np.ndarray: Averaged Hi-C contact matrix.
        """
        avg_mat = np.zeros(mat.shape)
        
        # Loop through each diagonal of the matrix
        for i in range(mat.shape[0]):
            # Normalize diagonal values by the average frequency and add to avg_mat
            avg_mat += np.diagflat(np.nanmean(np.diag(mat, k=i)) * np.ones(len(np.diag(mat, k=i))), i)

        # Reflect the upper triangle to the lower triangle to make it symmetric
        avg_mat = avg_mat.T + np.triu(avg_mat, 1)

        return avg_mat
    
    def update_kth_neighbor(self,matrix, k, val):
        res = matrix - np.diagflat(np.diag(matrix,k=k),k=k) - np.diagflat(np.diag(matrix,k=-k),k=-k) + np.diagflat(np.ones(len(np.diag(matrix,k=k)))*val,k=k) + np.diagflat(np.ones(len(np.diag(matrix,k=-k)))*val,k=-k) 
        return res
    
    def normalize_by_kth_neighbors(self,matrix, k):
        res = matrix/np.mean(np.diag(matrix,k=k))
        return res
    
    def normalize_hic(self, matrix, p2=0.5):
        res = self.filter_matrix(matrix, window_size=1)
        res = self._remove_neighbor(res, k=[0,1])
        res[0,-1] = 0.0
        res[-1,0] = 0.0
        print('Symmetric:', self.check_symmetric(res))
        res = self.symmetric_l1_normalization(res)
        print('Symmetric:', self.check_symmetric(res))
        # res = self.update_kth_neighbor(res, k=1, val=res.max())
        second_neighbor_counts = np.mean(np.diag(res,k=2))
        res *= p2/second_neighbor_counts
        res = self.update_kth_neighbor(res, k=1, val=1.0)
        res[0,-1] = 1.0
        res[-1,0] = 1.0
        np.fill_diagonal(res, 0.0)
        # res = self.normalize_by_kth_neighbors(res, k=1)
        return res        
            
    def get_numpy_from_hic(self, hicmap, skiprows=0, skipcols=0):
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
                        hic = np.loadtxt(hicmap, skiprows=skiprows, dtype=str)
                        if skipcols==1:
                            hic = np.delete(hic, 0, axis=1)
                        hic = np.array(hic, dtype=float)  
                        self.logger.info(f"Hi-C map loaded from file '{hicmap}' with shape {hic.shape}.")
                        return  hic# Success, exit function
                    else:
                        self.logger.warning(f"File '{hicmap}' found but unsupported format. Expecting .txt.")
                else:
                    self.logger.warning(f"Path '{hicmap}' is not a valid file.")
            except Exception as e:
                self.logger.error(f"Failed to load Hi-C map from file '{hicmap}': {e}")
            
        # If not a valid file or failed, check if it's a NumPy array
        if isinstance(hicmap, np.ndarray):
            try:
                if hicmap.ndim != 2:
                    raise ValueError(f"Provided NumPy array must be 2D. Got shape: {hicmap.shape}")
                hic = hicmap
                self.logger.info(f"Hi-C map loaded from NumPy array with shape {hicmap.shape}.")
                return  hic # Success, exit function
            except Exception as e:
                self.logger.error(f"Failed to use provided NumPy array: {e}")
            
        # If not an array, check if it's an integer for random generation
        if isinstance(hicmap, int):
            try:
                if hicmap <= 0:
                    raise ValueError("Integer for Hi-C map generation must be positive.")
                random_map = np.random.random(0, 1, size=(hicmap, hicmap))
                hic = (random_map + random_map.T) // 2  # Make symmetric
                # np.fill_diagonal(self.hic_map, 0)  # Optional: zero diagonal
                self.logger.info(f"Random symmetric Hi-C map generated (size: {hicmap}x{hicmap}, max value: {self.random_max_value}).")
                return  hic# Success, exit function
            except Exception as e:
                self.logger.error(f"Failed to generate random Hi-C map: {e}")

        # If none of the above succeeded, raise a clear error
        raise TypeError(f"[FATAL ERROR] Unable to interpret input '{hicmap}'. Must be path to '.txt' file, 2D NumPy array, or positive integer.")
        

    def get_hic_map(self):
        """Returns the loaded Hi-C map."""
        return self.hic_map

    def get_shape(self):
        """Returns shape of the Hi-C map."""
        return self.hic_map.shape if self.hic_map is not None else None
    
    def get_Ps(self, hic):
        Ps = []
        Ps_std = []
        for ii in range(hic.shape[0]):
            Ps.append(np.diag(hic,k=ii).mean())
            Ps_std.append(np.diag(hic,k=ii).std())
        # ax.errorbar(list(range(len(Ps))), Ps, yerr=Ps_std, fmt='.')
        return Ps,Ps_std
    
    def condense_matrix(self, matrix):
        N = matrix.shape[0]  # Original matrix size
        if N % 2 != 0:
            # raise ValueError("Matrix size must be divisible by 2")
            matrix = matrix[:-1,:-1]
            N = matrix.shape[0]

        n = N // 2  # New matrix size
        # Reshape the matrix into blocks of 2x2
        reshaped = matrix.reshape(n, 2, n, 2).swapaxes(1, 2)
        # Average over the last two axes to condense the blocks
        condensed_matrix = reshaped.mean(axis=(2, 3))
        
        return condensed_matrix
    
    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def cndb_to_numpy(self, traj_file, skip_frames=1):
        self.logger.info('Loading trajectory ...')
        xyz = []
        with h5py.File(traj_file,'r') as pos:
            frame_ids = []
            for val in pos.keys():
                try:
                    step = int(val)
                    frame_ids.append(step)
                except (ValueError):
                    pass
                    
            for key in sorted(frame_ids):
                xyz.append(pos[str(key)])        
            xyz=np.array(xyz)[::skip_frames]
        self.logger.info(f'Trajectory shape: {xyz.shape}')
        return xyz
    
    @staticmethod
    def _divide_into_subtraj(xyz, num_proc):
        sub_frames=xyz.shape[0]//num_proc
        inputs=[xyz[ii*sub_frames:(ii+1)*sub_frames,:,:] for ii in range(num_proc)]
        return inputs
    
    def gen_hic_from_cndb(self, traj_file, mu, rc, p, parallel=True, skip_frames=1):
        xyz = self.cndb_to_numpy(traj_file, skip_frames=skip_frames)
        serialize=False
        if parallel:
            try:
                import multiprocessing
                                
                # Safer multiprocessing start method (avoid memory duplication)
                multiprocessing.set_start_method("spawn", force=True)

                # Limit processes based on trajectory size
                # num_proc = min(multiprocessing.cpu_count(), 1 + xyz.shape[0] // 10)
                num_proc = min(multiprocessing.cpu_count(), 32, 1 + xyz.shape[0] // 10)

                # Split the trajectory BEFORE parallelizing
                subtraj_list = self._divide_into_subtraj(xyz, num_proc)
                self.logger.info(f"Using multiprocessing. Dividing into {num_proc} processes.")

                
                args_list = [(subtraj, mu, rc, p) for subtraj in subtraj_list]

                # Initialize accumulator
                hic = np.zeros((xyz.shape[1], xyz.shape[1]), dtype=np.float32)

                # Use context manager for clean pool closure
                with multiprocessing.Pool(processes=num_proc) as pool:
                    for partial_result in pool.imap_unordered(_wrap_calc, args_list):
                        hic += partial_result / num_proc  # running average

                self.logger.info(f"Generated HiC matrix of shape: {hic.shape}")
            except ModuleNotFoundError:
                self.logger.warning("Module `multiprocessing` not found.")
                serialize = True
        else:
            serialize=True
        
        if serialize:
            self.logger.info("Computing HiC serially...")
            hic = _calc_HiC_from_traj_array(xyz, mu, rc, p)
            self.logger.info(f"Generated HiC matrix of shape: {hic.shape}")
    
        return hic

def _calc_prob(data, mu, rc, p):
    # Compute condensed distance matrix
    r = pdist(data, metric='euclidean')
    
    # Compute probabilities for condensed distances
    f_condensed = np.where(
        r <= rc,
        0.5 * (1 + np.tanh(mu * (rc - r))),
        0.5 * (rc / r) ** p
    )

    # Convert to square symmetric matrix with zeros on the diagonal
    f = squareform(f_condensed)
    return f

# Function to wrap single-call processing (needed for Pool.map)
def _wrap_calc(subtraj_mu_rc_p):
    subtraj, mu, rc, p = subtraj_mu_rc_p
    return _calc_HiC_from_traj_array(subtraj, mu, rc, p).astype(np.float32)

def _calc_HiC_from_traj_array(traj, mu, rc, p):
    # print('Computing probability of contact versus contour distance')
    pol_size=traj.shape[1]
    Prob = np.zeros((pol_size, pol_size))
    for ii, snapshot in enumerate(traj):
        Prob += _calc_prob(snapshot, mu, rc, p)
    Prob=Prob/(ii+1)
    return Prob

# def _calc_prob(data, mu, rc, p):
#     r = distance.cdist(data, data, 'euclidean')
#     f = np.where(
#         r <= rc,
#         0.5 * (1 + np.tanh(mu * (rc - r))),
#         0.5 * (rc / r)**p
#     )
#     return f