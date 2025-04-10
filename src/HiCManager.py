import numpy as np
import os
from logger import LogManager
from scipy.ndimage import median_filter, uniform_filter
from sklearn.preprocessing import normalize

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
    
