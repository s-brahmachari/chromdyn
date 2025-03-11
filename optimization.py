import numpy as np
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
from analyzers import HiCManager
from ChromatinDynamics import ChromatinDynamics

class EnergyLandscapeOptimizer:
    def __init__(self):
        
        pass
    
    def loadHiC(self,hicmap, optimizer='hicinv'):
        hicman = HiCManager(hicmap)
        
        if optimizer.lower()=="hicinv":
            self.optimizer = HiCInversion(hicman.get_hic_map())
        
        self.topology = hicman.create_topology()
        self.hicman = hicman
    
    def simulation_setup(self, topology):
        
        sim = ChromatinDynamics(topology, integrator='langevin', platform_name="CPU", output_dir=f"output")
        
        # Setup system with harmonic trap
        sim.system_setup(mode='default')
            
        # Setup simulation (platform, integrator, positions)
        sim.simulation_setup()
        self.sim = sim
        
    def run_block(self,):
        self.sim.run(10000)
        self.sim.analyzer.print_force_info()
        # Compute Radius of Gyration (Rg)
        Rg = self.sim.analyzer.compute_RG()
        
    
class HiCInversion:
    """
    A generic optimizer to update interaction matrix (lambda or force field) 
    based on comparison between experimental and simulated contact matrices.
    """

    def __init__(self, phi_exp: np.ndarray, method: str = "adam",
                 eta: float = 0.01, beta1: float = 0.7, beta2: float = 0.99999,
                 epsilon: float = 1e-8, it: int = 1, regularization: float = 0.0):
        """
        Initialize optimizer with experimental Hi-C data and optimization parameters.
        
        Args:
            phi_exp (np.ndarray): Experimental contact matrix.
            method (str): Optimization method ('adam', 'rmsprop', 'adagrad', 'sgd').
            eta (float): Learning rate.
            beta1 (float): Beta1 parameter for Adam/Nadam.
            beta2 (float): Beta2 parameter for Adam/Nadam.
            epsilon (float): Small value to avoid division by zero.
            it (int): Initial iteration number.
            regularization (float): Threshold for PCA-based regularization (default 0 = no regularization).
        """
        self.phi_exp = phi_exp
        self.method = method.lower()
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = it
        self.regularize = regularization

        # Optimization tracking params (for Adam, RMSProp, etc.)
        shape = phi_exp.shape
        self.opt_params = {}
        if self.method in {"adam", "nadam"}:
            self.opt_params["m_dw"] = np.zeros(shape)
            self.opt_params["v_dw"] = np.zeros(shape)
        elif self.method == "rmsprop":
            self.opt_params["v_dw"] = np.zeros(shape)
        elif self.method == "adagrad":
            self.opt_params["G_dw"] = np.zeros(shape)

        self.mask = phi_exp == 0.0  # Mask from experimental data

    def compute_gradient(self, phi_sim: np.ndarray) -> np.ndarray:
        """
        Computes gradient of error between experimental and simulated matrices.
        
        Args:
            phi_sim (np.ndarray): Simulated contact matrix.
            
        Returns:
            np.ndarray: Gradient of loss function.
        """
        gt = self.phi_exp - phi_sim
        np.fill_diagonal(gt, 0.0)  # Remove diagonal
        gt -= np.diagflat(np.diag(gt, k=1), k=1)  # Remove immediate neighbors if needed
        gt = np.triu(gt) + np.triu(gt).T  # Symmetrize

        # Optional PCA regularization
        if self.regularize > 0.0:
            logging.info(f"[INFO] Performing PCA regularization with cutoff {self.regularize}")
            eig_vals, eig_vecs = np.linalg.eigh(gt)
            max_eig = eig_vals[-1]
            for idx, eig in enumerate(eig_vals):
                if abs(eig / max_eig) < self.regularize:
                    gt -= eig * np.outer(eig_vecs[:, idx], eig_vecs[:, idx])
                    logging.info(f"[INFO] Removed eigen component {idx} with eigenvalue {eig:.4e}")
        
        return gt

    def update_lambdas(self, current_lambdas: np.ndarray, phi_sim: np.ndarray) -> np.ndarray:
        """
        Updates interaction matrix (lambda/force field) using gradient descent method.
        
        Args:
            current_lambdas (np.ndarray): Current lambda matrix.
            phi_sim (np.ndarray): Simulated contact matrix corresponding to current lambda.
        
        Returns:
            np.ndarray: Updated lambda matrix.
        """
        grad = self.compute_gradient(phi_sim)

        if self.method in {"adam", "nadam"}:
            self.opt_params["m_dw"] = self.beta1 * self.opt_params["m_dw"] + (1 - self.beta1) * grad
            self.opt_params["v_dw"] = self.beta2 * self.opt_params["v_dw"] + (1 - self.beta2) * (grad ** 2)

            m_dw_corr = self.opt_params["m_dw"] / (1 - self.beta1 ** self.t)
            v_dw_corr = self.opt_params["v_dw"] / (1 - self.beta2 ** self.t)

            if self.method == "nadam":
                lookahead = (1 - self.beta1) * grad / (1 - self.beta1 ** self.t)
                m_dw_corr += lookahead

            updated_lambda = current_lambdas - self.eta * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)

        elif self.method == "rmsprop":
            self.opt_params["v_dw"] = self.beta1 * self.opt_params["v_dw"] + (1 - self.beta1) * (grad ** 2)
            updated_lambda = current_lambdas - self.eta * grad / (np.sqrt(self.opt_params["v_dw"]) + self.epsilon)

        elif self.method == "adagrad":
            self.opt_params["G_dw"] += grad ** 2
            updated_lambda = current_lambdas - self.eta * grad / (np.sqrt(self.opt_params["G_dw"]) + self.epsilon)

        elif self.method == "sgd":
            updated_lambda = current_lambdas - self.eta * grad

        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        self.t += 1  # Increment iteration count
        return updated_lambda

    def compute_error(self, phi_sim: np.ndarray) -> float:
        """
        Computes normalized error between experimental and simulated contact maps.
        
        Args:
            phi_sim (np.ndarray): Simulated contact matrix.
        
        Returns:
            float: Normalized L1 error.
        """
        phi_sim[self.mask] = 0.0  # Apply mask
        error = np.sum(np.abs(np.triu(phi_sim, k=2) - np.triu(self.phi_exp, k=2))) / np.sum(np.triu(self.phi_exp, k=2))
        return error

    def optimize(self, current_lambdas: np.ndarray, phi_sim: np.ndarray):
        """
        Full optimization step: compute gradient, update lambdas, return new lambdas and error.
        
        Args:
            current_lambdas (np.ndarray): Current lambda/interaction matrix.
            phi_sim (np.ndarray): Simulated contact matrix from current lambdas.
        
        Returns:
            Tuple[np.ndarray, float]: (Updated lambdas, error value)
        """
        updated_lambda = self.update_lambdas(current_lambdas, phi_sim)
        error = self.compute_error(phi_sim)
        logging.info(f"[INFO] Iteration {self.t} | Error: {error:.5f}")
        return updated_lambda, error