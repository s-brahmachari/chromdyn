import numpy as np

class EnergyLandscapeOptimizer:
    
    def __init__(self, method: str = "adam",
                 eta: float = 0.01, beta1: float = 0.7, beta2: float = 0.9999,
                 epsilon: float = 1e-8, it: int = 1,
                 scheduler: str = "none", scheduler_decay: float = 0.1, scheduler_step: int = 100,
                 scheduler_eta_min: float = 0.001, scheduler_T_max: int = 120):
        """
        Initializes the Energy Landscape Optimizer with given hyperparameters and learning rate scheduler.

        Parameters:
            mu, rc: Model hyperparameters.
            method: Optimizer method, e.g., "adam", "nadam", "rmsprop", "adagrad", "sgd".
            eta: Initial learning rate.
            beta1, beta2, epsilon: Adam-related hyperparameters.
            it: Initial iteration counter.
            error_pca_weight: Weight for regularizing error using PCA.
            scheduler: Type of learning rate scheduler ("none", "step", "cosine", or "exponential").
            scheduler_decay: Decay factor (used as lambda for exponential decay, or decay factor for step decay).
            scheduler_step: Number of iterations per step decay.
            scheduler_eta_min: Minimum learning rate for the cosine scheduler.
            scheduler_T_max: Total iterations for cosine annealing.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta0 = eta  # Store the initial learning rate
        self.eta = eta   # Current learning rate
        self.t = int(it)
        self.method = method.lower()
        self.opt_params = {}
        self.phi_exp = None  # Experimental Hi-C data
        self.force_field = None
        self.updated_force_field = None
        
        # Learning rate scheduler parameters
        self.scheduler = scheduler.lower()  # "none", "step", "cosine", or "exponential"
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.scheduler_eta_min = scheduler_eta_min
        self.scheduler_T_max = scheduler_T_max

    def update_learning_rate(self) -> None:
        """Updates the learning rate based on the selected scheduler."""
        if self.scheduler == "step":
            # Step decay: Every scheduler_step iterations, multiply by scheduler_decay.
            factor = self.scheduler_decay ** (self.t // self.scheduler_step)
            self.eta = self.eta0 * factor
        elif self.scheduler == "cosine":
            # Cosine annealing: Gradually decay the learning rate with a cosine schedule.
            self.eta = self.scheduler_eta_min + 0.5 * (self.eta0 - self.scheduler_eta_min) * \
                       (1 + np.cos(np.pi * self.t / self.scheduler_T_max))
        elif self.scheduler == "exponential":
            # Exponential decay: Learning rate decays as: eta_t = eta0 * exp(-lambda * t)
            self.eta = self.eta0 * np.exp(-self.scheduler_decay * self.t)
        # If scheduler is "none", self.eta remains unchanged.

    def load_HiC(self, hic_file: str, cutoff_low: float = 0.0, cutoff_high: float = 1.0, neighbors: int = 0) -> None:
        """
        Loads the Hi-C matrix from a text file, applies cutoffs, and initializes optimization parameters.
        """
        if not hic_file.endswith('.txt'):
            raise ValueError("Input Hi-C file should be a TXT file that can be handled by np.loadtxt.")

        hic_mat = np.loadtxt(hic_file)
        
        if not self.is_symmetric(hic_mat):
            raise ValueError("Experimental HiC input is NOT symmetric.")
        
        # Apply cutoffs to remove noise
        # hic_mat = np.clip(hic_mat, a_min=cutoff_low, a_max=cutoff_high)
        
        # Remove neighbor interactions within the given range
        neighbor_mask = np.abs(np.subtract.outer(np.arange(len(hic_mat)), np.arange(len(hic_mat)))) <= neighbors
        hic_mat[neighbor_mask] = 0.0
        hic_mat[hic_mat>cutoff_high]=0.0
        hic_mat[hic_mat<cutoff_low]=0.0
        self.phi_exp = hic_mat
        self.mask = (hic_mat != 0.0)
        self.init_optimization_params()

    @staticmethod
    def is_symmetric(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks if a matrix is symmetric."""
        return np.allclose(mat, mat.T, rtol=rtol, atol=atol)

    def init_optimization_params(self) -> None:
        """Initializes optimization parameters for different optimizers."""
        self.opt_params.clear()
        shape = self.phi_exp.shape

        if self.method in {"adam", "nadam"}:
            self.opt_params["m_dw"] = np.zeros(shape)
            self.opt_params["v_dw"] = np.zeros(shape)
        elif self.method == "rmsprop":
            self.opt_params["v_dw"] = np.zeros(shape)
        elif self.method == "adagrad":
            self.opt_params["G_dw"] = np.zeros(shape)

    def update_step(self, grad: np.ndarray, lambda_t: np.ndarray) -> np.ndarray:
        """Performs an optimization step based on the selected method, updating the learning rate if a scheduler is used."""
        # Update the learning rate based on the scheduler
        self.update_learning_rate()
        
        if self.method in {"adam", "nadam"}:
            self.opt_params["m_dw"] *= self.beta1
            self.opt_params["m_dw"] += (1 - self.beta1) * grad
            self.opt_params["v_dw"] *= self.beta2
            self.opt_params["v_dw"] += (1 - self.beta2) * (grad ** 2)

            m_dw_corr = self.opt_params["m_dw"] / (1 - self.beta1 ** self.t)
            v_dw_corr = self.opt_params["v_dw"] / (1 - self.beta2 ** self.t)

            if self.method == "nadam":
                lookahead_gradient = (1 - self.beta1) * grad / (1 - self.beta1 ** self.t)
                m_dw_corr += lookahead_gradient

            w = lambda_t - self.eta * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)

        elif self.method == "rmsprop":
            self.opt_params["v_dw"] *= self.beta1
            self.opt_params["v_dw"] += (1 - self.beta1) * (grad ** 2)
            w = lambda_t - self.eta * grad / (np.sqrt(self.opt_params["v_dw"]) + self.epsilon)

        elif self.method == "adagrad":
            self.opt_params["G_dw"] += grad ** 2
            w = lambda_t - self.eta * grad / (np.sqrt(self.opt_params["G_dw"]) + self.epsilon)
        
        elif self.method == "sgd":
            w = lambda_t - self.eta * grad

        self.t += 1
        return w

    def get_error_gradient(self, phi_sim: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the optimization objective."""
        gt = self.phi_exp - phi_sim
        np.fill_diagonal(gt, 0.0)
        # gt -= np.diagflat(np.diag(gt, k=1), k=1)
        gt = np.triu(gt) + np.triu(gt).T  # Ensure symmetry
        return gt

    # def compute_force_field(self, ff_current: str) -> pd.DataFrame:
    #     """Computes and updates the force field from the given file."""
    #     if self.Pi is None or self.NFrames == 0:
    #         raise ValueError("Contact probability matrix not initialized. Call compute_contact_prob before force field computation.")

    #     self.phi_sim = self.Pi / self.NFrames
    #     self.phi_sim[self.mask] = 0.0  # Apply the mask to filter out noise

    #     df = pd.read_csv(ff_current, sep=None, engine='python')
    #     current_force_field = df.values
    #     self.force_field = current_force_field
    #     grad = self.get_error_gradient()
    #     self.updated_force_field = self.update_step(grad)

    #     df_updated_ff = pd.DataFrame(self.updated_force_field, columns=list(df.columns.values))
    #     self.error = np.sum(np.abs(np.triu(self.phi_sim, k=2) - np.triu(self.phi_exp, k=2))) / np.sum(np.triu(self.phi_exp, k=2))
    #     return df_updated_ff
    
    def get_updated_params(self, lambda_t: np.ndarray, phi_sim: np.ndarray) -> np.ndarray:
        """Computes and updates the force field from the given file."""
        phi_sim *= self.mask
        grad = self.get_error_gradient(phi_sim)
        updated_lambda = self.update_step(grad, lambda_t)
        self.error = np.sum(np.abs(np.triu(phi_sim, k=2) - np.triu(self.phi_exp, k=2))) / np.sum(np.triu(self.phi_exp, k=2))
        return updated_lambda