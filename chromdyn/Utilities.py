import numpy as np
import random
import os
import logging

class LogManager:
    def __init__(self, log_level=logging.INFO,
                 log_format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
                 log_file=None):
        """
        Initialize LoggerManager with log level, format, and optional file logging.

        Args:
            log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
            log_format (str): Logging format.
            log_file (str, optional): Path to log file. If None, logs only to console.
        """
        self.log_level = log_level
        self.log_format = log_format
        self.log_file = log_file

    def get_logger(self, name, console=True):
        """
        Get a logger instance with console and optional file handler.

        Args:
            name (str): Module-specific logger name.

        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        while len(logger.handlers)>0:
            logger.handlers.clear()
            
        # Avoid adding multiple handlers if logger already has handlers
        if not logger.handlers:
            # Console handler
            if console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s'))
                logger.addHandler(console_handler)

            # Optional file handler
            if self.log_file:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                file_handler = logging.FileHandler(self.log_file, mode='w')
                file_handler.setLevel(logging.DEBUG)  # Log everything into file, including DEBUG
                file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

        return logger

    @staticmethod
    def log_header(logger, title, width=100, char="="):
        """
        Prints a formatted header in logs for better readability.

        Args:
            logger (logging.Logger): Logger instance to use.
            title (str): Header title to display.
            width (int): Total width of the header (default 100).
            char (str): Character to use for border lines (default '=').
        """
        logger.info(char * width)
        logger.info(f"{title:^{width}}")  # Center the title
        logger.info(char * width)
        
def compute_RG(positions):
    center_of_mass = np.mean(positions, axis=0)
    squared_distances = np.sum((positions - center_of_mass) ** 2, axis=1)
    Rg = np.sqrt(np.mean(squared_distances))
    return Rg

def gen_structure(mode, num_steps, **kwargs):
    step_size = kwargs.get('step_size', 1)
    max_restarts = kwargs.get('max_restarts', 5000)
    logger = kwargs.get('logger', LogManager().get_logger(__name__))
    
    if mode.lower()=='saw3d':
        try:
            path, msg = get_pos_3Dsaw(num_steps, step_size, max_restarts)
            logger.info(msg)
        except RuntimeError as e:
            path, msg = get_pos_3Drandom_walk(num_steps, step_size)
            msg = f"3D SAW failed! Returning a random walk instead. Error: {e}"
            logger.warning(msg)
        return path
    elif mode.lower()=='randomwalk':
        # Generate a random walk
        path, msg = get_pos_3Drandom_walk(num_steps, step_size)
        logger.info(msg)
        return path
    else:
        path, _ = get_pos_3Drandom_walk(num_steps, step_size)
        logger.warning(f"Invalid mode '{mode}'. Defaulting to random walk. Position shape: {path.shape}.")
        return path
    
def get_pos_3Dsaw(num_steps, step_size, max_restarts=5000):
        """
        Generate a 3D self-avoiding walk on a cubic lattice with restarts.
        
        Args:
            num_steps (int): Desired length of the walk.
            max_restarts (int): Maximum number of full retries if stuck.
            verbose (bool): Print progress info.

        Returns:
            path (np.ndarray): Array of shape (num_steps, 3) with 3D walk positions.
        """
        directions = [
            np.array([step_size, 0, 0]), np.array([-step_size, 0, 0]),
            np.array([0, step_size, 0]), np.array([0, -step_size, 0]),
            np.array([0, 0, step_size]), np.array([0, 0, -step_size]),
        ]

        for attempt in range(1, max_restarts + 1):
            position = np.array([0, 0, 0])
            visited = set()
            visited.add(tuple(position))
            path = [position.copy()]
            success = True

            for step in range(1, num_steps):
                # List all valid next positions
                valid_moves = []
                for d in directions:
                    candidate = tuple(position + d)
                    if candidate not in visited:
                        valid_moves.append(d)

                if not valid_moves:
                    success = False
                    # if verbose:
                    #     self.logger.info(f"[Attempt {attempt}] Stuck at step {step}, restarting...")
                    break  # restart the whole walk

                move = random.choice(valid_moves)
                position += move
                visited.add(tuple(position))
                path.append(position.copy())

            if success:
                path = np.array(path)
                msg = f"3D SAW created after {attempt} attempt(s). Position shape: {path.shape}"
                return path, msg
        raise RuntimeError(f"Failed to generate a self-avoiding walk after {max_restarts} attempts.")
            
def get_pos_3Drandom_walk(num_steps, step_size):
    # Generate random directions on the unit sphere
    directions = np.random.normal(size=(num_steps, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize
    steps = directions * step_size  # Scale to step size
    positions = np.cumsum(steps, axis=0)  # Cumulative sum for path
    msg = f"Random walk created. Position shape: {positions.shape}"
    return positions, msg  # Shape: (num_steps, 3)
