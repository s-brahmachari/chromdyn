import logging
import os

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

    def get_logger(self, name):
        """
        Get a logger instance with console and optional file handler.

        Args:
            name (str): Module-specific logger name.

        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        while logger.hasHandlers():
            logger.handlers.clear()
            
        # Avoid adding multiple handlers if logger already has handlers
        if not logger.handlers:
            formatter = logging.Formatter(self.log_format)
        
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s'))
            logger.addHandler(console_handler)

            # Optional file handler
            if self.log_file:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                file_handler = logging.FileHandler(self.log_file, mode='w')
                file_handler.setLevel(logging.DEBUG)  # Log everything into file, including DEBUG
                file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
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