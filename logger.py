import logging

class LoggerManager:
    def __init__(self, log_level=logging.INFO, log_format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"):
        """
        Initialize LoggerManager with log level and format.
        """
        self.log_level = log_level
        self.log_format = log_format
        self._setup_logger()

    def _setup_logger(self):
        """
        Set up the basic configuration for logging.
        """
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format,
            handlers=[
                logging.StreamHandler()  # Optionally add FileHandler here
            ]
        )

    def get_logger(self, name):
        """
        Get a logger with a module-specific name.
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
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
