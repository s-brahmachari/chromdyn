#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

from openmm import Platform
from .utilities import LogManager


# -------------------------------------------------------------------
# Platform Manager: Selects GPU/CPU platform and lists available platforms
# -------------------------------------------------------------------
class PlatformManager:
    def __init__(self, logger=None):
        """
        Initialize the PlatformManager.

        Args:
            platform_name (str): Name of the platform to use (e.g., "CUDA", "OpenCL", "CPU").
            logger (Logger, optional): Logger instance to use for logging.
        """
        self.logger = logger or LogManager().get_logger(__name__)
        self.available_platforms = self._get_available_platforms()

    def set_platform(self, platform_name="CUDA"):
        self.platform_name = platform_name
        # self.logger.info("-"*60)
        self._validate_platform()
        # self.logger.info("-"*60)

    def _get_available_platforms(self):
        """
        Get a list of available OpenMM platform names.

        Returns:
            list of str: Names of available platforms.
        """
        num_platforms = Platform.getNumPlatforms()
        platforms = [Platform.getPlatform(i).getName() for i in range(num_platforms)]
        return platforms

    def _validate_platform(self):
        """
        Validate whether the requested platform is available. Logs a warning if not found.
        """
        if self.platform_name not in self.available_platforms:

            self.logger.warning(
                f"Requested platform '{self.platform_name}' not found. "
                f"Available platforms: {', '.join(self.available_platforms)}. "
                "Defaulting to CPU."
            )
            self.platform_name = (
                "CPU"  # Default to CPU if requested platform is not found
            )
        else:
            self.logger.info(
                f"Platform '{self.platform_name}' is available and selected."
            )

    def get_platform(self):
        """
        Get the OpenMM platform object for the selected platform name.

        Returns:
            openmm.Platform: Platform object.
        """
        return Platform.getPlatformByName(self.platform_name)

    def list_platforms(self):
        """
        List available OpenMM platforms and their estimated speed.
        """
        self.logger.info("-" * 50)
        self.logger.info(
            f"Number of available OpenMM platforms: {len(self.available_platforms)}"
        )
        header = f"{'Index':<8} {'Platform Name':<20} {'Speed (estimated)':<20}"
        self.logger.info(header)
        self.logger.info("-" * len(header))

        for i, platform_name in enumerate(self.available_platforms):
            platform = Platform.getPlatform(i)
            speed = platform.getSpeed()
            line = f"{i:<8} {platform_name:<20} {speed:<20}"
            self.logger.info(line)
        self.logger.info("-" * 50)
