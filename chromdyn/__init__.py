#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *
"""
chromdyn: Tools for chromosome dynamics modeling and analysis.
"""

from importlib import metadata

# Expose the main simulation interface
from .chromatin_dynamics import ChromatinDynamics

# Optionally expose key components at the package level
from .platforms import PlatformManager
from .integrators import IntegratorManager
from .forcefield import ForceFieldManager
from .topology import TopologyGenerator
from .traj_utils import TrajectoryLoader, Analyzer, save_pdb
from .hic_utils import HiCManager

# Utility functions and logging
from .utilities import config_generator, LogManager

# Reporters
from .reporters import SaveStructure, StabilityReporter, EnergyReporter

__all__ = [
    "ChromatinDynamics",
    "PlatformManager",
    "IntegratorManager",
    "ForceFieldManager",
    "config_generator",
    "LogManager",
    "SaveStructure",
    "StabilityReporter",
    "EnergyReporter",
    "TopologyGenerator",
    "TrajectoryLoader",
    "Analyzer",
    "HiCManager",
    "save_pdb",
]


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"
