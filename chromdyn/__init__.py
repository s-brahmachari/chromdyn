"""
chromdyn: Tools for chromosome dynamics modeling and analysis.
"""

# Expose the main simulation interface
from .chromatin_dynamics import ChromatinDynamics

# Optionally expose key components at the package level
from .platforms import PlatformManager
from .integrators import IntegratorManager
from .forcefield import ForceFieldManager
from .topology import TopologyGenerator
from .traj_utils import TrajectoryLoader, Analyzer

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
]
# Version
try:
    from ._version import version as __version__
except ImportError:
    # Fallback if setuptools-scm hasn't written _version.py yet
    __version__ = "0.0.0"
