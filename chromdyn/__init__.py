"""
chromdyn: Tools for chromosome dynamics modeling and analysis.
"""

# Expose the main simulation interface
from .ChromatinDynamics import ChromatinDynamics

# Optionally expose key components at the package level
from .Platforms import PlatformManager
from .Integrators import IntegratorManager
from .Forcefield import ForceFieldManager

# Utility functions and logging
from .Utilities import config_generator, LogManager

# Reporters
from .Reporters import SaveStructure, StabilityReporter, EnergyReporter

# Version
try:
    from ._version import version as __version__
except ImportError:
    # Fallback if setuptools-scm hasn't written _version.py yet
    __version__ = "0.0.0"