# chromdyn/__init__.py
try:
    # written by setuptools-scm at build time
    from ._version import version as __version__
except Exception:
    __version__ = "0+unknown"

__all__ = ["__version__"]