"""KG4VD package metadata."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("kg4vd")
except PackageNotFoundError:  # editable / source-tree
    __version__ = "1.0.0"

__all__ = ["__version__"]
