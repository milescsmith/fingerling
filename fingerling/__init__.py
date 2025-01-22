from importlib.metadata import PackageNotFoundError, version

from fingerling.__main__ import read_quants_bin

try:
    __version__ = version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["read_quants_bin"]
