from importlib.metadata import PackageNotFoundError, version

from fingerling.__main__ import read_quants_bin

try:
    if isinstance(__package__, str):
        __version__ = version(__package__)
    else:
        __version__ = "unknown"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["read_quants_bin"]
