from importlib.metadata import PackageNotFoundError, metadata, version

from .__main__ import read_quants_bin

try:
    __version__ = version(__name__)
    __author__ = metadata(__name__)["author"]
    __email__ = metadata(__name__)["email"]
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["read_quants_bin"]