"""
Path: src/LinkNodeCommunity/__init__.py

Module: LinkNodeCommunity
Author: Jorge S. Martinez Armas

Overview:
---------
Public API for the LinkNodeCommunity package. Re-exports the clustering
workflow, overlap finder, and Newick utilities while exposing the package
version.

Key Components:
---------------
- Clustering
- NOCFinder
- LinkageToNewick
- LinkSimilarity

Notes:
------
- `__version__` resolves via importlib metadata and falls back to "0+unknown"
  when the distribution metadata is unavailable.
"""


try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("LinkNodeCommunity")
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

# Re-export key API from hierarmerge.
# Prefer the in-package path (after moving modules/ under src/LinkNodeCommunity),
# but fall back to the current top-level layout for compatibility during transition.
from .core.framework import Clustering  # type: ignore

from .core.nocs import NOCFinder  # type: ignore

from .core.tonewick import LinkageToNewick  # type: ignore

from .core.similarity import LinkSimilarity, ACCEPTED_SIMILARITY_INDICES  # type: ignore

__all__ = [
    "Clustering", 
    "NOCFinder", 
    "LinkageToNewick",
    "LinkSimilarity", 
    "ACCEPTED_SIMILARITY_INDICES",
    "__version__"
]