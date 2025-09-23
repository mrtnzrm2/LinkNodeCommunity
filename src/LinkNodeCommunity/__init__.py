"""
Public API for the LinkNodeCommunity package.

Exposes high-level entry points used by users.
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

from .core.similarity import LinkSimilarity   # type: ignore

__all__ = ["Clustering", "NOCFinder", "LinkageToNewick", "LinkSimilarity", "__version__"]
