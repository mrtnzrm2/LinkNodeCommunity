"""LinkNodeCommunity public API.

The top-level package only exposes lightweight metadata and defers importing
the heavier clustering utilities until they are explicitly requested. Downstream
code can either import from the specific submodules (preferred) or continue to
rely on the legacy symbols that are lazily re-exported here.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("LinkNodeCommunity")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

_LAZY_EXPORTS = {
    "Clustering": "LinkNodeCommunity.core.framework",
    "NOCFinder": "LinkNodeCommunity.core.nocs",
    "LinkageToNewick": "LinkNodeCommunity.core.tonewick",
    "LinkSimilarity": "LinkNodeCommunity.core.similarity",
    "ACCEPTED_SIMILARITY_INDICES": "LinkNodeCommunity.core.similarity",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name):
    """Lazy-load exported symbols to keep the top-level import lightweight."""

    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'LinkNodeCommunity' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover
    from LinkNodeCommunity.core.framework import Clustering
    from LinkNodeCommunity.core.nocs import NOCFinder
    from LinkNodeCommunity.core.tonewick import LinkageToNewick
    from LinkNodeCommunity.core.similarity import (
        LinkSimilarity,
        ACCEPTED_SIMILARITY_INDICES,
    )
