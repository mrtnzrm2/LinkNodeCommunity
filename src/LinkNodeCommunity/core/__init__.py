"""Core submodules for LinkNodeCommunity.

Exposes the main workflow building blocks while keeping imports lazy so that
interactive users can rely on tab completion without incurring heavy startup
costs. External code may either import the submodules explicitly or access the
re-exported symbols listed in ``__all__``.
"""

from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    "framework": "LinkNodeCommunity.core.framework",
    "nocs": "LinkNodeCommunity.core.nocs",
    "similarity": "LinkNodeCommunity.core.similarity",
    "tonewick": "LinkNodeCommunity.core.tonewick",
    "Clustering": "LinkNodeCommunity.core.framework",
    "NOCFinder": "LinkNodeCommunity.core.nocs",
    "LinkSimilarity": "LinkNodeCommunity.core.similarity",
    "ACCEPTED_SIMILARITY_INDICES": "LinkNodeCommunity.core.similarity",
    "LinkageToNewick": "LinkNodeCommunity.core.tonewick",
}

__all__ = [
    "framework",
    "nocs",
    "similarity",
    "tonewick",
    "Clustering",
    "NOCFinder",
    "LinkSimilarity",
    "ACCEPTED_SIMILARITY_INDICES",
    "LinkageToNewick",
]


def __getattr__(name):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        attr = getattr(module, name, module)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'LinkNodeCommunity.core' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover
    from LinkNodeCommunity.core.framework import Clustering
    from LinkNodeCommunity.core.nocs import NOCFinder
    from LinkNodeCommunity.core.similarity import (
        LinkSimilarity,
        ACCEPTED_SIMILARITY_INDICES,
    )
    from LinkNodeCommunity.core import framework, nocs, similarity, tonewick
