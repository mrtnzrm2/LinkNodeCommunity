"""
Public API for the LinkNodeCommunity package.

Exposes high-level entry points used by users. During the ongoing
package re-organization, this file supports both the current layout
(modules/ at repo root) and the target layout (modules/ inside the
package under src/LinkNodeCommunity/modules).
"""

from _version import get_version
__version__ = get_version()

# Re-export key API from hierarmerge.
# Prefer the in-package path (after moving modules/ under src/LinkNodeCommunity),
# but fall back to the current top-level layout for compatibility during transition.
try:
    from .core.framework import Clustering  # type: ignore
except Exception:  # pragma: no cover - transitional import
    from src.linknode.core.framework import Clustering  # type: ignore

__all__ = ["Clustering"]
