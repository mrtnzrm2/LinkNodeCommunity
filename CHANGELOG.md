## [Unreleased]

## [v0.2.1] - 09-28-2025
- Reworked `pyproject.toml` metadata provider configuration so scikit-build-core pulls the exact tag version from `setuptools_scm`.
- Enabled the scikit-build experimental metadata provider hook to unblock wheel and sdist builds for TestPyPI.

## [v0.2.0] - 09-28-2025
- Added project logo.
- Implemented lazy loading for core classes to speed top-level imports.

## [v0.1.0] - 09-27-2025
- Initial publication of LinkNodeCommunity with link-based community detection pipeline.
- Core similarity, Tonewick, and NOCS modules for computing hierarchical link communities.
- Sample C++ backends and Python bindings for performance-critical routines.
- Documentation scaffold (README, notebooks) and pytest suite covering clustering and metrics.