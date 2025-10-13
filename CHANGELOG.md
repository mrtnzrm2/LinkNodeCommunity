## [Unreleased]

## [v0.2.7] - 10-13-2025
- Added the utils_cpp in the __init__.py script.
- Added weight parameter in the Clustering class to select the weight attribute to be used for computing node neighborhood similarities.
- Added tests to validate the weight attribute label used in the Clustering class.
- Relabeled Clustering member linkdist_matrix to linkdist_condensed_matrix.
- Fixed bug in equivalence_partition method.

## [v0.2.6] - 09-28-2025
- Reworked `pyproject.toml` metadata provider configuration so scikit-build-core pulls the exact tag version from `setuptools_scm`.
- Enabled the scikit-build experimental metadata provider hook to unblock wheel and sdist builds for TestPyPI.
- Removed `cmake` and `ninja` from the build requirements now that scikit-build-core injects them automatically.
- Updated the top-level `CMakeLists.txt` to adopt policy CMP0148 and use `FindPython`, removing the pybind11 warning during builds.
- Switched the README logo to an absolute raw GitHub URL so the image renders on TestPyPI.

## [v0.2.0] - 09-28-2025
- Added project logo.
- Implemented lazy loading for core classes to speed top-level imports.

## [v0.1.0] - 09-27-2025
- Initial publication of LinkNodeCommunity with link-based community detection pipeline.
- Core similarity, Tonewick, and NOCS modules for computing hierarchical link communities.
- Sample C++ backends and Python bindings for performance-critical routines.
- Documentation scaffold (README, notebooks) and pytest suite covering clustering and metrics.
