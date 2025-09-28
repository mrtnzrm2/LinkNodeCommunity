<!-- [![LinkNodeCommunity Logo](docs/logo6.svg)](https://github.com/mrtnzrm2/LinkNodeCommunity) -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
  <img alt="LinkNodeCommunity" src="assets/logo-light.svg" width="780">
</picture>

<!-- TODO: add PyPI, build, docs, and coverage badges when available -->

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Key Features](#key-features)
- [Scientific Motivation](#scientific-motivation)
  - [Neuroscience](#neuroscience)
  - [Information Theory](#information-theory)
  - [Network Science](#network-science)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Verifying the Build](#verifying-the-build)
- [Quickstart](#quickstart)
- [API Highlights](#api-highlights)
- [Data \& Notebooks](#data--notebooks)
- [Repository Layout](#repository-layout)
- [Testing \& Quality](#testing--quality)
- [Development Tips](#development-tips)
- [Citation](#citation)
- [Funding \& Acknowledgements](#funding--acknowledgements)
- [License](#license)
- [References](#references)

## Overview
LinkNodeCommunity provides an information-theoretic framework for detecting link and node communities in dense, directed, and weighted networks. Originally developed for mesoscale cortical connectivity analysis, it formalizes a workflow that integrates link similarity, hierarchical clustering, and statistical diagnostics to reveal nested community structure.

## Key Features
- Information-theoretic link similarity based on the Hellinger metric and 1/2-Rényi divergence
- Dual hierarchies: link communities and node communities derived from the same similarity landscape
- Loop entropy scoring to identify a "Goldilocks" resolution for mesoscale organization
- Support for directed or undirected graphs, heterogeneous weights, and optional subgraph conditioning
- Export utilities to export hierarchies as Newick trees for downstream phylogenetic-style visualization
- Benchmarks and diagnostics aligned with neuroscience-grade anatomical datasets and synthetic benchmarks

## Scientific Motivation
### Neuroscience
- Captures the mesoscale organization of macaque cortical networks reconstructed via retrograde tract-tracing.
- Aligns structural partitions with functional hypotheses by identifying link communities that preserve densely interconnected feedback circuits.

### Information Theory
- Measures similarity through the Hellinger distance and the associated 1/2-Rényi divergence, ensuring that community assignments reflect distinguishable connection profiles.
- Uses loop entropy to balance description length against redundancy, surfacing an optimal hierarchy level for information processing.

### Network Science
- Extends the link community paradigm of Ahn, Bagrow, and Lehmann (2010) to weighted, directed, and dense graphs.
- Employs hierarchical clustering routines optimized in C++ (`link_hierarchy_statistics`, `node_community_hierarchy`) to handle thousands of links efficiently.

## Getting Started
### Prerequisites
- Python 3.10+
- A C++17 toolchain compatible with scikit-build-core, CMake (>=3.24), and Ninja
- Recommended Python dependencies: `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `scikit-learn`, `seaborn`
- Optional: R (for `ggtree` visualizations accessed via `rpy2`)

On Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build g++
```
On macOS
```bash
brew install cmake ninja
```
On Windows:
- Install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with C++17 support
- Add CMake and Ninja to your PATH

### Installation
**From PyPI (planned):**
```bash
pip install LinkNodeCommunity
```

**From TestPyPI (pre-release builds):**
```bash
pip install -i https://test.pypi.org/simple/ LinkNodeCommunity
```

**From source (developer mode):**
```bash
# clone your fork or the upstream repository
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```
The editable install (`-e`) links the local source into your environment and installs developer dependencies.
During this process, scikit-build automatically compiles the C++ extensions under `cpp/`.

### Verifying the Build
```python
>>> import LinkNodeCommunity as lnc
>>> lnc.__version__
'0.0.0'

# Lazy exports keep the top-level import fast while providing submodule discovery
>>> from LinkNodeCommunity import Clustering
>>> from LinkNodeCommunity.core import similarity
```

## Quickstart
```python
import networkx as nx
from LinkNodeCommunity import Clustering

# Build a small directed, weighted graph
g = nx.DiGraph()
g.add_weighted_edges_from([
    ("V1", "V2", 0.8),
    ("V2", "V4", 0.6),
    ("V1", "V3", 0.7),
    ("V3", "V4", 0.9),
    ("V4", "V1", 0.5),
])

model = Clustering(g, similarity_index="bhattacharyya_coefficient", edge_complete=True)
model.fit(method="matrix")

# Linkage matrix for downstream cuts or dendrograms
linkage = model.get_hierarchy_matrix()

# Loop entropy-guided partition
partition_dict = model.equivalence_partition(score="S")
print(partition_dict)
```

## API Highlights
- `Clustering`: orchestrates similarity computation, hierarchical linkage, entropy metrics, and convenience exports.
- `NOCFinder`: derives node-overlap communities aligned with link-level partitions.
- `LinkageToNewick`: converts linkage matrices to Newick strings for compatibility with tree visualizers (e.g., `ggtree`).
- `LinkSimilarity` / `ACCEPTED_SIMILARITY_INDICES`: expose alternative similarity indices (Bhattacharyya, cosine, Pearson, weighted Jaccard, Tanimoto, and more) and parallel computation options.
- `utils`: helper functions for edgelist management, equivalence partitions, validity checks, and fast dendrogram cuts.

## Data & Notebooks
- `data/macaque`: curated cortical connectivity matrices (FLN weights, anatomical distances, region labels) used throughout the associated manuscript.
- `notebooks/`: exploratory analyses (e.g., `PaperAnalysis.ipynb`, `HierarchicaScaleFree.ipynb`, `ErdosRenyi.ipynb`) illustrating usage on empirical and synthetic graphs.

## Repository Layout
```
assets/                   # Logos and static resources for documentation
cpp/                      # pybind11 extensions (C++ implementations of hierarchy core loops)
notebooks/                # reproducible analyses and figures
src/LinkNodeCommunity/    # Python package (core algorithms and utilities)
docs/                     # API documentation and tutorials (to be expanded)
tests/                    # pytest-based regression tests
CMakeLists.txt            # top-level CMake for native extensions
CHANGELOG.md              # Project history (releases, fixes, new features)
LICENSE                   # License terms for reuse
README.md                 # Project overview and usage guide
pyproject.toml            # build metadata (scikit-build-core, dependencies)
```

## Testing & Quality
- Run the unit suite with `pytest -q`.
- Optional style checks (if `dev` extras installed):
  - `ruff check src tests`
  - `mypy src`
- Continuous integration hooks can be wired by reusing `pyproject.toml`'s cibuildwheel configuration.

## Development Tips
- Prefer `method="matrix"` for sparse graphs; switch to `method="edgelist"` when memory is constrained in dense graphs.
- Use `Clustering.edge_complete=True` to focus on the strongly connected core of directed networks when statistics demand edge-complete nodes.
- Explore community resolutions via `equivalence_partition(score="S"|"D")` and compare against loop entropy maxima.
- Export Newick strings to inspect hierarchies with phylogenetic visualization libraries.

## Citation
If you use this software in academic work, please cite:
```
@article{MartinezArmas2024LinkNodeCommunity,
  title   = {An information theoretic approach to community detection in dense cortical networks reveals a nested hierarchical structure},
  author  = {Martinez Armas, Jorge S. and Knoblauch, Kenneth and Kennedy, Henry and Toroczkai, Zoltan},
  journal = {Cold Spring Harbor Laboratory Preprint},
  year    = {2024},
  doi     = {10.1101/2024.08.07.606907}
}
```

## Funding & Acknowledgements
Supported by NSF IIS-1724297, French ANR grants (A2P2MC ANR-17-NEUC-0004, ANR-17-FLAG-ERA-HBP-CORTICITY, ANR-19-CE37-0025-DUAL_STREAMS), and collaborative efforts across Notre Dame, Washington University, Université Claude Bernard Lyon 1, and partner institutes. We thank B. Molnár, L. Magrou, and Y. Hou for insightful discussions, and acknowledge the use of the CORE-NETS connectivity datasets.

## License
Distributed under the terms of the repository's [LICENSE](LICENSE).

## References
- Ahn, Y.-Y., Bagrow, J. P., & Lehmann, S. (2010). Link communities reveal multiscale network complexity. *Nature*, 466, 761–764. https://doi.org/10.1038/nature09182
- Lancichinetti, A., & Fortunato, S. (2009). Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities. *Phys. Rev. E*, 80, 016118.
- Martinez Armas, J., Knoblauch, K., Kennedy, H., & Toroczkai, Z. (2024). *An information theoretic approach to community detection in dense cortical networks reveals a nested hierarchical structure* (preprint to be submitted shortly).
- Markov, N., et al. (2011). Weight consistency specifies the regularities of macaque cortical networks. *Cerebral Cortex*, 21, 1254–1272.
- Markov, N., et al. (2012). A weighted and directed interareal connectivity matrix for macaque cerebral cortex. *Cerebral Cortex*. https://doi.org/10.1093/cercor/bhs127
- Müllner, D. (2013). fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for R and Python. *Journal of Statistical Software*, 53(9), 1–18.
