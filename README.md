# LinkNodeCommunity

<!-- TODO: add PyPI, build, docs, and coverage badges when available -->

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Scientific Motivation](#scientific-motivation)
- [Getting Started](#getting-started)
- [Quickstart](#quickstart)
- [API Highlights](#api-highlights)
- [Data & Notebooks](#data--notebooks)
- [Repository Layout](#repository-layout)
- [Testing & Quality](#testing--quality)
- [Development Tips](#development-tips)
- [Citation](#citation)
- [Funding & Acknowledgements](#funding--acknowledgements)
- [License](#license)
- [References](#references)

## Overview
LinkNodeCommunity implements an information-theoretic link community algorithm tailored to dense, directed, and weighted networks. It grew out of mesoscale cortical connectivity studies and formalizes a workflow that couples link similarity, hierarchical clustering, and statistical diagnostics to expose nested community structure. The package bundles high-performance C++ extensions (via pybind11) with a Python-first API that integrates with NetworkX, pandas, and SciPy tooling.

## Key Features
- Information-theoretic link similarity based on the Hellinger metric and 1/2-Rényi divergence
- Dual hierarchies: link communities and node communities derived from the same similarity landscape
- Loop entropy scoring to pinpoint a "Goldilocks" resolution for mesoscale organization
- Support for directed or undirected graphs, heterogeneous weights, and optional subgraph conditioning
- Utilities to export hierarchies as Newick trees for downstream phylogenetic-style visualization
- Benchmarks and diagnostics aligned with neuroscience-grade anatomical datasets and synthetic benchmarks

## Scientific Motivation
### Neuroscience
- Captures the mesoscale organization of macaque cortical networks obtained from retrograde tract-tracing (see `data/macaque`).
- Aligns structural partitions with functional hypotheses by identifying link communities that preserve densely interconnected feedback circuits.

### Information Theory
- Measures similarity through the Hellinger distance and the associated 1/2-Rényi divergence, ensuring that community assignments reflect distinguishable connection profiles.
- Uses loop entropy to balance description length against redundancy, surfacing an optimal hierarchy level for information processing.

### Network Science
- Extends the link community paradigm of Ahn, Bagrow, and Lehmann (2010) to weighted, directed, and dense graphs.
- Employs hierarchical clustering routines optimized in C++ (`link_hierarchy_statistics_cpp`, `node_community_hierarchy_cpp`) to handle thousands of links efficiently.

## Getting Started
### Prerequisites
- Python 3.10+
- A C++17 toolchain compatible with scikit-build-core, CMake (>=3.24), and Ninja
- Recommended Python dependencies: `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `scikit-learn`, `seaborn`
- Optional: R (for `ggtree` visualizations accessed via `rpy2`)

### Installation
**From PyPI (planned):**
```bash
pip install LinkNodeCommunity
```

**From TestPyPI (pre-release builds):**
```bash
pip install -i https://test.pypi.org/simple/ LinkNodeCommunity
```

**From source:**
```bash
# clone your fork or the upstream repository
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install .[dev]
```
The editable install triggers scikit-build to compile the C++ extensions located under `cpp/`.

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

# Build a small weighted directed graph
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
- `LinkNodeCommunity.core`: package exposing `framework`, `similarity`, `nocs`, and `tonewick` modules with lazy loading for fast imports and IDE completion support.
- `utils`: helper functions for edgelist management, equivalence partitions, validity checks, and fast dendrogram cuts.

## Data & Notebooks
- `data/macaque`: curated cortical connectivity matrices (FLN weights, anatomical distances, region labels) used throughout the associated manuscript.
- `notebooks/`: exploratory analyses (e.g., `PaparAnalysis.ipynb`, `HierarchicaScaleFree.ipynb`, `ErdosRenyi.ipynb`) illustrating usage on empirical and synthetic graphs.

## Repository Layout
```
cpp/                      # pybind11 extensions (C++ implementations of hierarchy core loops)
data/                     # reference datasets (macaque)
notebooks/                # reproducible analyses and figures
src/LinkNodeCommunity/    # Python package (core algorithms and utilities)
tests/                    # pytest-based regression tests
CMakeLists.txt            # top-level CMake for native extensions
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
- Martinez Armas, J., Knoblauch, K., Kennedy, H., & Toroczkai, Z. (2024). *An information theoretic approach to community detection in dense cortical networks reveals a nested hierarchical structure*. Cold Spring Harbor Laboratory. https://doi.org/10.1101/2024.08.07.606907
- Markov, N., et al. (2011). Weight consistency specifies the regularities of macaque cortical networks. *Cerebral Cortex*, 21, 1254–1272.
- Markov, N., et al. (2012). A weighted and directed interareal connectivity matrix for macaque cerebral cortex. *Cerebral Cortex*. https://doi.org/10.1093/cercor/bhs127
- Müllner, D. (2013). fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for R and Python. *Journal of Statistical Software*, 53(9), 1–18.
