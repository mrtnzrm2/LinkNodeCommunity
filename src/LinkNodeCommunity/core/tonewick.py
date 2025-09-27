"""
Path: src/LinkNodeCommunity/core/tonewick.py

Module: LinkNodeCommunity.core.tonewick
Author: Jorge S. Martinez Armas

Overview:
---------
Converts SciPy linkage matrices into Newick strings while exposing the
intermediate tree representation used by LinkNodeCommunity visualisations.

Key Components:
---------------
- LinkageToNewick: validates linkage input, builds a nested tree dictionary, and
  renders deterministic Newick output.

Notes:
------
- Supports optional branch-length scaling and preserves child ordering for
  stable outputs.
"""

import numpy as np
import numpy.typing as npt

class LinkageToNewick:
    def __init__(self, Z: npt.NDArray, labels: npt.ArrayLike | None = None, branch_length: bool = True) -> None:
        self.Z = Z
        self.N = Z.shape[0] + 1
        self.branch_length = branch_length

        if labels is None:
            # Default labels are 0..N-1 as strings
            self.labels = np.arange(self.N).astype(str)
        elif len(labels) == self.N:
            self.labels = labels
        else:
            raise ValueError("Labels must be the same length as the number of nodes.")
        self.root = "L0_0"
        # Validate linkage input for structural and height consistency
        self._validate_Z()

    def fit(self):
        """Build an internal tree from linkage `Z` and compute the Newick string.

        Populates `self.tree` (nested dict structure) and `self.newick` (str).
        """
        self.tree = {}

        level = 1 # start from root level

        # Create the membership sets
        self.T = {i: [set([i]), [None], 0] for i in range(self.N)}

        for i in np.arange(self.N - 1):
            nx, ny = int(self.Z[i, 0]), int(self.Z[i, 1])
            self.T[(self.N+i)] = [self.T[(nx)][0].union(self.T[(ny)][0]), [nx, ny], self.Z[i, 2]]

        # Build the tree recursively
        self.fit_recursive(self.tree, self.root, (2*self.N-2), level)

        # Convert to Newick format
        self.zdict2newick_(self.tree, branch_length=self.branch_length)

    def fit_recursive(self, tree: dict, parent_string_key: str, parent_merge_key: int, level: int) -> None:
        """Recursively populate `tree` from `self.T` starting at `parent_merge_key`.

        - Adds "height" at every internal node and "label" at leaves.
        - Keys are synthetic labels like `L{level}_{id}`; values are nested dicts.
        """

        current_nodes, current_children, current_height = self.T[parent_merge_key]
        if current_children != [None]:
            for c in current_children:
                children_key = f"L{level}_{int(c)}" # String key for the current node
                if parent_string_key not in tree.keys():
                    tree[parent_string_key] = {
                    "height" : current_height
                    }

                self.fit_recursive(tree[parent_string_key], children_key, (c), level + 1)
        else:
            if parent_string_key not in tree.keys():
                tree[parent_string_key] = {
                "height" : current_height,
                "label" : self.labels[list(current_nodes)[0]]
                }

    def is_leaf(self, tree : dict) -> bool:
        if "label" in tree.keys():
            return True
        else: return False

    def zdict2pre_newick(
            self, 
            tree : dict, 
            root :str, 
            key_pred : str, 
            pre_newick : dict, 
            branch_length=True
        ) -> None:
        """Convert nested tree dict into a child-distance map for Newick rendering.

        Produces a mapping in `pre_newick`: parent_node -> {child_node: distance}.
        When `branch_length` is False, uses unit distances.
        """
        # If current node is a leaf, nothing to add
        if self.is_leaf(tree[root]):
            return

        if not isinstance(tree[root], dict):
            raise ValueError("Tree structure not recognized.")

        leaf_labels = [leaf for leaf in tree[root].keys() if leaf != "height" and leaf != "label"]
        # NOTE: Sorted for determinism; check perf impact later.
        leaf_labels.sort()
        is_root_leaf = [self.is_leaf(tree[root][leaf]) for leaf in leaf_labels]
        distances = [tree[root]["height"] - tree[root][leaf]["height"] for leaf in leaf_labels]
        if branch_length:
            if np.any(is_root_leaf): 
                dic1 = {key_pred + root + k: dis for is_leaf, k, dis in zip(is_root_leaf, leaf_labels, distances) if not is_leaf}
                dic2 = {tree[root][leaf]["label"] : 0 for is_leaf, leaf, dis in zip(is_root_leaf, leaf_labels, distances) if is_leaf}
                pre_newick[key_pred + root] = {**dic1, **dic2}
            else:
                pre_newick[key_pred + root] = {key_pred + root + k: dis for k, dis in zip(leaf_labels, distances)}
        else:
            if np.any(is_root_leaf):
                dic1 = {key_pred + root + k: 1 for is_leaf, k in zip(is_root_leaf, leaf_labels) if not is_leaf}
                dic2 = {tree[root][leaf]["label"] : 1 for is_leaf, leaf in zip(is_root_leaf, leaf_labels) if is_leaf}
                pre_newick[key_pred + root] = {**dic1, **dic2}
            else:
                pre_newick[key_pred + root] = {key_pred + root + k: 1 for k in leaf_labels}
        for leaves in leaf_labels:
            self.zdict2pre_newick(tree[root], leaves, key_pred + root, pre_newick, branch_length=branch_length)

    def zdict2newick_(self, tree: dict, branch_length: bool = True) -> None:
        """Compute `self.newick` from a nested tree dict using Newick format."""
        pre_newick = {}
        self.zdict2pre_newick(tree, self.root, "", pre_newick, branch_length=branch_length)
        self.newick = self.newickify(pre_newick, root_node=self.root)

    def _validate_Z(self) -> None:
        """Validate the linkage matrix `Z` for structure and monotonic heights.

        Requirements:
        - `Z` is shape (N-1, >=3), numeric and finite in column 2 (heights).
        - Heights are non-negative and non-decreasing (within small tolerance).
        - Child indices at row i are in [0, N+i) and not equal.
        """
        if not isinstance(self.Z, np.ndarray) or self.Z.ndim != 2 or self.Z.shape[0] < 1 or self.Z.shape[1] < 3:
            raise ValueError("Z must be a 2D array with shape (n-1, >=3).")

        n = self.Z.shape[0] + 1
        heights = self.Z[:, 2]
        if not np.issubdtype(self.Z.dtype, np.number):
            raise ValueError("Z must be numeric.")
        if not np.all(np.isfinite(heights)):
            raise ValueError("Heights (Z[:,2]) must be finite.")
        if np.any(heights < 0):
            raise ValueError("Heights (Z[:,2]) must be non-negative.")

        # Enforce non-decreasing heights (allow tiny numerical jitter)
        if np.any(np.diff(heights) < -1e-12):
            raise ValueError("Heights (Z[:,2]) must be non-decreasing to avoid negative branch lengths.")

        # Validate child indices progressively
        for i in range(n - 1):
            try:
                nx = int(self.Z[i, 0])
                ny = int(self.Z[i, 1])
            except Exception as e:
                raise ValueError(f"Z child indices must be integers at row {i}.") from e
            if nx == ny:
                raise ValueError(f"Invalid linkage at row {i}: identical children ({nx}).")
            if nx < 0 or ny < 0:
                raise ValueError(f"Invalid linkage at row {i}: negative child index.")
            # At step i, valid indices are [0, n + i)
            if nx >= n + i or ny >= n + i:
                raise ValueError(
                    f"Invalid linkage at row {i}: child index out of range (must be < {n + i})."
                )

    def newickify(self, node_to_children, root_node) -> str:
        """
        Source code:
        https://stackoverflow.com/questions/50003007/how-to-convert-python-dictionary-to-newick-form-format
        """
        visited_nodes = set()
        def newick_render_node(name, distance: float) -> str:
            assert name not in visited_nodes, "Error: The tree may not be circular!"
            if name not in node_to_children:
                # Leafs
                return F'{name}:{distance}'
            else:
                # Nodes
                visited_nodes.add(name)
                children = node_to_children[name]
                children_strings = [newick_render_node(child, children[child]) for child in children.keys()]
                children_strings = ",".join(children_strings)
                return F'({children_strings}):{distance}'
        newick_string = newick_render_node(root_node, 0) + ';'
        # Ensure no entries in the dictionary are left unused.
        assert visited_nodes == set(node_to_children.keys()), "Error: some nodes aren't in the tree"
        return newick_string
