import numpy as np
from collections import Counter
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform

def get_hierarchical_association(n, Z, perm=(False, None)):
    hierarchical_association = np.zeros((n, n))
    for z in np.arange(1, n):
      node_partition = cut_tree(Z, n_clusters=z).ravel().astype(int)
      if perm[0]:
        node_partition = node_partition[perm[1][perm[1]]]
      communities = Counter(node_partition)
      communities = [k for k in communities.keys() if communities[k] > 1]
      for k in communities:
        nodes = np.where(node_partition == k)[0]
        x, y = np.meshgrid(nodes, nodes)
        keep = x != y
        x = x[keep]
        y = y[keep]
        hierarchical_association[x, y] = Z[n - 1 - z, 2]
    return hierarchical_association

A = np.random.randn(5, 5)
A = A + A.T
A = np.abs(A)
np.fill_diagonal(A, 0)

ZA = linkage(squareform(A))

print(ZA)


H = get_hierarchical_association(A.shape[0], ZA)
# print(H)
# H = np.sum(H, axis=0)
# print(H)

ZH = linkage(squareform(H))

print("\n\n")
print(ZH)