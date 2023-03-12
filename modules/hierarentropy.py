import numpy as np

class Hierarchical_Entropy:
  def __init__(self, Z, nodes : int) -> None:
    from scipy.cluster.hierarchy import cut_tree
    self.A = np.zeros((nodes, nodes))
    self.nodes = np.arange(nodes, dtype=int)
    self.A[nodes - 1, :] = self.nodes
    for i in np.arange(nodes - 1, 0, -1):
      self.A[i - 1, :] = cut_tree(Z, i).ravel()

  def sum_vertices(self, tree : dict, i):
    for key in tree.keys():
      i += 1
      if len(tree[key]) == 0: continue
      self.sum_vertices(tree[key], i )

  def ML(self, tree : dict, ml : dict):
    for key in tree.keys():
      ski = key.split("_")[0]
      if ski not in ml.keys(): ml[ski] = 1
      else: ml[ski] += 1
      self.ML(tree[key], ml)

  def SH(self, tree : dict, Ml : dict, Sh):
    for key in tree.keys():
      if len(tree[key]) == 0: continue
      i = int(key.split("_")[0][1:])
      Mul = len(tree[key])
      Sh -= Mul * np.log(Mul / Ml[f"L{i+1}"])
      self.SH(tree[key], Ml, Sh)

  def SV(self, Ml : dict, M, Sv):
    for key in Ml.keys():
      Sv -= Ml[key] * np.log(Ml[key] / M)

  def S(self, a):
    M = np.array([0])
    Ml = {}
    self.sum_vertices(a, M)
    self.ML(a, Ml)
    Sh =np.zeros(1)
    Sv = np.zeros(1)
    self.SH(a, Ml, Sh)
    self.SV(Ml, M, Sv)
    Sh /= self.nodes.shape[0]
    Sv /= self.nodes.shape[0]
    print(f"\n\tNode entropy : {(Sh[0] + Sv[0]):.4f}, Sh : {Sh[0]:.4f}, and Sv : {Sv[0]:.4f}\n")
    return Sh[0] + Sv[0], Sv[0], Sh[0]

  def Z2dict_long(self, M, tree : dict, key_prev, nodes_prev : set, L, tL):
    if L < M.shape[0] and len(nodes_prev) > 1:
      coms = [M[L, i] for i in nodes_prev]
      for i, com in enumerate(np.unique(coms)):
        key = f"L{tL}_{i}"
        nodes_com = set(list(np.where(M[L, :] == com)[0]))
        compare = nodes_com.intersection(nodes_prev)
        if len(compare) > 0:
          if key_prev not in tree.keys(): tree[key_prev] = {}
          self.Z2dict_long(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
    else: tree[key_prev] = {}

  def Z2dict_short(self, M, tree : dict, key_prev, nodes_prev : set, L, tL):
    if L < M.shape[0] and len(nodes_prev) > 1:
      coms = [M[L, i] for i in nodes_prev]
      for i, com in enumerate(np.unique(coms)):
        key = f"L{tL}_{com}"
        nodes_com = set(list(np.where(M[L, :] == com)[0]))
        compare = nodes_com.intersection(nodes_prev)
        if len(compare) == 0: continue
        if len(nodes_com) == len(nodes_prev):
          self.Z2dict_short(M, tree, key_prev, nodes_com, L + 1, tL)
        elif len(nodes_com) < len(nodes_prev):
          if key_prev not in tree.keys(): tree[key_prev] = {}
          self.Z2dict_short(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
    else: tree[key_prev] = {}

  def Z2dict(self, Z2):
    self.tree = {}
    nodes = set(list(self.nodes))
    L = 1
    tL = 1
    if Z2 == "short":
      self.Z2dict_short(self.A, self.tree, "L00_0", nodes, L, tL)
    elif Z2 == "long":
      self.Z2dict_long(self.A, self.tree, "L00_0", nodes, L, tL)
    else: raise ValueError("Only Z2 short or long")