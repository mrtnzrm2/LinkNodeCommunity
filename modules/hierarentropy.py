import numpy as np

class Hierarchical_Entropy:
  def __init__(self, Z, nodes : int) -> None:
    from scipy.cluster.hierarchy import cut_tree
    self.total_nodes = nodes
    self.A = np.zeros((nodes, nodes))
    self.nodes = np.arange(nodes, dtype=int)
    self.height = np.zeros(nodes)
    self.A[nodes - 1, :] = self.nodes
    for i in np.arange(nodes - 1, 0, -1):
      self.A[i - 1, :] = cut_tree(Z, i).ravel()
    # print(self.A)
    self.get_height_Z(Z)
    # print(self.height)

  def get_height_Z(self, Z):
    self.height[1:] = Z[:, 2]

  def sum_vertices(self, tree : dict, i):
    for key in tree.keys():
      if key == "height": continue
      i += 1
      if len(tree[key]) == 1: continue
      self.sum_vertices(tree[key], i )

  def ML(self, tree : dict, ml : dict):
    for key in tree.keys():
      if key == "height": continue
      ski = key.split("_")[0]
      if ski not in ml.keys(): ml[ski] = 1
      else: ml[ski] += 1
      self.ML(tree[key], ml)

  def ML_height(self, tree : dict, ml : dict, maxlvl : int):
    for key in tree.keys():
      if key == "height": continue
      ski = key.split("_")[0]
      if self.is_leaf(tree[key]):
        if ski not in ml.keys():
          ml[ski] = {
            "size" : 1,
            "height" : tree[key]["height"]
          }
        else:
          ml[ski]["size"] += 1
          ml[ski]["height"] += tree[key]["height"]
        ski2 = int(ski[1:]) + 1
        if ski2 <= maxlvl:
          ski2 = f"L{ski2}"
          if ski2 not in ml.keys():
            ml[ski2] = {
              "size" : 1,
              "height" : 0
            }
          else:
            ml[ski2]["size"] += 1
      else:
        if ski not in ml.keys():
          ml[ski] = {
            "size" : 1,
            "height" : 0
          }
          for key2 in tree[key].keys():
            if key2 == "height": continue    
            ml[ski]["height"] += tree[key]["height"] - tree[key][key2]["height"]
        else:
          ml[ski]["size"] += 1
          for key2 in tree[key].keys():
            if key2 == "height": continue    
            ml[ski]["height"] += tree[key]["height"] - tree[key][key2]["height"]
      self.ML_height(tree[key], ml, maxlvl)

  def SH(self, tree : dict, Ml : dict, Sh):
    for key in tree.keys():
      if key == "height": continue
      if len(tree[key]) == 1: continue
      i = int(key.split("_")[0][1:])
      Mul = len(tree[key]) - 1
      Sh[self.total_nodes - i -1] -= Mul * np.log(Mul / Ml[f"L{i+1}"])
      self.SH(tree[key], Ml, Sh)

  def SV(self, Ml : dict, M, Sv):
    for key in Ml.keys():
      i = int(key[1:])
      Sv[self.total_nodes - i - 1] -= Ml[key] * np.log(Ml[key] / M)

  def is_leaf(self, tree : dict):
    if len(tree.keys()) == 1:
      return True
    else: return False

  def SH_height(self, tree : dict, key, Ml : dict, maxl : int, Sh):
    i = int(key.split("_")[0][1:])
    Mul = len(tree[key]) - 1
    if Mul > 1 and i < maxl:
      for key2 in tree[key].keys():
        if key2 == "height": continue
        # print(key, key2, tree[key]["height"], tree[key][key2]["height"], Mul, Ml[f"L{i+1}"]["size"])
        Sh[self.total_nodes - i -1] -= (tree[key]["height"] - tree[key][key2]["height"]) * np.log(Mul / Ml[f"L{i+1}"]["size"])
        self.SH_height(tree[key], key2, Ml, maxl, Sh)
    elif i < maxl:
      # print("**", key, tree[key]["height"], Mul, Ml[f"L{i+1}"]["size"])
      Sh[self.total_nodes - i -1] -= tree[key]["height"] * np.log(1 / Ml[f"L{i+1}"]["size"])

  def SV_height(self, Ml : dict, M, Sv):
    for key in Ml.keys():
      if key == "height": continue
      i = int(key[1:])
      Sv[self.total_nodes - i - 1] -= Ml[key]["height"] * np.log(Ml[key]["size"] / M)

  def max_level(self, tree : dict, max_lvl):
    for key in tree.keys():
      if key == "height": continue
      ski = int(key.split("_")[0][1:])
      if max_lvl[0] < ski:
        max_lvl[0] = ski
      self.max_level(tree[key], max_lvl)

  def S(self, a):
    M = np.array([0])
    Ml = {}
    maxlevl = np.array([0])
    self.max_level(a, maxlevl)
    maxlevl = maxlevl[0]
    Sh =np.zeros(self.total_nodes)
    Sv = np.zeros(self.total_nodes)
    self.sum_vertices(a, M)
    self.ML(a, Ml)
    self.SH(a, Ml, Sh)
    self.SV(Ml, M, Sv)
    Sh /= self.nodes.shape[0]
    Sv /= self.nodes.shape[0]
    print(f"\n\tNode entropy : {np.sum(Sh + Sv):.4f}, Sh : {np.sum(Sh):.4f}, and Sv : {np.sum(Sv):.4f}\n")
    return np.vstack([Sh[maxlevl:], Sv[maxlevl:]])
  
  def print_tree(self, a : dict, key_pred=""):
    for key in a.keys():
      if key == "height": continue
      # if len(a[key]) == 1:continue
      print(f"{key_pred}{key}", "\t\t", a[key]["height"])
      self.print_tree(a[key], key_pred=f"{key_pred}{key}")

  def print_tree_ml(self, a : dict):
    for key in a.keys():
      if key == "height": continue
      print(key, a[key]["size"], a[key]["height"])

  def S_height(self, a):
    # self.print_tree(a, "")
    M = np.array([0])
    Ml = {}
    maxlevl = np.array([0])
    self.max_level(a, maxlevl)
    maxlevl = maxlevl[0]
    Sh =np.zeros(self.total_nodes)
    Sv = np.zeros(self.total_nodes)
    self.sum_vertices(a, M)
    self.ML_height(a, Ml, maxlevl)
    # self.print_tree_ml(Ml)
    self.SH_height(a, "L0_0", Ml, maxlevl, Sh)
    self.SV_height(Ml, M, Sv)
    Sh /= self.nodes.shape[0]
    Sv /= self.nodes.shape[0]
    print(f"\n\tNode entropy H: {np.sum(Sh + Sv):.4f}, Sh : {np.sum(Sh):.4f}, and Sv : {np.sum(Sv):.4f}\n")
    return np.vstack([Sh[maxlevl:], Sv[maxlevl:]])

  def Z2dict_long(self, M, tree : dict, key_prev, nodes_prev : set, L, tL):
    if L < M.shape[0] and len(nodes_prev) > 1:
      coms = [M[L, i] for i in nodes_prev]
      for i, com in enumerate(np.unique(coms)):
        key = f"L{tL}_{int(com)}"
        nodes_com = set(list(np.where(M[L, :] == com)[0]))
        compare = nodes_com.intersection(nodes_prev)
        if len(compare) > 0:
          if key_prev not in tree.keys():
            tree[key_prev] = {
              "height" : self.height[self.total_nodes - L]
            }
          self.Z2dict_long(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
    else:
      tree[key_prev] = {
        "height" : self.height[self.total_nodes - L]
      }

  def Z2dict_short(self, M, tree : dict, key_prev, nodes_prev : set, L, tL):
    if L < M.shape[0] and len(nodes_prev) > 1:
      coms = [M[L, i] for i in nodes_prev]
      for i, com in enumerate(np.unique(coms)):
        key = f"L{tL}_{int(com)}"
        nodes_com = set(list(np.where(M[L, :] == com)[0]))
        compare = nodes_com.intersection(nodes_prev)
        if len(compare) == 0: continue
        if len(nodes_com) == len(nodes_prev):
          self.Z2dict_short(M, tree, key_prev, nodes_com, L + 1, tL)
        elif len(nodes_com) < len(nodes_prev):
          if key_prev not in tree.keys():
            tree[key_prev] = {
              "height" : self.height[self.total_nodes - L]
            }
          self.Z2dict_short(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
    else:
      if key_prev not in tree.keys():
        tree[key_prev] = {
          "height" : self.height[self.total_nodes - L]
        }

  def Z2dict(self, Z2):
    self.tree = {"height" : None}
    nodes = set(list(self.nodes))
    L = 1
    tL = 1
    if Z2 == "short":
      self.Z2dict_short(self.A, self.tree, "L0_0", nodes, L, tL)
    elif Z2 == "long":
      self.Z2dict_long(self.A, self.tree, "L0_0", nodes, L, tL)
    else: raise ValueError("Only Z2 short or long")