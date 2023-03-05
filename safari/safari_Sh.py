import numpy as np

a = {
    "L00" : {
      "L11" : {
        "L21" :{
          "L31" : {},
          "L32" : {}
        },
        "L22" :{
          "L33" : {
            "L41" :{},
            "L42": {
              "L51" : {
                "L61" : {},
                "L62" : {
                  "L71" : {},
                  "L72" : {
                    "L81" :{}
                  }
                }
              }
            }
          }
        }
      }
    }
  }


def sum_vertices(tree : dict, i):
  for key in tree.keys():
    i += 1
    if len(tree[key]) == 0: continue
    sum_vertices(tree[key], i )

def ML(tree : dict, ml : dict):
  for key in tree.keys():
    if key[:2] not in ml.keys(): ml[key[:2]] = 1
    else: ml[key[:2]] += 1
    ML(tree[key], ml)

def SH(tree : dict, Ml : dict, Sh):
  for key in tree.keys():
    if len(tree[key]) == 0: continue
    i = int(key[1])
    Mul = len(tree[key])
    Sh -= Mul * np.log(Mul / Ml[f"L{i+1}"])
    SH(tree[key], Ml, Sh)

def SV(Ml : dict, M, Sv):
  for key in Ml.keys():
    Sv -= Ml[key] * np.log(Ml[key] / M)

def S(a):
  M = np.array([0])
  Ml = {}
  sum_vertices(a, M)
  ML(a, Ml)
  Sh =np.zeros(1)
  Sv = np.zeros(1)
  SH(a, Ml, Sh)
  SV(Ml, M, Sv)
  return Sh[0] + Sv[0], Sh[0], Sv[0]

am = {
  "L00" : {
    "L10" : {
      "L20" : {},
      "L21" : {}
    },
    "L11" : {
      "L23" : {},
      "L24" : {
        "L30" : {},
        "L31" : {}
      }
    }
  }
}

A = np.array(
  [
    [1, 1, 1, 1, 1],
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 3],
    [1, 2, 3, 3, 4],
    [1, 2, 3, 4, 5]
  ]
)

def Z2dict_long(M, tree : dict, key_prev, nodes_prev : set, L, tL):
  if L < M.shape[0]:
    coms = M[L, :]
    for i, com in enumerate(np.unique(coms)):
      key = f"L{tL}{i}"
      nodes_com = set(list(np.where(coms == com)[0]))
      compare = nodes_com.intersection(nodes_prev)
      if len(compare) > 0:
        if key_prev not in tree.keys(): tree[key_prev] = {}
        Z2dict_long(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
  else: tree[key_prev] = {}

def Z2dict_short(M, tree : dict, key_prev, nodes_prev : set, L, tL):
  if L < M.shape[0]:
    coms = M[L, :]
    for i, com in enumerate(np.unique(coms)):
      key = f"L{tL}{i}"
      nodes_com = set(list(np.where(coms == com)[0]))
      compare = nodes_com.intersection(nodes_prev)
      if len(compare) > 0:
        if len(nodes_com) == len(nodes_prev):
          Z2dict_short(M, tree, key_prev, nodes_com, L + 1, tL)
        elif len(nodes_com) < len(nodes_prev):
          if key_prev not in tree.keys(): tree[key_prev] = {}
          Z2dict_short(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
  else: tree[key_prev] = {}

def Z2dict(A, f):
  nodes = set(list(np.arange(A.shape[1], dtype=int)))
  a_tree = {}
  L = 1
  tL = 1
  f(A, a_tree, "L00", nodes, L, tL)
  return a_tree

if __name__ == "__main__":
  tree = Z2dict(A, Z2dict_short)
  print(tree)
  print(am)
  # print(S(a))