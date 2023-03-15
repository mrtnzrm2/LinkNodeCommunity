# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from modules.hierarentropy import Hierarchical_Entropy

a = {
    "L0_0" : {
      "L1_1" : {
        "L2_1" :{
          "L3_1" : {},
          "L3_2" : {}
        },
        "L2_2" :{
          "L3_3" : {
            "L4_1" :{},
            "L4_2": {
              "L5_1" : {
                "L6_1" : {},
                "L6_2" : {
                  "L7_1" : {},
                  "L7_2" : {
                    "L8_1" :{}
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
    i = int(key.split("_")[0][1:])
    Mul = len(tree[key])
    Sh -= Mul * np.log(Mul / Ml[f"L{i+1}"])
    SH(tree[key], Ml, Sh)

def SV(Ml : dict, M, Sv):
  for key in Ml.keys():
    Sv -= Ml[key] * np.log(Ml[key] / M)

def S(a, nodes):
  M = np.array([0])
  Ml = {}
  sum_vertices(a, M)
  ML(a, Ml)
  Sh = np.zeros(1)
  Sv = np.zeros(1)
  SH(a, Ml, Sh)
  SV(Ml, M, Sv)
  Sh /= nodes
  Sv /= nodes
  return Sh[0] + Sv[0], Sh[0], Sv[0]

am = {
  "L0_0" : {
    "L1_0" : {
      "L2_0" : {},
      "L2_1" : {}
    },
    "L1_1" : {
      "L2_3" : {},
      "L2_4" : {
        "L3_0" : {},
        "L3_1" : {}
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
A2 = np.array(
  [
    [0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0, 0 ,0],
    [0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0 ,1, 0, 0, 0 ,0 ,0],
    [0 ,0 ,1 ,1 ,1, 0 ,0 ,0, 0 ,0 ,2 ,1, 1 ,1,1 ,1],
    [0, 0 ,1, 1 ,1 ,0, 0 ,0 ,0 ,0, 2, 3 ,3 ,3, 3, 3],
    [0 ,0 ,1, 1, 1, 0, 0, 0, 0 ,0, 2 ,3 ,3, 4, 3, 3],
    [0 ,0 ,1, 1, 1, 2 ,2 ,2, 2, 2 ,3, 4 ,4 ,5,4, 4],
    [0 ,0 ,1 ,1 ,2, 3 ,3 ,3, 3 ,3 ,4 ,5, 5 ,6 ,5 ,5],
    [0 ,0, 1 ,1, 2, 3, 3 ,3, 3, 3 ,4 ,5 ,6, 7, 5 ,5],
    [0 ,0 ,1 ,1, 2, 3, 3 ,3, 4 ,3 ,5, 6, 7, 8, 6, 6],
    [0 ,1 ,2 ,2 ,3, 4, 4 ,4 ,5 ,4 ,6 ,7 , 8, 9, 7, 7],
    [0, 1, 2, 2, 3, 4, 4, 5, 6, 5, 7, 8, 9, 10, 8, 8],
    [0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 9, 9],
    [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 10, 10],
    [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,11],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  ]
)

def Z2dict_long(M, tree : dict, key_prev, nodes_prev : set, L, tL):
  if L < M.shape[0]:
    coms = M[L, :]
    for i, com in enumerate(np.unique(coms)):
      key = f"L{tL}_{com}"
      nodes_com = set(list(np.where(coms == com)[0]))
      compare = nodes_com.intersection(nodes_prev)
      if len(compare) > 0:
        if key_prev not in tree.keys(): tree[key_prev] = {}
        Z2dict_long(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
  else: tree[key_prev] = {}

def Z2dict_short(M, tree : dict, key_prev, nodes_prev : set, L, tL):
  if L < M.shape[0] and len(nodes_prev) > 1:
    coms = M[L, :]
    for i, com in enumerate(np.unique(coms)):
      key = f"L{tL}_{com}"
      nodes_com = set(list(np.where(coms == com)[0]))
      compare = nodes_com.intersection(nodes_prev)
      if len(compare) == 0: continue
      if len(nodes_com) == len(nodes_prev):
        Z2dict_short(M, tree, key_prev, nodes_com, L + 1, tL)
      elif len(nodes_com) < len(nodes_prev):
        if key_prev not in tree.keys(): tree[key_prev] = {}
        Z2dict_short(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
  else: tree[key_prev] = {}

def print_tree(a : dict, key_pred=""):
    for key in a.keys():
      print(f"{key_pred}{key}", "\t\t")
      print_tree(a[key], key_pred=f"{key_pred}{key}")

def Z2dict(A, f):
  nodes = set(list(np.arange(A.shape[1], dtype=int)))
  a_tree = {}
  L = 1
  tL = 1
  f(A, a_tree, "L00", nodes, L, tL)
  return a_tree

if __name__ == "__main__":
  tree = Z2dict(A, Z2dict_short)
  # print(tree)
  print_tree(tree)
  print(S(tree, 1))
  # H =  Hierarchical_Entropy(0, 16)
  # H.A = A2
  # H.Z2dict("short")
  # _, _, _ = H.S(H.tree)