# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import fsolve
from sklearn.manifold import MDS
sns.set_theme()
# Personal libraries ----
from networks.MAC.mac57 import MAC57

def equations(p, *args):
    x, y, z = p
    norm = np.sqrt(x**2 + y**2 + z **2)
    x = x / norm
    y = y / norm
    z = z / norm
    return (args[0][0]*x + args[0][1]*y + args[0][2]*z, args[1][0]*x + args[1][1]*y + args[1][2]*z - np.cos(args[2]), (y*args[0][2] - z*args[0][1]) ** 2 + (z*args[0][0] - x*args[0][2]) ** 2 + (x*args[0][1]-y*args[0][0])**2 - 1)


def ref_lambda(CC, DD):
    D = DD.copy()
    C = CC.copy()

    mind = np.min(D[D>0])
    maxd = np.max(D)
    
    Lx = 16

    d = np.linspace(mind, maxd + 1e-4, Lx)

    A = np.zeros((Lx-1,))

    for i in np.arange(C.shape[0]):
        for j in np.arange(C.shape[1]):
            if i == j : continue
            dij = D[i, j]
            Cij = C[i, j]

            for ii in np.arange(Lx - 1):
                if dij >= d[ii] and dij < d[ii + 1]:
                    break

            A[ii] += Cij

    dD = (d[1] - d[0]) / 2
    d = d + dD

    lda = (np.sum(A) - 2) / (np.sum(A * d[:-1])) 
    print(lda)
    return lda

def referenc_lambda(CC, DD):
    D = DD.copy()
    C = CC.copy()

    mind = np.min(D[D>0])
    maxd = np.max(D)

    model = MDS(n_components=3, dissimilarity="precomputed", max_iter=1000, random_state=1)
    coords = model.fit_transform(D)

    for i in np.arange(3):
      coords[:, i] = (coords[:, i] - np.mean(coords[:, i])) / np.std(coords[:, i])
    
    Lx = 16
    Ly = 17
    delta = 0.0

    d = np.linspace(mind, maxd + 1e-4, Lx)
    phi = np.linspace(0 - delta, np.pi * 2 + 1e-4 - delta, Ly)

    A = np.zeros((Lx-1, Ly-1))

    for i in np.arange(C.shape[0]):
        for j in np.arange(C.shape[1]):
            if i == j : continue
            dij = D[i, j]
            Cij = C[i, j]
            vij = coords[j, :2] - coords[i, :2]

            if vij[0] >= 0 and vij[1] >= 0:
              thetaij = np.arctan(np.abs(vij[1]/vij[0])) - delta
            elif vij[0] >=0 and vij[1] < 0:
              thetaij = 2*np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
            elif vij[0] <0 and vij[1] >= 0:
              thetaij = np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
            else:
               thetaij = np.pi + np.arctan(np.abs(vij[1]/vij[0])) - delta

            for ii in np.arange(Lx - 1):
                if dij >= d[ii] and dij < d[ii + 1]:
                    break
            for jj in np.arange(Ly-1):
                if thetaij >= phi[jj] and thetaij < phi[jj + 1]:
                   break
            A[ii, jj] += Cij

    A = A.T

    dD = (d[1] - d[0]) / 2
    d += dD

    Lda = [(np.sum(A[i, :]) - 2) / (np.sum(A[i, :] * d[:-1])) for i in np.arange(Ly-1)]
    Mda = np.zeros(D.shape)

    for i in np.arange(D.shape[0]):
      for j in np.arange(D.shape[1]):
        if i == j: continue
        vij = coords[j, :2] - coords[i, :2]
        if vij[0] >= 0 and vij[1] >= 0:
          thetaij = np.arctan(np.abs(vij[1]/vij[0])) - delta
        elif vij[0] >=0 and vij[1] < 0:
          thetaij = 2*np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
        elif vij[0] <0 and vij[1] >= 0:
          thetaij = np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
        else:
            thetaij = np.pi + np.arctan(np.abs(vij[1]/vij[0])) - delta
        for jj in np.arange(Ly-1):
            if thetaij >= phi[jj] and thetaij < phi[jj + 1]:
                break
        Mda[i, j] = Lda[jj]

    return Mda

def lambda_main_axis(CC, DD, labels, N=1000, Lx=16, Ly=17):
    from itertools import combinations
    from scipy.special import comb

    C = CC.copy()
    D = DD.copy()

    mind = np.min(D[D>0])
    maxd = np.max(D)

    a11 = np.where(labels == "v1pcuf")[0][0]
    a12 = np.where(labels == "gu")[0][0]

    Mda = np.zeros((N, D.shape[0], D.shape[1]))

    kk = 0
    while kk < N:

      model = MDS(n_components=3, dissimilarity="precomputed", max_iter=1000)
      coords = model.fit_transform(D)

      for i in np.arange(3):
        coords[:, i] = (coords[:, i] - np.mean(coords[:, i])) / np.std(coords[:, i])

      if np.random.randint(2) == 1:
        x_axis = coords[a11, :] - coords[a12, :]
      else:
        x_axis = coords[a12, :] - coords[a11, :]
      x_axis = x_axis / np.linalg.norm(x_axis)

      y_axis = np.zeros((comb(D.shape[0], 2).astype(int), 2))
      for k, ij in enumerate(combinations(np.arange(D.shape[0]), 2)):
          i, j = ij
          y_axis[k, 0] = np.abs(np.dot(x_axis , (coords[i, :] - coords[j, :]) / np.linalg.norm(coords[i, :] - coords[j, :])))
          y_axis[k, 1] = D[i, j]

      y_axis = y_axis[y_axis[:, 0] < 1e-3, :]
      if y_axis.shape[0] < 1: continue
      max_D2 = np.max(y_axis[:, 1])

      max_D2 = np.where(D == max_D2)

      if np.random.randint(2) == 1:
        d2x, d2y = max_D2[0][0], max_D2[0][1]
      else:
        d2y, d2x = max_D2[0][0], max_D2[0][1]

      y_axis = coords[d2x, :] - coords[d2y, :]
      y_axis = y_axis / np.linalg.norm(y_axis)

      theta = np.arccos(np.dot(x_axis, y_axis))
      if theta > np.pi / 2:
        phi = theta - np.pi / 2
      else:
        phi = np.pi / 2 - theta

      x_, y_, z_ =  fsolve(equations, x0=(0.5, 0.5, 0.5), args=(x_axis, y_axis, phi))

      y_axis = np.array([x_, y_, z_])
      y_axis = y_axis / np.linalg.norm(y_axis)

      if np.abs(np.dot(x_axis, y_axis)) != 0: continue

      z_axis = np.cross(x_axis, y_axis)

      A  = np.vstack([x_axis, y_axis, z_axis])
    
      for i in np.arange(D.shape[0]):
          coords[i, :] = np.matmul(coords[i, :].reshape(1, 3), A)

      # delta = 0
      delta = (np.random.randint(2)*2 - 1) * np.random.rand() * 2 * np.pi / (Ly+1)

      d = np.linspace(mind, maxd + 1e-4, Lx)
      phi = np.linspace(0 - delta, np.pi * 2 + 1e-4 - delta, Ly)
      # print(np.pi / (phi[1] - phi[0]))

      A = np.zeros((Lx-1, Ly-1))

      for i in np.arange(C.shape[0]):
          for j in np.arange(C.shape[1]):
              if i == j : continue
              dij = D[i, j]
              Cij = C[i, j]
              vij = coords[j, :2] - coords[i, :2]

              if vij[0] >= 0 and vij[1] >= 0:
                thetaij = np.arctan(np.abs(vij[1]/vij[0])) - delta
              elif vij[0] >=0 and vij[1] < 0:
                thetaij = 2*np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
              elif vij[0] <0 and vij[1] >= 0:
                thetaij = np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
              else:
                thetaij = np.pi + np.arctan(np.abs(vij[1]/vij[0])) - delta

              for ii in np.arange(Lx - 1):
                  if dij >= d[ii] and dij < d[ii + 1]:
                      break
              for jj in np.arange(Ly-1):
                  if thetaij >= phi[jj] and thetaij < phi[jj + 1]:
                    break
              A[ii, jj] += Cij

      A = A.T

      dD = (d[1] - d[0]) / 2
      d += dD

      Lda = [(np.sum(A[i, :]) - 2) / (np.sum(A[i, :] * d[:-1])) for i in np.arange(Ly-1)]
      
      for i in np.arange(D.shape[0]):
        for j in np.arange(D.shape[1]):
          if i == j: continue
          vij = coords[j, :2] - coords[i, :2]
          if vij[0] >= 0 and vij[1] >= 0:
            thetaij = np.arctan(np.abs(vij[1]/vij[0])) - delta
          elif vij[0] >=0 and vij[1] < 0:
            thetaij = 2*np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
          elif vij[0] <0 and vij[1] >= 0:
            thetaij = np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
          else:
              thetaij = np.pi + np.arctan(np.abs(vij[1]/vij[0])) - delta
          for jj in np.arange(Ly-1):
              if thetaij >= phi[jj] and thetaij < phi[jj + 1]:
                  break
          Mda[kk, i, j] = Lda[jj]
      kk += 1
    return Mda

def lambda_shuffle(CC, DD, N=1000, Lx=16, Ly=17):
    from itertools import combinations
    from scipy.special import comb

    C = CC.copy()
    D = DD.copy()

    mind = np.min(D[D>0])
    maxd = np.max(D)

    Mda = np.zeros((N, D.shape[0], D.shape[1]))

    kk = 0
    while kk < N:

      x, y = np.random.permutation(np.arange(D.shape[0]))[:2]

      model = MDS(n_components=3, dissimilarity="precomputed", max_iter=1000)
      coords = model.fit_transform(D)

      for i in np.arange(3):
        coords[:, i] = (coords[:, i] - np.mean(coords[:, i])) / np.std(coords[:, i])

      x_axis = coords[x, :] - coords[y, :]
      x_axis = x_axis / np.linalg.norm(x_axis)

      y_axis = np.zeros((comb(D.shape[0], 2).astype(int), 2))
      for k, ij in enumerate(combinations(np.arange(D.shape[0]), 2)):
          i, j = ij
          y_axis[k, 0] = np.abs(np.dot(x_axis , (coords[i, :] - coords[j, :]) / np.linalg.norm(coords[i, :] - coords[j, :])))
          y_axis[k, 1] = D[i, j]

      y_axis = y_axis[y_axis[:, 0] < 1e-3, :]
      if y_axis.shape[0] < 1: continue
      max_D2 = np.max(y_axis[:, 1])

      max_D2 = np.where(D == max_D2)

      if np.random.randint(2) == 1:
        d2x, d2y = max_D2[0][0], max_D2[0][1]
      else:
        d2y, d2x = max_D2[0][0], max_D2[0][1]

      y_axis = coords[d2x, :] - coords[d2y, :]
      y_axis = y_axis / np.linalg.norm(y_axis)

      theta = np.arccos(np.dot(x_axis, y_axis))
      if theta > np.pi / 2:
        phi = theta - np.pi / 2
      else:
        phi = np.pi / 2 - theta

      x, y, z =  fsolve(equations, x0=(0.5, 0.5, 0.5), args=(x_axis, y_axis, phi))

      y_axis = np.array([x, y, z])
      y_axis = y_axis / np.linalg.norm(y_axis)

      if np.abs(np.dot(x_axis, y_axis)) != 0: continue

      z_axis = np.cross(x_axis, y_axis)

      A  = np.vstack([x_axis, y_axis, z_axis])
    
      for i in np.arange(D.shape[0]):
          coords[i, :] = np.matmul(coords[i, :].reshape(1, 3), A)

      # delta = 0
      delta = (np.random.randint(2)*2 - 1) * np.random.rand() * 2 * np.pi / (Ly+1)

      d = np.linspace(mind, maxd + 1e-4, Lx)
      phi = np.linspace(0 - delta, np.pi * 2 + 1e-4 - delta, Ly)
      # print(np.pi / (phi[1] - phi[0]))

      A = np.zeros((Lx-1, Ly-1))

      for i in np.arange(C.shape[0]):
          for j in np.arange(C.shape[1]):
              if i == j : continue
              dij = D[i, j]
              Cij = C[i, j]
              vij = coords[j, :2] - coords[i, :2]

              if vij[0] >= 0 and vij[1] >= 0:
                thetaij = np.arctan(np.abs(vij[1]/vij[0])) - delta
              elif vij[0] >=0 and vij[1] < 0:
                thetaij = 2*np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
              elif vij[0] <0 and vij[1] >= 0:
                thetaij = np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
              else:
                thetaij = np.pi + np.arctan(np.abs(vij[1]/vij[0])) - delta

              for ii in np.arange(Lx - 1):
                  if dij >= d[ii] and dij < d[ii + 1]:
                      break
              for jj in np.arange(Ly-1):
                  if thetaij >= phi[jj] and thetaij < phi[jj + 1]:
                    break
              A[ii, jj] += Cij

      A = A.T

      dD = (d[1] - d[0]) / 2
      d += dD

      Lda = [(np.sum(A[i, :]) - 2) / (np.sum(A[i, :] * d[:-1])) for i in np.arange(Ly-1)]
      
      for i in np.arange(D.shape[0]):
        for j in np.arange(D.shape[1]):
          if i == j: continue
          vij = coords[j, :2] - coords[i, :2]
          if vij[0] >= 0 and vij[1] >= 0:
            thetaij = np.arctan(np.abs(vij[1]/vij[0])) - delta
          elif vij[0] >=0 and vij[1] < 0:
            thetaij = 2*np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
          elif vij[0] <0 and vij[1] >= 0:
            thetaij = np.pi - np.arctan(np.abs(vij[1]/vij[0])) - delta
          else:
              thetaij = np.pi + np.arctan(np.abs(vij[1]/vij[0])) - delta
          for jj in np.arange(Ly-1):
              if thetaij >= phi[jj] and thetaij < phi[jj + 1]:
                  break
          Mda[kk, i, j] = Lda[jj]
      kk += 1
    return Mda

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
structure = "LN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "D1_2_4"
discovery = "discovery_8"
bias = 0.
alpha = 0.
opt_score = ["_X", "_S"]
version = "57d106"
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
    NET = MAC57(
      linkage, mode,
      nlog10 = nlog10,
      structure = structure,
      lookup = lookup,
      version = version,
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      topology = topology,
      index = index,
      mapping = mapping,
      cut = cut,
      b = bias,
      alpha = alpha,
      discovery = discovery
    )

    C = NET.C.copy()
    D = NET.D.copy()
    D = (D + D.T) / 2

    N = 500
    Lx = 16
    Ly = 17

    lambda0 = ref_lambda(C, D)

    lambda_shf = lambda_shuffle(C, D, N=N, Lx=Lx, Ly=Ly)

    lambda_shf_ = pd.DataFrame()
    for i in np.arange(N):
       lambda_shf_ = pd.concat(
          [
             lambda_shf_,
             pd.DataFrame(
                {
                   "lambda" : lambda_shf[i].ravel(),
                   "target" : np.tile(NET.labels, D.shape[0]),
                   "source" : np.repeat(NET.labels, D.shape[1]),
                   "set" : ["random_axis"] * NET.labels.shape[0] * D.shape[0],
                   "iteration" : [i] * NET.labels.shape[0] * D.shape[0]
                }
             )
          ], ignore_index=True
       )

    data = pd.concat([lambda_shf_], ignore_index=True)

    data = data.groupby(["source", "target", "set"])["lambda"].mean().reset_index()
    data = data.loc[data["lambda"] > 0]

    from scipy.stats import ttest_1samp

    for l in NET.labels:
       lamb = data["lambda"].loc[data.source == l]
       test = ttest_1samp(lamb, popmean=lambda0)
       if test.pvalue <= 0.05 and test.pvalue > 0.01:
          print("*", l,  test.pvalue, np.mean(lamb))
       elif test.pvalue <= 0.01 and test.pvalue > 0.001:
          print("**", l,  test.pvalue, np.mean(lamb))
       elif test.pvalue <= 0.001 and test.pvalue > 0.0001:
          print("***", l,  test.pvalue, np.mean(lamb))
       elif test.pvalue <= 0.0001:
          print("****", l,  test.pvalue, np.mean(lamb)) 
       else:
          print("ns", l, test.pvalue, np.mean(lamb)) 

    data = data.loc[np.isin(data.source, NET.labels[:57])]

    sns.violinplot(
       data=data,
       x="source",
       y="lambda",
       hue="set"
    )

    plt.xticks(rotation=90)
    plt.show()


    # print(np.mean(Lda))

    # sns.scatterplot(x=phi[:-1], y=Lda)
    # plt.xticks(np.arange(np.pi*2 + np.pi/2, step=np.pi/2), ['0', 'π/2', 'π', '3π/2', '2π'])
    # plt.show()
       

    # x = []
    # for j in phi[:-1]:
    #   x += [f"({i:.2f},{j:.2f})" for i in d[:-1]]

    # from scipy.stats import expon

    # y = []
    # for lda in Lda:
    #    y += list(expon.pdf(d[:-1] , scale=1/lda))

    # for i in np.arange(Ly-1):
    #    A[i, :] = A[i, :] / (np.sum(A[i, :]) * dD)

    # sns.barplot(x=np.arange((Lx-1) * (Ly-1)), y=A.ravel(), color=sns.color_palette("deep")[0])
    # sns.scatterplot(x=np.arange((Lx-1) * (Ly-1)), y=y, color=sns.color_palette("deep")[1])

    # plt.yscale("log")
    # plt.xticks(np.arange((Lx-1) * (Ly-1)), x, rotation=90)

    # plt.show()


    # color = [[]] * D.shape[0]
    # for i in np.arange(D.shape[0]):
    #    vi , vj = coords[i, 0], coords[i, 1]
    #    if vi >= 0 and vj >= 0:
    #       color[i] = sns.color_palette("deep")[0]
    #    elif vi >= 0 and vj < 0:
    #       color[i] = sns.color_palette("deep")[1]
    #    elif vi < 0 and vj >= 0:
    #       color[i] = sns.color_palette("deep")[2]
    #    else:
    #       color[i] = sns.color_palette("deep")[3]

    # rn_coords = coords + np.random.randn(106, 3)/ 10
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(coords[:,0], coords[:, 1], coords[:, 2], c=color)
    # for i in np.arange(106):
    #   if NET.labels[i] in ["v1pcuf", "gu", "v3fplf", "tead"]:
    #     ax.text(rn_coords[i, 0], rn_coords[i, 1], rn_coords[i, 2], NET.labels[i], size=10, c="red")
    #   else:
    #     ax.text(rn_coords[i, 0], rn_coords[i, 1], rn_coords[i, 2], NET.labels[i], size=10)

    # plt.show()

    # from itertools import combinations
    # from scipy.special import comb

    # N = 1000

    # max_D2 = np.zeros(N)

    # for kk in range(N):

    #   model = MDS(n_components=3, dissimilarity="precomputed", max_iter=1000)
    #   coords = model.fit_transform(D)

    #   x_axis = coords[x, :] - coords[y, :]
    #   x_axis = x_axis / np.linalg.norm(x_axis)

    #   y_axis = np.zeros((comb(D.shape[0], 2).astype(int), 2))
    #   for k, ij in enumerate(combinations(np.arange(D.shape[0]), 2)):
    #       i, j = ij
    #       y_axis[k, 0] = np.abs(np.dot(x_axis , (coords[i, :] - coords[j, :]) / np.linalg.norm(coords[i, :] - coords[j, :])))
    #       y_axis[k, 1] = D[i, j]

    #   order = np.argsort(y_axis[:, 0])
    #   y_axis = y_axis[order, :]
    #   max_D2[kk] = y_axis[0, 1]

    # from collections import Counter

    # cD2 = Counter(max_D2)
    # print(cD2)
    # max_D2 = list(cD2.keys())[0]
    # d2x, d2y = np.where(D == max_D2)[0]

    # print(NET.labels[x], NET.labels[y])
    # print(NET.labels[d2x], NET.labels[d2y])



    

