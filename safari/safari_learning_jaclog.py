# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.stats import pareto
from scipy.stats import cauchy
from scipy.stats import norm
from scipy.stats import invgamma
import ctools as ct

# Personal libs ----
from modules.hierarmerge import Hierarchy
from various.data_transformations import maps
from networks.structure import MAC57
from various.network_tools import *

# Boolean aliases ----
T = True
F = False

def cos(u, v):
  dotuu = np.dot(u,u)
  dotvv = np.dot(v,v)
  if dotuu > 0 and dotvv > 0:
    return np.dot(u, v) / (np.sqrt(dotuu) * np.sqrt(dotvv))
  else: return 0 


def jacw(u, v):
    n = u.shape[0]
    A = np.vstack([u, v])
    return (np.nansum(np.abs(np.nanmin(A, axis=0))) - np.nansum(np.abs(np.nanmax(A, axis=0)))) / n


def sim_1_1(F, path):
   X = np.array([1, 1])
   N = 25
   Lx = np.linspace(-1, 1, N)
   ZZ = {}
   for key, value in F.items():
      Z = np.zeros((N, N))
      for i, x in enumerate(Lx):
         for j, y in enumerate(Lx):
            if len(F[key]["args"]) == 0:
               Z[i, j] = F[key]["feat"](X, np.array([1+x, 1+y]))
            elif len(F[key]["args"]) == 1:
               Z[i, j] = F[key]["feat"](X, np.array([1+x, 1+y]), F[key]["args"][0])
      ZZ[key] = Z.ravel()

   data = pd.DataFrame()
   for key in ZZ.keys():
      data = pd.concat(
         [
            data,
            pd.DataFrame(
               {
                  "score" : ZZ[key],
                  "label" : [key] * ZZ[key].shape[0],
                  "(x,y)" :  np.arange(ZZ[key].shape[0])
               }
            )
         ], ignore_index=True
      )

   # Derive xtick labels

   loc = np.arange(ZZ["jacp"].shape[0], step=N)

   zz = [f"{l:.2f},-1" for l in Lx]

   # print(data.loc[data.label == "cos"].iloc[loc])

   sns.lineplot(
      data=data,
      x="(x,y)",
      y="score",
      hue="label",
      linewidth=1,
      alpha=0.7
   )
   plt.xticks(loc, zz, rotation=45)
   fig = plt.gcf()
   fig.set_figwidth(12)
   fig.tight_layout()
   plt.savefig(path+"/sim_1_1.png", dpi=300)
   plt.close()

def sim_0_x(F, path):

   N = 100
   Lx = np.linspace(0, 1000, N)
   ZZ = {}
   for key, value in F.items():
      Z = np.zeros((N,))
      for i, x in enumerate(Lx):
            if len(F[key]["args"]) == 0:
               Z[i] = F[key]["feat"](np.array([0, x]), np.array([0, x]))
            elif len(F[key]["args"]) == 1:
               Z[i] = F[key]["feat"](np.array([0, x]), np.array([0, x]), F[key]["args"][0])
      ZZ[key] = Z

   data = pd.DataFrame()
   for key in ZZ.keys():
      data = pd.concat(
         [
            data,
            pd.DataFrame(
               {
                  "score" : ZZ[key],
                  "label" : [key] * ZZ[key].shape[0],
                  "x" :  Lx
               }
            )
         ], ignore_index=True
      )

   sns.lineplot(
      data=data,
      x="x",
      y="score",
      hue="label",
      linewidth=1,
      alpha=0.6
   )
   ax = plt.gca()
   # ax.set_xscale("log")
   # plt.xticks(rotation=90)
   plt.savefig(path+"/sim_0_x.png", dpi=300)
   plt.close()

def sim_x_0_0_x(F, path):

   N = 100
   Lx = np.linspace(0, 1000, N)
   ZZ = {}
   for key, value in F.items():
      Z = np.zeros((N,))
      for i, x in enumerate(Lx):
            if len(F[key]["args"]) == 0:
               Z[i] = F[key]["feat"](np.array([x, 0]), np.array([0, x]))
            elif len(F[key]["args"]) == 1:
               Z[i] = F[key]["feat"](np.array([x, 0]), np.array([0, x]), F[key]["args"][0])
      ZZ[key] = Z

   data = pd.DataFrame()
   for key in ZZ.keys():
      data = pd.concat(
         [
            data,
            pd.DataFrame(
               {
                  "score" : ZZ[key],
                  "label" : [key] * ZZ[key].shape[0],
                  "x" :  Lx
               }
            )
         ], ignore_index=True
      )

   sns.lineplot(
      data=data,
      x="x",
      y="score",
      hue="label",
      linewidth=1,
      alpha=0.7
   )
   ax = plt.gca()
   # ax.set_xscale("log")
   # plt.xticks(rotation=90)
   plt.savefig(path+"/sim_x_0_0_x.png", dpi=300)
   plt.close()

def sim_rot_scl(F, path):

   X = np.array([1, 1])
   N = 25
   theta = np.linspace(-np.pi/4, np.pi/4, N)
   labda = np.hstack([np.linspace(0, 1, 25), np.linspace(1, 100, 25)])
   NL = labda.shape[0]
   ZZ = {}
   R = lambda t: np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
   for key, value in F.items():
      Z = np.zeros((N, NL))
      for i, x in enumerate(theta):
            for j, l in enumerate(labda):
               if len(F[key]["args"]) == 0:
                  Z[i, j] = F[key]["feat"](X, l * np.matmul(R(x), X))
               elif len(F[key]["args"]) == 1:
                  Z[i, j] = F[key]["feat"](X, l * np.matmul(R(x), X), F[key]["args"][0])
      ZZ[key] = Z.ravel()

   data = pd.DataFrame()
   for key in ZZ.keys():
      data = pd.concat(
         [
            data,
            pd.DataFrame(
               {
                  "similarity score" : ZZ[key],
                  "label" : [key] * ZZ[key].shape[0],
                  "pararmeters" :  np.arange(ZZ[key].shape[0])
               }
            )
         ], ignore_index=True
      )

   # Derive tick labels and positions ----

   loc = np.arange(ZZ["jacp"].shape[0], step=labda.shape[0])

   zz = [f"{t:.2f}"+r"$\pi$"+",0" for t in theta/np.pi]

   sns.lineplot(
      data=data,
      x="pararmeters",
      y="similarity score",
      hue="label",
      linewidth=1,
      alpha=0.7
   )
   plt.xticks(loc, zz, rotation=45)
   ax = plt.gca()
   ax.set_xlabel(r"($\theta, \lambda$)")
   fig = plt.gcf()
   fig.set_figwidth(12)
   fig.tight_layout()
   plt.savefig(path+"/sim_rot_lambda.png", dpi=300)
   plt.close()

def hist_sim_theta_lambda(F, path):
   N = 1000
   X = np.array([1] * N)
   N1 = 3
   N2 = 5
   T = 1000
   p1 = np.linspace(0.1, 2, N1)
   p2 = np.linspace(1, 10, N2)

   ZZ = pd.DataFrame()

   for t1 in p1:
      for t2 in p2:
         Z = np.zeros((T, len(F)))
         for i in np.arange(T):
            dr = t1 * norm.rvs(size=N)
            Y = pareto.rvs(t2) * (X + dr)
            Y[Y <= 0] = 0
            for j, key in enumerate(F.keys()):
               Z[i, j] = F[key](X, Y)
         for j, key in enumerate(F.keys()):
            ZZ = pd.concat(
               [
                  ZZ,
                  pd.DataFrame(
                     {
                        "score" : Z[:, j].ravel(),
                        "label" : [key] * T,
                        r"$\sigma$" : [t1] * T,
                        r"$\lambda$" : [t2] * T
                     }
                  )
               ], ignore_index=True
            )

   g = sns.FacetGrid(
      data=ZZ,
      row=r"$\sigma$",
      col=r"$\lambda$",
      hue="label",
      sharex=False,
      sharey=False,
      aspect=1.3,
      height=10
   )

   g.map_dataframe(
      sns.histplot,
      x="score",
      alpha=0.3
   )

   g.add_legend()
   
   plt.savefig(path+"/perturbation.png", dpi=300) 
   plt.close()

def standardized(x):
   return (x - np.mean(x)) / np.std(x)

def Data_sims(path):
   S = {}
   T = {}
   # Declare global variables ----
   linkage = "single"
   nlog10 = T
   lookup = F
   prob = F
   cut = F
   structure = "LN"
   mode = "BETA"
   distance = "tracto16"
   nature = "original"
   imputation_method = ""
   topology = "MIX"
   mapping = "trivial"
   bias = 0.
   alpha = 0.
   version = "57d106"
   __nodes__ = 57
   __inj__ = 57

   sims = ["simple2", "D1_2_2"]
   for index in sims:
      # Load structure ----
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
         alpha = alpha
      )
      # Transform data for analysis ----
      R, lookup, _ = maps[mapping](
         NET.C, nlog10, lookup, prob, b=bias
      )
      H = Hierarchy(
         NET, NET.C, R, NET.D,
         __nodes__, linkage, mode, lookup=lookup, alpha = alpha
      )
      S[index] = adj2df(H.source_sim_matrix)
      S[index] = S[index].loc[S[index].source > S[index].target]
      T[index] = adj2df(H.target_sim_matrix)
      T[index] = T[index].loc[T[index].source > T[index].target]
   # Declare global variables ----
   linkage = "single"
   nlog10 = T
   lookup = F
   prob = T
   cut = F
   structure = "FLN"
   mode = "BETA"
   distance = "tracto16"
   nature = "original"
   imputation_method = ""
   topology = "MIX"
   mapping = "R2"
   index = "jacw"
   bias = 1e-5
   alpha = 1/3
   version = "57d106"
   __nodes__ = 57
   __inj__ = 57
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
         alpha = alpha
   )
   # Transform data for analysis ----
   R, lookup, _ = maps[mapping](
         NET.A, nlog10, lookup, prob, b=bias
   )
   H = Hierarchy(
         NET, NET.A, R, NET.D,
         __nodes__, linkage, mode, lookup=lookup, alpha = alpha
   )
   S[index] = adj2df(H.source_sim_matrix)
   S[index] = S[index].loc[S[index].source > S[index].target]
   T[index] = adj2df(H.target_sim_matrix)
   T[index] = T[index].loc[T[index].source > T[index].target]

   ##

   D = NET.D[:__inj__, :__inj__]
   D = adj2df(D)
   D = D.loc[D.source > D.target]

   #

   G = {
      "simple2" : "jaclog_LN",
      "jacw" : "jacw_FLNe",
      # "D1b" : "D1_LN",
      # "D1" : "D1_LN",
      # "D1_2" : "D1_2_LN",
      "D1_2_2" : "D1_2_2_LN"
   }
   data = pd.DataFrame()

   for i, g in enumerate(G.keys()):
      for j, h in enumerate(G.keys()):
         # if i <= j: continue
         data = pd.concat(
            [
            data,
               pd.DataFrame(
                  {
                     "x" : S[g].weight,
                     "y" : S[h].weight,
                     "x_index" : [G[g]] * D.weight.shape[0],
                     "y_index" : [G[h]] * D.weight.shape[0],
                     "dir" : ["source"] * D.weight.shape[0]
                  }
               )
            ], ignore_index=True
         )
         data = pd.concat(
            [
               data,
               pd.DataFrame(
                  {
                     "x" : T[g].weight,
                     "y" : T[h].weight,
                     "x_index" : [G[g]] * D.weight.shape[0],
                     "y_index" : [G[h]] * D.weight.shape[0],
                     "dir" : ["target"] * D.weight.shape[0]
                  }
               )
            ], ignore_index=True
         )
   
   g = sns.lmplot(
      data=data,
      col="x_index",
      row="y_index",
      hue="dir",
      x="x",
      y="y",
      lowess=True,
      #  scatter=False,
      scatter_kws={"s": 3, "alpha":0.7},
      # line_kws={"linewidth" : 1, "alpha" : 0.6},
      sharex=False,
      sharey=False
   )

   plt.savefig(path+"/sim_sim.png", dpi=300)
   plt.close()

   data = pd.DataFrame()

   for g in G.keys():
      data = pd.concat(
         [
            data,
            pd.DataFrame(
               {
                  "stdr_score" : standardized(S[g].weight),
                  "distance" : D.weight,
                  "index" : [G[g]] * D.weight.shape[0],
                  "dir" : ["source"] * D.weight.shape[0]
               }
            )
         ], ignore_index=True
      )
      data = pd.concat(
         [
            data,
            pd.DataFrame(
               {
                  "stdr_score" : standardized(T[g].weight),
                  "distance" : D.weight,
                  "index" : [G[g]] * D.weight.shape[0],
                  "dir" : ["target"] * D.weight.shape[0]
               }
            )
         ], ignore_index=True
      )
   
   g = sns.lmplot(
      data=data,
      col="dir",
      x="distance",
      y="stdr_score",
      hue="index",
      lowess=True,
      scatter=False,
      #  scatter_kws={"s": 2, "alpha":0.4},
      line_kws={"linewidth" : 1, "alpha" : 0.7}
   )
   
   plt.savefig(path+"/sim_dist.png", dpi=300)
   plt.close()

plot_path = "../plots/TOY/learning_jaclog"

F = {
   # "cos" : {"feat": cos, "args" : []},
   "jacp" : {"feat": ct.jacp, "args" : []},
   "jaclog" : {"feat": ct.jaclog, "args" : []},
   # "jacsqrt" : {"feat": ct.jacsqrt, "args" : []},
   # "D1b" : {"feat": ct.D1b, "args" : []},
   # "D1" : {"feat": ct.D1, "args" : []},
   "D1_2" : {"feat": ct.D1_2_2, "args" : []},
   # "D1_2" : {"feat": ct.D1_2, "args" : []},
   # "D2" : {"feat": ct.D2, "args" : []},
   # "Dinf": {"feat": ct.Dinf, "args" : []},
   # "Dalpha" : {"feat": ct.Dalpha, "args" : [1/3]},
   # "simbin" : ct.simbin
}

sim_1_1(F, plot_path)
sim_0_x(F, plot_path)
sim_rot_scl(F, plot_path)
sim_x_0_0_x(F, plot_path)
# hist_sim_theta_lambda(F, plot_path)
Data_sims(plot_path)

  



      