# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Any
import os
# Personal libs ----
from modules.sign.hierarmerge import Hierarchy as signHierarchy
from modules.hierarmerge import Hierarchy
from various.network_tools import *
from modules.flatmap import FLATMAP

class Plot_H:
  def __init__(self, NET, H) -> None:
    ## Attributes ----
    self.linkage = H.linkage
    self.BH = H.BH
    self.Z = H.Z
    self.H = H.H
    self.A = H.A
    self.nonzero = H.nonzero
    self.dA = H.dA
    self.nodes = H.nodes
    self.mode = H.mode
    self.leaves = H.leaves
    self.index = H.index
    self.entropy = H.entropy
    self.R = H.R
    self.source_sim_matrix = H.source_sim_matrix
    self.target_sim_matrix = H.target_sim_matrix
    # Net ----
    self.path = H.plot_path
    self.labels = NET.struct_labels
    # Get regions and colors ----
    self.colregion = H.colregion
    self.colregion.get_regions()

  def sln_matrix(self, data, cover : dict, cbarlabel="SLN", on=True):
    if on:
      print(">>> Print SLN matrix")
      Z = len(cover.keys())
      membership_matrix = np.arange(Z**2).reshape(Z, Z)

      average_sln_membership = np.zeros((Z,Z))

      for zi in np.arange(Z):
        for zj in np.arange(Z):
          x = data["SLN"].loc[data["group"] == membership_matrix[zi, zj].astype(int).astype(str)]
          average_sln_membership[zi, zj] = np.mean(x)

      import matplotlib
      cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b20000","#cca3ff","#0047AB"])

      sns.heatmap(
        average_sln_membership,
        annot=average_sln_membership,
        cmap=cmap,
        alpha=0.7,
        center=0.5,
        cbar_kws={'label': cbarlabel}
      ) 

      xlabels = plt.gca().get_xticklabels()
      xlabels = [f"C{i.get_text()}" for i in xlabels]
      plt.gca().set_xticklabels(xlabels)

      ylabels = plt.gca().get_yticklabels()
      ylabels = [f"C{i.get_text()}" for i in ylabels]
      plt.gca().set_yticklabels(ylabels)

      plt.gcf().tight_layout()

      # from pathlib import Path
      Path(f"{self.path}/sln").mkdir(parents=True, exist_ok=True)
      
      plt.savefig(
        f"{self.path}/sln/sln_clusters.svg",
        transparent=True
      )

      plt.close()
    else:
      print(">>> No SLN matrix")

  def sln_trace_hist(self, data, cover : dict, on=True):
    if on:
      print(">>> Plot SLN trace histogram")
      Z = len(cover.keys())
      membership_matrix = np.arange(Z**2).reshape(Z, Z)

      trace_data_numeric = np.diag(membership_matrix).astype(int).astype(str)
      trace_data = data.loc[np.isin(data["group"], trace_data_numeric)]

      for new, old in enumerate(trace_data_numeric):
        trace_data["group"].loc[trace_data["group"] == old] = str(new)

      g=sns.FacetGrid(
        data=trace_data,
        col="group",
        col_wrap=Z // 2 + 1,
      )

      g.map_dataframe(
        sns.histplot,
        x="SLN",
        stat="density"
      )


      plt.gcf().tight_layout()

      # from pathlib import Path
      Path(f"{self.path}/sln").mkdir(parents=True, exist_ok=True)

      plt.savefig(
        f"{self.path}/sln/sln_trace_hist.svg",
        transparent=True
      )
      plt.close()
    else:
      print(">>> No SLN versus H2 @ first merge")

  def sln_trace(self, data, cover : dict, ylabel="ODR", on=True):
    if on:
      print(">>> Plot SLN-H2 @ first merge")
      Z = len(cover)
      membership_matrix = np.arange(Z**2).reshape(Z, Z)

      trace_data_numeric = np.diag(membership_matrix).astype(int).astype(str)
      trace_data = data.loc[np.isin(data["group"], trace_data_numeric)]

      for new, old in enumerate(trace_data_numeric):
        trace_data["group"].loc[trace_data["group"] == old] = str(new)
    
      trace_data[ylabel] = trace_data["SLN"]
      trace_data["group"] = pd.Categorical(trace_data["group"], np.arange(Z).astype(str))

      if Z > 1:
        g=sns.FacetGrid(
          data=trace_data,
          col="group",
          col_wrap=Z // 2 + 1,
        )

        g.map_dataframe(
          sns.scatterplot,
          x=fr"$H^{2}_i-H^{2}_j$ @ first merge" ,
          y=ylabel
        )

        from scipy.stats import pearsonr

        for ax in g.axes.flatten():
          title = ax.get_title().split(" = ")[-1]
          hax = trace_data[fr"$H^{2}_i-H^{2}_j$ @ first merge" ].loc[trace_data["group"] == title]
          bax = trace_data[ylabel].loc[trace_data["group"] == title]
          r, pval = pearsonr(hax, bax)
          pval = pvalue2asterisks(pval)
          ax.set_title(ax.get_title() + "\n" + fr"$\rho = {r:.2f}$ ({pval})")

        g.add_legend()
      elif Z == 1:
        from scipy.stats import pearsonr

        sns.scatterplot(
          data=trace_data,
          x=fr"$H^{2}_i-H^{2}_j$ @ first merge",
          y=ylabel,
          s=30
        )
        ax= plt.gca()
        hax = trace_data[fr"$H^{2}_i-H^{2}_j$ @ first merge" ]
        bax = trace_data[ylabel]
        r, pval = pearsonr(hax, bax)
        p_val_trans = -np.floor(-np.log10(pval)).astype(int)
        if p_val_trans == 0:
          ax.set_title(ax.get_title() + "\n" + fr"$\rho = {r:.2f}$ with $p=n.s.$")
        else:
          ax.set_title(ax.get_title() + "\n" + fr"$\rho = {r:.2f}$ with $p<1E{p_val_trans}$")

      plt.gcf().tight_layout()

      # from pathlib import Path
      Path(f"{self.path}/sln").mkdir(parents=True, exist_ok=True)

      plt.savefig(
        f"{self.path}/sln/{ylabel}_h2@fistmerge.svg",
        transparent=True
      )
      plt.close()
    else:
      print(">>> No SLN versus H2 @ first merge")


  def sln_offdiagonal(self, data, cover : dict, on=True):
    if on:
      print(">>> Plot SLN-H2 @ first merge | no trace")
      Z = len(cover.keys())
      membership_matrix = np.arange(Z**2).reshape(Z, Z)

      # _, ax = plt.subplots(1,1)
      # ax.axvline()

      notrace_data_numeric = np.diag(membership_matrix)
      notrace_data_numeric = np.array(
        [n for n in np.arange(Z**2) if n not in notrace_data_numeric]
        ).astype(int).astype(str)
      trace_data = data.loc[np.isin(data["group"], notrace_data_numeric)]

      for i in np.arange(Z):
        for j in np.arange(Z):
          if i == j : continue
          trace_data["group"].loc[trace_data["group"] == membership_matrix[i, j].astype(int).astype(str)] = f"{i}|{j}"

      g=sns.FacetGrid(
        data=trace_data,
        col="group",
        col_wrap=Z // 2 + 1,
      )

      g.map_dataframe(
        sns.scatterplot,
        x=fr"$H^{2}_i-H^{2}_j$ @ first merge" ,
        y="SLN"
      )

      from scipy.stats import pearsonr

      for ax in g.axes.flatten():
        title = ax.get_title().split(" = ")[-1]
        hax = trace_data[fr"$H^{2}_i-H^{2}_j$ @ first merge" ].loc[trace_data["group"] == title]
        bax = trace_data["SLN"].loc[trace_data["group"] == title]
        r, pval = pearsonr(hax, bax)
        p_val_trans = -np.floor(-np.log10(pval)).astype(int)
        if p_val_trans == 0:
          ax.set_title(ax.get_title() + "\n" + fr"$\rho = {r:.2f}$ with $p=n.s.$")
        else:
          ax.set_title(ax.get_title() + "\n" + fr"$\rho = {r:.2f}$ with $p<1E{p_val_trans}$")

      g.add_legend()

      plt.gcf().tight_layout()

      # from pathlib import Path
      Path(f"{self.path}/sln").mkdir(parents=True, exist_ok=True)

      plt.savefig(
        f"{self.path}/sln/sln_h2@fistmerge_off.svg",
        transparent=True
      )
      plt.close()
    else:
      print(">>> No SLN versus H2 @ first merge")

  def Mu_plotly(self, on=True, **kwargs):
    if on:
      import plotly.express as px
      print("Visualize MU with plotly!!!")
      from pandas import DataFrame, concat
      # Create Data ----
      dF = DataFrame()
      # Concatenate over n and beta ----
      for i in np.arange(len(self.BH)):
        dF = concat([dF, self.BH[i]])
      fig = px.line(
        dF,
        x = "K",
        y = "mu",
        color = "beta",
        facet_col = "alpha",
        facet_col_wrap = 2,
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path,  "Features")
      # Crate path ----
      Path(plot_path).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "Mu_{}.html".format(self.linkage)
        )
      )
    else:
      print("No Mu with plotly")
    
  def plot_newick_R(self, tree_newick, nodes, root_position = 0, threshold=False, weighted=False, on=True):
    if on:
      print("Plot tree in Newick format from R!!!")
      import subprocess
      # Arrange path ----
      plot_path = join(self.path, "NEWICK")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      if not weighted:
        if not threshold:
          subprocess.run(["Rscript", "R/plot_newick_tree.R", tree_newick, join(plot_path, "tree_newick.png"), f"{nodes}", "0"])
        else:
          subprocess.run(["Rscript", "R/plot_newick_tree.R", tree_newick, join(plot_path, "tree_newick.png"), f"{nodes}", f"{threshold}"])
      else:
        subprocess.run(["Rscript", "R/plot_newick_tree_H.R", tree_newick, join(plot_path, "tree_newick_H"), f"{root_position}"])
    else:
      print("No tree in Newick format")
  
  def plot_newick_R_PIC(self, tree_newick, picture_path, on=True):
    if on:
      print("Plot tree in Newick format from R!!!")
      import subprocess
      # Arrange path ----
      plot_path = join(self.path, "NEWICK")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      subprocess.run(["Rscript", "R/plot_newick_tree_PIC.R", tree_newick, join(plot_path, "tree_newick_pic.png"), picture_path])
    else:
      print("No tree in Newick format")

  def plot_measurements_mu(self, on=False, **kwargs):
    if on:
      print("Plot Mu iterations")
      from pandas import DataFrame, concat
      # Create Data ----
      dF = DataFrame()
      # Concatenate over n and beta ----
      for i in np.arange(len(self.BH)):
        dF = concat(
          [
            dF,
            self.BH[i]
          ], ignore_index=True
        )
      dF.alpha = dF.alpha.to_numpy().astype(str)
      dF.beta = dF.beta.to_numpy().astype(str)
      # Create figure ----
      g = sns.FacetGrid(
        dF, col="alpha",
        sharey=False
      )
      g.map_dataframe(
        sns.lineplot,
        x="K",
        y="mu",
        hue="beta"
      ).set(xscale="log")
      g.add_legend()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "Mu_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No Mu iterations")

  def Entropy_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize Entropy with plotly!!!")
      # Create data ----
      dim = self.link_entropy.shape[1]
      k = np.arange(dim, 0, -1)
      data = pd.DataFrame(
        {
        "K" : np.hstack([k, k]),
        "S" : self.link_entropy.ravel(),
        "c" : ["Sh"] * dim + ["Sv"] * dim
        }
      )
      # Create figure ----
      fig = px.line(
        data,
        x = "K",
        y = "S",
        color = "c",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "Entropy_{}.html".format(self.linkage)
        )
      )
    else:
      print("No Entropy with plotly")

  def Entropy_H_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize Entropy with plotly!!!")
      # Create data ----
      dim = self.link_entropy_H.shape[1]
      k = np.arange(dim, 0, -1)
      data = pd.DataFrame(
        {
        "K" : np.hstack([k, k]),
        "S" : self.link_entropy_H.ravel(),
        "c" : ["Sh"] * dim + ["Sv"] * dim
        }
      )
      # Create figure ----
      fig = px.line(
        data,
        x = "K",
        y = "S",
        color = "c",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "Entropy_H_{}.html".format(self.linkage)
        )
      )
    else:
      print("No Entropy with plotly")

  def plot_measurements_Entropy(self, on=True):
    if on:
      print("Visualize Entropy iterations!!!")
      # Create data ----
      dim = self.entropy[0].shape[1]
      print(f"Levels node hierarchy: {dim}")
      data = pd.DataFrame(
        {
          "S" : np.hstack([self.entropy[0].ravel(), self.entropy[1].ravel()]),
          "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
          "c" : ["node_hierarchy"] * 2 * dim + ["node_hierarchy_H"] * 2 * dim,
          "level" : list(np.arange(dim, 0, -1)) * 4
        }
      )
      dim = self.entropy[2].shape[1]
      print(f"Levels link hierarchy: {dim}")
      data = pd.concat(
        [
          data,
          pd.DataFrame(
            {
              "S" : np.hstack([self.entropy[2].ravel(), self.entropy[3].ravel()]),
              "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
              "c" : ["link_hierarchy"] * 2 * dim + ["link_hierarchy_H"] * 2 * dim,
              "level" : list(np.arange(dim, 0, -1)) * 4
            }
          )
        ], ignore_index=True
      )
      mx = data.iloc[
        data.groupby(["c", "dir"])["S"].transform("idxmax").drop_duplicates(keep="first").to_numpy()
      ].sort_values("c", ascending=False)
      print(mx)
      # Create figure ----
      g = sns.FacetGrid(
        data=data,
        col = "c",
        hue = "dir",
        col_wrap=2,
        sharex=False,
        sharey=False
      )
      g.map_dataframe(
        sns.lineplot,
        x="level",
        y="S"
      )#.set(xscale="log")
      g.add_legend()
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      plt.savefig(
        join(
          plot_path, "Entropy_levels.png"
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No Entropy iterations")

  def D_plotly(self, on=True, **kwargs):
    if on:
      import plotly.express as px
      print("Visualize D with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "D",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "D_{}.html".format(self.linkage)
        )
      )
    else:
      print("No D with plotly")

  def plot_measurements_D(self, on=False, **kwargs):
    if on:
      print("Plot D iterations")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="height",
        y="D",
        estimator="max",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.ylabel(r"$\bar{\delta'}$")
      # plt.ylabel("D")
      plt.xlabel(r"$H^{2}$")
      # plt.xlabel(r"$\mathcal{D}_{JAC}$")
      # print(self.BH[0].loc[self.BH[0]["D"] == np.nanmax(self.BH[0]["D"])])
      # print(self.BH[0].loc[self.BH[0]["D"] > 0.6])
      h = self.BH[0]["height"].loc[self.BH[0]["D"] == np.nanmax(self.BH[0]["D"])].to_numpy()
      h = h[-1]
      plt.axvline(x=h, linestyle="--", color="gray")

      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "D_h_H2.png"
        ),
        # transparent=True
        dpi=300
      )
      plt.close()
    else: print("No D iterations")

  def plot_measurements_S(self, on=False, **kwargs):
    if on:
      print("Plot S iterations")
      sns.set_style("whitegrid")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="height",
        y="S",
        ax=ax
      )
      hmax = self.BH[0]["height"].loc[self.BH[0]["S"] == np.max(self.BH[0]["S"])].to_numpy()[0]
      ax.text(0.1, 0.7, f"best: {hmax:.2f}", transform=ax.transAxes, size=15)
      ax.set_ylabel(r"$S_{L}$")
      ax.set_xlabel("height " + r"($H^{2}$)")
      plt.legend([],[], frameon=False)
      # plt.xscale("log")

      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "S_h.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No S iterations")

  def plot_measurements_SD(self, on=False, **kwargs):
    if on:
      print("Plot SD iterations")
      # Create figure ----
      if "SD" not in self.BH[0].columns:
        self.BH[0]["SD"] = (self.BH[0].D / np.nansum(self.BH[0].D)) * (self.BH[0].S / np.nansum(self.BH[0].S))
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="SD",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "SD_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No SD iterations")

  def X_plotly(self, on=True, **kwargs):
    if on:
      import plotly.express as px
      print("Visualize X with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "X",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "X_{}.html".format(self.linkage)
        )
      )
    else:
      print("No X with plotly")

  def plot_measurements_X(self, on=False, **kwargs):
    if on:
      print("Plot X iterations")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="X",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "X_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No X iterations")
  
  def order_parameter_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize the order parameter with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "m",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "mt_{}.html".format(self.linkage)
        )
      )
    else:
      print("No the order parameter with plotly")

  def plot_measurements_order_parameter(self, on=False, **kwargs):
    if on:
      print("Plot order parameter iterations")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="m",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "order_parameter_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No order parameter iterations")
  
  def susceptibility_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize xm with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "xm",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "xmt_{}.html".format(self.linkage)
        )
      )
    else:
      print("No xm(t) with plotly")

  def plot_measurements_susceptibility(self, on=False, **kwargs):
    if on:
      print("Plot susceptibility iterations")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="xm",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "xm_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No susceptibility iterations")

  def ntrees_plotly(self, on=True):
    if on:
      import plotly.express as px
      print("Visualize ntree with plotly!!!")
      fig = px.line(
        self.BH[0],
        x = "K",
        y = "ntrees",
        markers = True,
        log_x = True
      )
      fig.update_traces(
        patch={
          "line" : {
            "dash" : "dot"
          }
        }
      )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      fig.write_html(
        os.path.join(
          plot_path, "ntrees_{}.html".format(self.linkage)
        )
      )
    else:
      print("No ntrees with plotly")

  def plot_measurements_ntrees(self, on=False, **kwargs):
    if on:
      print("Plot ntrees iterations")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="ntrees",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "ntrees_logK.png"
        ),
        dpi=300
      )
      plt.close()
    else: print("No ntrees iterations")
  
  def core_dendrogram(self, R : list, score="", cmap_name="hls", leaf_font_size=20, remove_labels=False, on=False, **kwargs):
    if on:
      print("Visualize node-community dendrogram!!!")
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      from scipy.cluster import hierarchy
      import matplotlib.colors
      # Create figure ----
      for r in R:
        if r == 1:
          r += 1
          sname = "fake"
        else: sname = ""
        #
        partition = hierarchy.cut_tree(self.Z, r).ravel()
        new_partition = skim_partition(partition)
        unique_clusters_id = np.unique(new_partition)
        cm = sns.color_palette(cmap_name, len(unique_clusters_id))
        # dlf_col = "#808080"
        dlf_col = "#808080"
        ##
        D_leaf_colors = {}
        for i, _ in enumerate(self.colregion.labels[:self.nodes]):
          if new_partition[i] != -1:
            D_leaf_colors[i] = matplotlib.colors.to_hex(cm[new_partition[i]])
          else: D_leaf_colors[i] = dlf_col
        ##
        link_cols = {}
        for i, i12 in enumerate(self.Z[:,:2].astype(int)):
          c1, c2 = (link_cols[x] if x > len(self.Z) else D_leaf_colors[x]
            for x in i12)
          link_cols[i+1+len(self.Z)] = c1 if c1 == c2 else dlf_col
        # sns.set_style("white")
        # plt.style.use("dark_background")
        # sns.set_context("talk")
        fig, ax = plt.subplots(1, 1)
        ax.grid(False)
        if not remove_labels:
          hierarchy.dendrogram(
            self.Z,
            labels=self.colregion.labels[:self.nodes],
            color_threshold=self.Z[self.nodes - r, 2],
            link_color_func = lambda k: link_cols[k],
            leaf_rotation=90, leaf_font_size=leaf_font_size, **kwargs
          )
        else:
          hierarchy.dendrogram(
            self.Z,
            no_labels=True,
            color_threshold=self.Z[self.nodes - r, 2],
            link_color_func = lambda k: link_cols[k]
          )
        fig.set_figwidth(10)
        fig.set_figheight(7)
        plt.ylabel("Height " + r"$(H^{2})$")
        sns.despine()
        # plt.show()
        # Save plot ----
        plt.savefig(
          os.path.join(
            plot_path, "core_dendrogram_{}_{}{}{}.svg".format(self.linkage, r, score, sname)
          ),
          # dpi=300,
          # transparent=True
        )
        plt.close()
        sns.set_theme()
    else:
      print("No node-community dendrogram")

  def heatmap_pure(self, r, R, score="", linewidth=1.5, on=True, **kwargs):
    if on:
      print("Visualize pure logFLN heatmap!!!")
      if "labels" in kwargs.keys():
        ids = kwargs["labels"]
        I, fq = sort_by_size(ids, self.nodes)
      else:
        I = np.arange(self.nodes, dtype=int)
        fq = {}
      # Transform FLNs ----
      W = R.copy()
      nonzero = (W != 0)
      W[~nonzero] = np.nan
      W = W[I, :][:, I]
      # Configure labels ----
      labels = self.colregion.labels[I]
      labels = [str(r) for r in labels]
      rlabels = [
        str(r) for r in self.colregion.regions[
          "AREA"
        ]
      ]
      colors = self.colregion.regions.loc[
        match(
          labels,
          rlabels
        ),
        "COLOR"
      ].to_numpy()
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      fig.set_figwidth(19)
      fig.set_figheight(15 * W.shape[0]/ self.nodes)
      sns.heatmap(
        W,
        cmap=sns.color_palette("viridis", as_cmap=True),
        xticklabels=labels[:self.nodes],
        yticklabels=labels,
        ax = ax
      )
      if "font_size" in kwargs.keys():
        if kwargs["font_size"] > 0:
          ax.set_xticklabels(
            ax.get_xmajorticklabels(), fontsize = kwargs["font_size"]
          )
          ax.set_yticklabels(
            ax.get_ymajorticklabels(), fontsize = kwargs["font_size"]
          )
      # Setting labels colors ----
      [t.set_color(i) for i,t in
        zip(
          colors,
          ax.xaxis.get_ticklabels()
        )
      ]
      [t.set_color(i) for i,t in
        zip(
          colors,
          ax.yaxis.get_ticklabels()
        )
      ]
      # Add black lines ----
      if "labels" in kwargs.keys():
        c = 0
        for key in fq:
          c += fq[key]
          if c < self.nodes:
            ax.vlines(
              c, ymin=0, ymax=self.nodes,
              linewidth=linewidth,
              colors=["#C70039"]
            )
            ax.hlines(
              c, xmin=0, xmax=self.nodes,
              linewidth=linewidth,
              colors=["#C70039"]
            )
     # Arrange path ----
      plot_path = os.path.join(self.path, "Heatmap_pure")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "heatmap{}_{}.png".format(score, r)
        ),
        dpi = 300
      )
      plt.close()
    else:
      print("No pure logFLN heat map")

  def heatmap_dendro(self, r, R, score="", cmap="viridis", font_color = None, center=None, linewidth=1.5, on=True, **kwargs):
    if on:
      print("Plot heatmap structure!!!")
      plt.box()
      # Transform FLNs ----
      W = R.copy()
      W[W == 0] = np.nan
      W[W == -np.Inf] = np.nan
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(
        hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]
      ).astype(int)
      memberships = hierarchy.cut_tree(self.Z, r).ravel()
      memberships = skim_partition(memberships)[den_order]
      C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      D = np.where(memberships == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
      W = W[den_order, :][:, den_order]
      # Configure labels ----
      labels = self.colregion.labels
      labels =  np.char.lower(labels[den_order].astype(str))
      rlabels = [str(re) for re in self.colregion.regions["AREA"]]
      colors = self.colregion.regions.loc[
        match(
          labels,
          rlabels
        ),
        "COLOR"
      ].to_numpy()
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      fig.set_figwidth(18)
      fig.set_figheight(15)
      sns.heatmap(
        W,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        center=center,
        ax = ax
      )
      if "font_size" in kwargs.keys():
        if kwargs["font_size"] > 0:
          ax.set_xticklabels(
            ax.get_xmajorticklabels(), fontsize = kwargs["font_size"]
          )
          ax.set_yticklabels(
            ax.get_ymajorticklabels(), fontsize = kwargs["font_size"]
          )
      for c in C:
        ax.vlines(
          c, ymin=0, ymax=self.nodes,
          linewidth=linewidth,
          colors=["#C70039"]
        )
        ax.hlines(
          c, xmin=0, xmax=self.nodes,
          linewidth=linewidth,
          colors=["#C70039"]
        )
      # Setting labels colors ----
      [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
      [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]

      if font_color:
        [t.set_color(font_color) for t in ax.xaxis.get_ticklabels()]
        [t.set_color(font_color) for t in ax.yaxis.get_ticklabels()]

      plt.xticks(rotation=90)
      plt.yticks(rotation=0)
      plt.ylabel("Source")
      plt.xlabel("Target")
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Heatmap_single"
      )
      # Crate path ----
      Path(
        plot_path
    ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, f"dendrogram_order_{r}{score}.svg"
        ),
        # dpi = 300
      )
      plt.close()
    else:
      print("No heatmap structure")

  def heatmaply_dendro(self, r, R, score="", cmap="viridis", font_color = None, center=None, linewidth=1.5, on=True, **kwargs):
    if on:
      print("Plot heatmaply structure!!!")
      import plotly.express as px
      import plotly.graph_objects as go
      # Transform FLNs ----
      W = R.copy()
      W[W == 0] = np.nan
      W[W == -np.Inf] = np.nan
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(
        hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]
      ).astype(int)
      memberships = hierarchy.cut_tree(self.Z, r).ravel()
      memberships = skim_partition(memberships)[den_order]
      C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      D = np.where(memberships == -1)[0] + 1
      C = np.array(list(set(C).union(set(list(D)))))
      W = W[den_order, :][:, den_order]
      # Configure labels ----
      labels = self.colregion.labels
      labels =  np.char.lower(labels[den_order].astype(str))

      rlabels = [
        str(re) for re in self.colregion.regions[
          "AREA"
        ]
      ]
      colors = self.colregion.regions.loc[
        match(
          labels,
          rlabels
        ),
        "COLOR"
      ].to_numpy()

      # Create figure ----
      dW = pd.DataFrame(
        W, index=labels, columns=labels
      )

      fig = go.Figure(
        data=go.Heatmap(
          z=dW,
          x=labels,
          y=labels,
          colorscale=cmap,
          zmid=0
        )
      )

      for i in np.arange(C.shape[0]-1):
        fig.add_vrect(
          x0=C[i]-0.5, x1=C[i+1]-0.5,
          line_color="#C70039",
          line_width=linewidth
        )
        fig.add_hrect(
          y0=C[i]-0.5, y1=C[i+1]-0.5,
          line_color="#C70039",
          line_width=linewidth
        )

      # fig['layout']['yaxis'].update(autorange = True)
      # fig['layout']['xaxis'].update(autorange = True)

      fig['layout']['yaxis']['autorange'] = "reversed"
      fig.update_layout(
        xaxis_nticks=self.nodes,
        yaxis_nticks=self.nodes,
        template="plotly_dark"
      )

      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Heatmaply_single"
      )
      # Crate path ----
      Path(plot_path).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      fig.write_html(f"{plot_path}/{r}_{score}.html", include_mathjax='cdn')
    else:
      print("No heatmaply structure")

  def lcmap_pure(self, K, score="", cmap_name="husl", linewidth=1.5, undirected=False, on=True, **kwargs):
    if on:
      print("Visualize pure LC memberships!!!")
      # Get elemets from colregion ----
      labels = self.colregion.labels
      regions = self.colregion.regions
      if "labels" in kwargs.keys():
        ids = kwargs["labels"]
        I, fq = sort_by_size(ids, self.nodes)
        flag_fq = True
      else:
        I = np.arange(self.nodes, dtype=int)
        fq = {}
        flag_fq = False
      if "order" in kwargs.keys():
        I = kwargs["order"]
      for k in K:
        # FLN to dataframe and filter FLN = 0 ----
        dFLN = self.dA.copy()
        # Add id with aesthethis ----
        from scipy.cluster.hierarchy import cut_tree
        if not undirected:
          dFLN["id"] =  cut_tree(
            self.H,
            n_clusters = k
          ).ravel()
        else:
          dFLN["id"] = np.tile(cut_tree(
            self.H,
            n_clusters = k
          ).ravel(), 2)
        minus_one_Dc(dFLN, undirected)
        aesthetic_ids(dFLN)
        keff = np.unique(
          dFLN["id"].to_numpy()
        ).shape[0]
        # Transform dFLN to Adj ----
        dFLN = df2adj(dFLN, var="id")
        dFLN = dFLN[I, :][:, I]
        dFLN[dFLN == 0] = np.nan
        dFLN[dFLN > 0] = dFLN[dFLN > 0] - 1
        # dFLN = dFLN.T
        # Configure labels ----
        labels = self.colregion.labels[I]
        labels = [str(r) for r in labels]
        rlabels = [str(r) for r in regions.AREA]
        colors = regions.COLOR.loc[match(labels, rlabels)].to_numpy()
        # Create figure ----
        fig, ax = plt.subplots(1, 1)
        # Check colors with and without trees (-1) ---
        if -1 in dFLN:
          save_colors = sns.color_palette(cmap_name, keff - 1)
          cmap_heatmap = [[]] * keff
          cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
          cmap_heatmap[1:] = save_colors
        else:
          cmap_heatmap = sns.color_palette(cmap_name, keff)
        fig.set_figwidth(19)
        fig.set_figheight(15 * dFLN.shape[0]/ self.nodes)
        sns.heatmap(
          dFLN,
          xticklabels=labels[:self.nodes],
          yticklabels=labels,
          cmap=cmap_heatmap,
          annot_kws = {"size" : 12},
          ax = ax
        )
        # Setting labels colors ----
        [t.set_color(i) for i,t in
          zip(
            colors,
            ax.xaxis.get_ticklabels()
          )
        ]
        [t.set_color(i) for i,t in
          zip(
            colors,
            ax.yaxis.get_ticklabels()
          )
        ]
        # Add black lines ----
        if flag_fq:
          c = 0
          for key in fq:
            c += fq[key]
            if c < self.nodes:
              ax.vlines(
                c, ymin=0, ymax=self.nodes,
                colors=["black"], linewidth = linewidth
              )
              ax.hlines(
                c, xmin=0, xmax=self.nodes,
                colors=["black"], linewidth = linewidth
              )
        # Arrange path ----
        plot_path = os.path.join(self.path, "Matrix_pure")
        # Crate path ----
        Path(
          plot_path
        ).mkdir(exist_ok=True, parents=True)
        # Save plot ----
        plt.savefig(
          os.path.join(
            plot_path, "k_{}{}.png".format(k, score)
          ),
          dpi = 300
        )
        plt.close()
    else:
      print("No pure logFLN heat map")
    
  def plot_histoscatter_linkcomm(self, k, ncol=1, on=False, **kwargs):
    if on:
      print("Plot histoscatter linkcomm for A!!!")
      dA = self.dA.copy()
      # Add id with aesthethis ----
      from scipy.cluster.hierarchy import cut_tree
      dA["id"] =  cut_tree(
        self.H,
        n_clusters = k
      ).reshape(-1)
      minus_one_Dc(dA)
      aesthetic_ids(dA)
      dA.weight = np.log(dA.weight)
      dA.id = dA.id.astype(str)
      dA = dA.sort_values(by="id", ignore_index=True)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dA,
        x="weight",
        hue="id",
        element="poly",
        stat="density",
        alpha=0.5,
        fill=False,
        common_norm=True,
        ax=ax
      )
      ax.set_yscale("log")
      sns.move_legend(ax, "upper left", bbox_to_anchor=(1.03, 1), ncol=ncol)
      fig.tight_layout()
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "histoscatter_fln_link.png"
        ),
        dpi=300
      )
    else: print("No histoscatter linkcomm")

  def lcmap_dendro(
    self, K, R, score="", cmap_name="hls",
    link_com_list=False, remove_labels=False, linewidth=1.5, font_color=None, undirected=False, on=False, **kwargs
  ):
    if on:
      print("Visualize k LCs!!!")
      plt.box()
      # K loop ----
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Matrix_single"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Get labels ----
      labels = self.colregion.labels
      regions = self.colregion.regions
      # FLN to dataframe and filter FLN = 0 ----
      dFLN = self.dA.copy()
      # Add id with aesthethis ----
      from scipy.cluster.hierarchy import cut_tree
      if not undirected:
        dFLN["id"] =  cut_tree(
          self.H,
          n_clusters = K
        ).ravel()
      else:
        # print()
        dFLN["id"] =  np.tile(cut_tree(
          self.H,
          n_clusters = K
        ).ravel(), 2)
      ##
      dFLN["source_label"] = labels[dFLN.source]
      dFLN["target_label"] = labels[dFLN.target]
      # Print link community list ----
      if link_com_list:
        dFLN.sort_values(by="id").to_csv(
          f"{plot_path}/link_community_list_{K}.csv"
        )
      ##
      minus_one_Dc(dFLN, undirected)
      aesthetic_ids(dFLN)

      keff = np.unique(dFLN.id)
      keff = keff.shape[0]
      # Transform dFLN to Adj ----
      dFLN = df2adj(dFLN, var="id")
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]).astype(int)
      memberships = hierarchy.cut_tree(self.Z, R).ravel()
      memberships = skim_partition(memberships)[den_order]
      C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      D = np.where(memberships == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
      #
      dFLN = dFLN[den_order, :][:, den_order]
      dFLN[dFLN == 0] = np.nan
      dFLN[dFLN > 0] = dFLN[dFLN > 0] - 1
      # Configure labels ----
      labels =  np.char.lower(labels[den_order].astype(str))
      rlabels = np.array([str(r).lower() for r in regions.AREA])
      colors = regions.COLOR.loc[match( labels, rlabels)].to_numpy()
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      # Check colors with and without trees (-1) ---
      if -1 in dFLN:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        cmap_heatmap = sns.color_palette(cmap_name, keff)
      if not remove_labels:
        plot = sns.heatmap(
          dFLN,
          cmap=cmap_heatmap,
          xticklabels=labels,
          yticklabels=labels,
          ax=ax
        )
        if "font_size" in kwargs.keys():
          if kwargs["font_size"] > 0:
            plot.set_xticklabels(
              plot.get_xmajorticklabels(), fontsize = kwargs["font_size"]
            )
            plot.set_yticklabels(
              plot.get_ymajorticklabels(), fontsize = kwargs["font_size"]
            )
        # Setting labels colors ----
        [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
        [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]

        if font_color:
          [t.set_color(font_color) for t in ax.xaxis.get_ticklabels()]
          [t.set_color(font_color) for t in ax.yaxis.get_ticklabels()]
      else:
        sns.heatmap(
          dFLN,
          cmap=cmap_heatmap,
          xticklabels=False,
          yticklabels=False,
          ax=ax
        )
      for c in C:
        ax.vlines(
          c, ymin=0, ymax=self.nodes,
          linewidth=linewidth,
          colors=["black"]
        )
        ax.hlines(
          c, xmin=0, xmax=self.nodes,
          linewidth=linewidth,
          colors=["black"]
        )
      fig.set_figwidth(18)
      fig.set_figheight(15)
      plt.ylabel("Source")
      plt.xlabel("Target")
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "{}{}.svg".format(K, score)
        )
      )
      plt.close()
    else:
      print("No k LCs")

  def lcmaply_dendro(
    self, K, R, score="", cmap_name="hls", remove_labels=None, linewidth=1.5, undirected=False, on=False, **kwargs
  ):
    if on:
      print("Visualize k LClys!!!")
      import plotly.graph_objects as go
      # K loop ----
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Matrixly_single"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Get labels ----
      labels = self.colregion.labels
      regions = self.colregion.regions
      # FLN to dataframe and filter FLN = 0 ----
      dFLN = self.dA.copy()
      # Add id with aesthethis ----
      from scipy.cluster.hierarchy import cut_tree
      if not undirected:
        dFLN["id"] =  cut_tree(
          self.H,
          n_clusters = K
        ).ravel()
      else:
        dFLN["id"] =  np.tile(cut_tree(
          self.H,
          n_clusters = K
        ).ravel(), 2)
      ##
        
      dFLN["source_label"] = labels[dFLN.source]
      dFLN["target_label"] = labels[dFLN.target]

      minus_one_Dc(dFLN, undirected)
      aesthetic_ids(dFLN)

      keff = np.sort(np.unique(dFLN.id))
      nkeff = keff.shape[0]
      # Transform dFLN to Adj ----
      dFLN = df2adj(dFLN, var="id")
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]).astype(int)
      memberships = hierarchy.cut_tree(self.Z, R).ravel()
      memberships = skim_partition(memberships)[den_order]
      C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      D = np.where(memberships == -1)[0] + 1
      C = np.array(list(set(C).union(set(list(D)))))
      #
      dFLN = dFLN[den_order, :][:, den_order]
      dFLN[dFLN == 0] = np.nan
      dFLN[dFLN > 0] = dFLN[dFLN > 0] - 1
      # Configure labels ----
      labels =  np.char.lower(labels[den_order].astype(str))
      rlabels = np.array([str(r).lower() for r in regions.AREA])
      colors = regions.COLOR.loc[match( labels, rlabels)].to_numpy()
      # Create figure ----
      # Check colors with and without trees (-1) ---
      if -1 in dFLN:
        save_colors = sns.color_palette(cmap_name, nkeff - 1)
        cmap_heatmap = [[]] * nkeff
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        cmap_heatmap = sns.color_palette(cmap_name, nkeff)
     
      dFLN = pd.DataFrame(
        dFLN, index=labels, columns=labels
      )

      def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
  
      fig = go.Figure(
        data=go.Heatmap(
          z=dFLN,
          x=labels,
          y=labels,
          colorscale=[[k, hex2(c)] for k, c in zip(normalize(keff), cmap_heatmap)],
        )
      )

      if not remove_labels:
        fig.update_layout(
          xaxis_nticks=self.nodes,
          yaxis_nticks=self.nodes
        )
      else:
       fig.update_layout(
          yaxis_showticklabels=False,
          xaxis_showticklabels=False
        ) 

      for i in np.arange(C.shape[0]-1):
        fig.add_vrect(
          x0=C[i]-0.5, x1=C[i+1]-0.5,
          line_color="white",
          line_width=linewidth
        )
        fig.add_hrect(
          y0=C[i]-0.5, y1=C[i+1]-0.5,
          line_color="white",
          line_width=linewidth
        )

      # fig['layout']['yaxis'].update(autorange = True)
      # fig['layout']['xaxis'].update(autorange = True)

      fig['layout']['yaxis']['autorange'] = "reversed"
      fig.update_layout(
        template="plotly_dark"
      )

      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Matrixly_single"
      )
      # Crate path ----
      Path(plot_path).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      fig.write_html(f"{plot_path}/{K}_{score}.html", include_mathjax='cdn')

    else:
      print("No k LClys")

  def threshold_color_map(
    self, R, h, index="Hellinger2", score="", cmap_name="deep", remove_labels=False, linewidth=1.5, on=False, **kwargs
  ):
    if on:
      print("Visualize threhold color map !!!")
      # K loop ----
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "threshold"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Get labels ----
      labels = self.colregion.labels
      regions = self.colregion.regions
      # FLN to dataframe and filter FLN = 0 ----
      if index == "Hellinger2":
        src = 1 - self.source_sim_matrix
        tgt = 1 - self.target_sim_matrix
      else:
        np.seterr(divide='ignore', invalid='ignore')
        src = (1 / self.source_sim_matrix) - 1
        np.seterr(divide='ignore', invalid='ignore')
        tgt = (1 / self.target_sim_matrix) - 1

      xsrc, ysrc = np.where(src < h)
      xtgt, ytgt = np.where(tgt < h)

      SRC = np.zeros(src.shape)
      TGT = np.zeros(tgt.shape)

      SRC[xsrc, ysrc] = 1
      TGT[xtgt, ytgt] = 2

      G = SRC + TGT
      G[G == 0] = np.nan
      # Get nodes ordering ----
      from scipy.cluster import hierarchy
      den_order = np.array(hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]).astype(int)
      memberships = hierarchy.cut_tree(self.Z, R).ravel()
      memberships = skim_partition(memberships)[den_order]
      C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
      D = np.where(memberships == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
      #
      G = G[den_order, :][:, den_order]
      # Configure labels ----
      labels =  np.char.lower(labels[den_order].astype(str))
      rlabels = np.array([str(r).lower() for r in regions.AREA])
      colors = regions.COLOR.loc[match( labels, rlabels)].to_numpy()
      # Create figure ----
      # plt.style.use("dark_background")
      sns.set_context("talk")
      fig, ax = plt.subplots(1, 1)
      # Check colors with and without trees (-1) ---
      cmap_heatmap = sns.color_palette(cmap_name, 3)
      if not remove_labels:
        plot = sns.heatmap(
          G,
          cmap=cmap_heatmap,
          xticklabels=labels,
          yticklabels=labels,
          cbar_kws={'ticks': [1.35, 2, 2.65]},
          # cbar=False,
          ax=ax
        )
        if "font_size" in kwargs.keys():
          if kwargs["font_size"] > 0:
            plot.set_xticklabels( 
              plot.get_xmajorticklabels(), fontsize = kwargs["font_size"]
            )
            plot.set_yticklabels(
              plot.get_ymajorticklabels(), fontsize = kwargs["font_size"]
            )
        # Setting labels colors ----
        [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
        [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]

        source_label = r"$H^{2}_{+}$"
        target_label = r"$H^{2}_{-}$"

        ax.collections[0].colorbar.set_ticklabels([t for t in [source_label, target_label, "both"]])
      else:
        sns.heatmap(
          G,
          cmap=cmap_heatmap,
          xticklabels=False,
          yticklabels=False,
          # cbar=False,
          cbar_kws={'ticks': [1.35, 2, 2.65]},
          ax=ax
        )
      for c in C:
        ax.vlines(
          c, ymin=0, ymax=self.nodes,
          linewidth=linewidth,
          colors=["black"]
        )
        ax.hlines(
          c, xmin=0, xmax=self.nodes,
          linewidth=linewidth,
          colors=["black"]
        )
      fig.set_figwidth(13)
      fig.set_figheight(11)
      plt.ylabel("Areas")
      plt.xlabel("Areas")
      fig.tight_layout()
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "{:.2f}{}.svg".format(h, score)
        ),
        # dpi=300
      )
      plt.close()
    else:
      print("No threshold map")
      
  def flatmap_dendro(self, NET, K : int, R : int, partition, on=True, direction="source", cmap_name="deep", **kwargs):
    if on:
      print("Plot flatmap!!!")
      # Nocs have -1 membership
      ids = partition.copy()
      if len(NET.overlap) > 0:
        ids[match(NET.overlap, self.labels)] = -1

      # Enumerate memberships from 0 to Number of memberships
      u = np.unique(ids)
      u = u[u != -1]
      ids2 = ids.copy()
      for i, ii in enumerate(u): ids[ids2 == ii] = i

      F = FLATMAP(
        NET.nodes, NET.version, NET.struct_labels, self.colregion.regions, NET.plot_path, **kwargs
      )
      
      F.set_para(K, R, ids)
      F.plot_flatmap(direction=direction, cmap_name=cmap_name)
    else:
      print("No flatmap")

  def flatmap_index(self, NET, pivot : str, values : npt.DTypeLike, max_value=None, index_name="Hellinger2", on=True, cmap_name="flare", **kwargs):
    if on:
      print("Plot flatmap index!!!")
      # Start old-new mapping ---
      F = FLATMAP(
        NET.nodes, NET.version, NET.struct_labels, self.colregion.regions, NET.plot_path, **kwargs
      )
      F.plot_flatmap_index(pivot, values, max_value=max_value, index_name=index_name, cmap_name=cmap_name)
    else:
      print("No flatmap index")
  
  def flatmap_regions(self, NET, K : int, R : int, partition, on=True, direction="source", cmap_name="deep", **kwargs):
    if on:
      print("Plot flatmap regions!!!")
      # Start old-new mapping ---
      ids = partition.copy()
      ids[match(NET.overlap, self.labels)] = -1
      u = np.unique(ids)
      u = u[u != -1]
      ids2 = ids.copy()
      for i, ii in enumerate(u): ids[ids2 == ii] = i
      F = FLATMAP(
        NET.nodes, NET.version, NET.struct_labels, self.colregion.regions, NET.plot_path, **kwargs
      )
      
      # ids = np.array([2] * 57)

      F.set_para(K, R, NET.struct_labels[:self.nodes])
      F.plot_regions(direction=direction, cmap_name=cmap_name)
    else:
      print("No flatmap")
  
  def flatmap_labels(self, r, labels, on=True, **kwargs):
    if on:
      print("Plotting average-linkage flatmap!!!")
      labels = aesthetic_ids_vector(labels)
      fmap = FLATMAP(
        self.NET, self.colregion.regions.copy(), **kwargs
      )
      print("Flatmap with labels")
      fmap.set_para(r, r, labels)
      fmap.plot_flatmap()
    else:
      print("No average-linkage flatmap")

  def plot_networx(self, rlabels, cmap_name="husl", figwidth=10, figheight=10, **kwargs):
    print("Draw networkx!!!")
    unique_labels = np.unique(rlabels)
    number_of_communities = unique_labels.shape[0]
    if -1 in unique_labels:
      save_colors = sns.color_palette(cmap_name, number_of_communities - 1)
      color_map = [[]] * number_of_communities
      color_map[0] = [199/ 255.0, 0, 57/ 255.0]
      color_map[1:] = save_colors
    else:
      color_map = sns.color_palette(cmap_name, number_of_communities)
    color_dict = dict()
    for i, lab in enumerate(unique_labels):
      if lab != -1: color_dict[lab] = color_map[i]
      else: color_dict[-1] = "#808080"
    node_colors = [
      color_dict[lab] for lab in rlabels
    ]
    G = nx.from_numpy_array(
      self.A, create_using=nx.DiGraph
    )
    Ainv = self.A.copy()
    Ainv[Ainv != 0] = 1 / Ainv[Ainv != 0]
    Ginv = nx.from_numpy_array(
      Ainv, create_using=nx.DiGraph
    )
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    pos = nx.kamada_kawai_layout(Ginv)
    pos = nx.spring_layout(
      G, pos=pos, iterations=5, seed=212
    )
    nx.draw_networkx(
      G,
      pos=pos,
      node_color=node_colors,
      connectionstyle="arc3,rad=-0.2",
      ax=ax, **kwargs
    )
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)
    plt.show()

  def chrod_diagram(self, k : int, r : int, Cr : npt.NDArray[np.int64], direction="source", on=True):
    if on:
      print("Plot chord diagram!!!")

      hv.extension('bokeh')
      hv.output(size=200)

      links = self.dA.copy()
      links["weight"] = -1/np.log(links["weight"])

      cr = Cr.copy()
      cr[cr == -1] = np.max(cr) + 1

      realCr = np.sort(np.unique(cr))
      realR = realCr.shape[0]
      realLinks = np.zeros((realR, realR))

      crdic = {r : i for i, r in enumerate(realCr)}
      srdic = {i : r for i, r in enumerate(realCr)}
      ncrdic = {i : r for i, r in enumerate(cr)}


      for s, t, w in zip(links.source, links.target, links.weight):
        realLinks[crdic[ncrdic[s]], crdic[ncrdic[t]]] += w

      links = adj2df(realLinks)
      links.columns = ["source", "target", "value"]

      links.source = [srdic[i] for i in links.source]
      links.target = [srdic[i] for i in links.target]

      links.index = np.arange(links.shape[0], dtype=int)
      links["value"] = [int(w) for w in links["value"]]

      nodes = hv.Dataset(pd.DataFrame({"Groups" : realCr}), "index")

      chord = hv.Chord((links, nodes))
      chord.opts(
        opts.Chord(
          cmap="hsv", edge_cmap="hsv", edge_color=dim('source').str(),
          labels='Groups', node_color=dim('index').str()
        )
      )
      chord.opts(width=1000, height=1000, label_text_font_size='30pt')
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Chords"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      hv.save(chord, f'{plot_path}/{direction}.png', fmt='png', dpi=300)
    else:
      print("No Chord diagram")

  def plot_link_communities(self, K, cmap_name="hls", figwidth=10, figheight=10, **kwargs):
    print("Draw networkx link communities!!!")
    dA = self.dA.copy()
    from scipy.cluster.hierarchy import cut_tree
    labels = cut_tree(self.H, K).ravel()
    dA["id"] = labels
    # minus_one_Dc(dA)
    # aesthetic_ids(dA)
    labels = dA.id.to_numpy()
    # labels[labels > 0] = labels[labels > 0] - 1
    unique_labels = np.unique(labels)
    number_of_communities = unique_labels.shape[0]
    if -1 in unique_labels:
      save_colors = sns.color_palette(cmap_name, number_of_communities - 1)
      color_map = [[]] * number_of_communities
      color_map[0] = [199/ 255.0, 0, 57/ 255.0]
      color_map[1:] = save_colors
    else:
      color_map = sns.color_palette(cmap_name, number_of_communities)
    color_dict = dict()
    for i, lab in enumerate(unique_labels):
      if lab != -1: color_dict[lab] = color_map[i]
      else: color_dict[-1] = "#808080"
    edge_colors = [
      color_dict[lab] for lab in labels
    ]
    G = nx.from_numpy_array(
      self.A, create_using=nx.DiGraph
    )
    Ainv = self.A.copy()
    Ainv[Ainv != 0] = 1 / Ainv[Ainv != 0]
    Ginv = nx.from_numpy_array(
      Ainv, create_using=nx.DiGraph
    )
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    pos = nx.kamada_kawai_layout(Ginv)
    pos = nx.spring_layout(
      G, pos=pos, iterations=5, seed=212
    )
    nx.draw_networkx(
      G, pos=pos,
      edge_color=edge_colors,
      connectionstyle="arc3,rad=-0.2",
      ax=ax, **kwargs
    )
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)

    plt.show()

  def plot_network_kk(self, H : Hierarchy, partition, nocs : dict, sizes : dict, labels, ang=0, score="", front_edges=False, log_data=False, font_size=0.1, undirected=False, cmap_name="hls"):
    print("Printing network space")
    # new_partition = skim_partition(partition)
    unique_clusters_id = np.unique(partition)
    keff = len(unique_clusters_id)
    # Generate all the colors in the color map -----
    if -1 in unique_clusters_id:
      save_colors = sns.color_palette(cmap_name, keff - 1)
      cmap_heatmap = [[]] * keff
      cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
      cmap_heatmap[1:] = save_colors
    else:
      save_colors = sns.color_palette(cmap_name, keff)
      cmap_heatmap = [[]] * (keff+1)
      cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
      cmap_heatmap[1:] = save_colors
    # Assign memberships to nodes ----
    if -1 in unique_clusters_id:
      nodes_memberships = {
        k : {"id" : [0] * keff, "size" : [0] * keff} for k in np.arange(len(partition))
      }
    else:
      nodes_memberships = {
        k : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for k in np.arange(len(partition))
      }
    for i, id in enumerate(partition):
      if id == -1: continue
      nodes_memberships[i]["id"][id + 1] = 1
      nodes_memberships[i]["size"][id + 1] = 1
    for i, key in enumerate(nocs.keys()):
      index_key = np.where(labels == key)[0][0]
      for id in nocs[key]:
        if id == -1:
          nodes_memberships[index_key]["id"][0] = 1
          nodes_memberships[index_key]["size"][0] = 1
        else:
          nodes_memberships[index_key]["id"][id + 1] = 1
          nodes_memberships[index_key]["size"][id + 1] = sizes[key][id]
    # Check unassigned ----
    for i in np.arange(len(partition)):
      if np.sum(np.array(nodes_memberships[i]["id"]) != 0) == 0:
        nodes_memberships[i]["id"][0] = 1
        nodes_memberships[i]["size"][0] = 1

    if not log_data:
      A = H.A
    else:
      A = np.log(1 + H.A)

    if not undirected:
      G = nx.DiGraph(A)
    else:
      G = nx.Graph(A, directed=False)
    pos = nx.kamada_kawai_layout(G)
    ang = ang * np.pi/ 180
    rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
    pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
    fig, ax = plt.subplots(1, 1)
    # Create labels ---
    labs = {n : labels[n] for n in G.nodes}
    nx.draw_networkx_labels(G, pos=pos, labels=labs, font_size=font_size, ax=ax)
    if not front_edges:
      if undirected:
        nx.draw_networkx_edges(G, pos=pos, arrows=False, ax=ax)
      else:
        nx.draw_networkx_edges(G, pos=pos, arrows=True, ax=ax)
    for node in G.nodes:
      # print(nodes_memberships[node]["id"])
      # print(nodes_memberships[node]["size"])
      plt.pie(
        [s for s in nodes_memberships[node]["size"] if s != 0], # s.t. all wedges have equal size
        center=pos[node], 
        colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0],
        radius=0.05
      )
    if front_edges:
      if undirected:
        nx.draw_networkx_edges(G, pos=pos, arrows=False, ax=ax)
      else:
        nx.draw_networkx_edges(G, pos=pos, arrows=True, ax=ax)
    array_pos = np.array([list(pos[v]) for v in pos.keys()])
    plt.xlim(-0.1 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.1)
    plt.ylim(-0.1 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.1)
    fig.set_figheight(9)
    fig.set_figwidth(9)
    plt.show()