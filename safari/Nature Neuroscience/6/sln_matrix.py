import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mlp

from various.network_tools import *

def sln_matrix_check_BB(data_sln : dict, iterations : int, ax : plt.Axes, on=True):
      
    if isinstance(data_sln["1"], pd.DataFrame):
      data_sln_one = data_sln["1"]
    else:
      raise ValueError("Corr SLN value must be a pandas dataframe")
          
    data_sln_one = data_sln_one.pivot(
      index="source_cover", columns="target_cover", values="correlation"
    )

    data_sln_one = data_sln_one.to_numpy()

    if isinstance(data_sln["0"], pd.DataFrame):
      data_sln_zero = data_sln["0"]
      data_sln_zero = data_sln_zero.loc[data_sln_zero["key"] == "conf"]
    else:
      raise ValueError("Corr SLN value must be a pandas dataframe")

    Z = data_sln_one.shape[0]
    L = iterations

    membership_matrix = np.arange(Z**2).reshape(Z, Z)
    xlabel = "Correlation"
    data = pd.DataFrame()

    for i in np.arange(Z):
      for j in np.arange(Z):
          data = pd.concat(
            [
              data,
              pd.DataFrame(
                {
                  "group" : [f"{membership_matrix[i,j]}"] * L,
                  xlabel :  data_sln_zero["correlation"].loc[
                    (data_sln_zero["source_cover"] == i+1) &
                    (data_sln_zero["target_cover"] == j+1)
                  ]
                }
              )
            ], ignore_index=True
          )

    from scipy.stats import ttest_1samp
    sln_significance = np.array([""]* Z**2, dtype="<U21").reshape(Z,Z)

    for ix in np.arange(Z):
      for iy in np.arange(Z):
        x = data[xlabel].loc[data["group"]  == f"{membership_matrix[ix, iy]}"].to_numpy()
        _, pval = ttest_1samp(x, data_sln_one[ix, iy], alternative="less")
        sln_significance[ix, iy] = pvalue2asterisks(pval)
        pval = pvalue2asterisks(pval)

    annotate_sln = np.array([""]*Z**2, dtype="<U21")
    for i, (av, pval) in enumerate(zip(data_sln_one.ravel(), sln_significance.ravel())):
      annotate_sln[i] = f"{av:.2f}\n{pval}"

    annotate_sln = annotate_sln.reshape(Z, Z)

    cmp = sns.color_palette("deep")
    cmap = mlp.colors.LinearSegmentedColormap.from_list("", [cmp[0], "#ffffff", cmp[1]])

    sns.heatmap(
      data_sln_one,
      annot=annotate_sln,
      annot_kws= {"fontsize" : 9},
      fmt="", 
      cmap=cmap,
      alpha=0.7,
      cbar_kws={"label" : "Correlation"},
      ax=ax
    )

    xlabels = ax.get_xticklabels()
    xlabels = [f"C{int(i.get_text())+1}" for i in xlabels]
    ax.set_xticklabels(xlabels)

    ylabels = ax.get_yticklabels()
    ylabels = [f"C{int(i.get_text())+1}" for i in ylabels]
    ax.set_yticklabels(ylabels)