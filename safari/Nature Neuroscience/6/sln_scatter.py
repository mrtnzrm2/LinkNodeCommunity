import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mlp

from various.network_tools import *

def sln_matrix_shuffle_test_BB(data_sln : dict, iterations : int, ax : plt.Axes):
      if isinstance(data_sln["1"], pd.DataFrame):
        data_sln_one = data_sln["1"].pivot(
          index="source_cover", columns="target_cover", values="correlation"
        )

      else:
        raise ValueError("Corr SLN value must be a pandas dataframe")

      if isinstance(data_sln["0"], pd.DataFrame):
        data_sln_zero = data_sln["0"]
        data_sln_zero = data_sln_zero.loc[data_sln_zero["key"] == "shuffle"]
      else:
        raise ValueError("Corr SLN value must be a pandas dataframe")

      Z = data_sln_one.shape[0]
      L = iterations

      sln_zero_mean_noravel = []
      for i in np.arange(L):
        sln_zero_mean_noravel.append(np.sort(data_sln_zero.loc[data_sln_zero["iter"] == i].pivot(
          index="source_cover", columns="target_cover", values="correlation"
        ).to_numpy().ravel()))
      sln_zero_mean_noravel = np.array(sln_zero_mean_noravel)

      sln_zero_mean = np.nanmean(sln_zero_mean_noravel, axis=0)
      sln_zero_std = np.nanstd(sln_zero_mean_noravel, axis=0)


      data_sln_one = np.sort(data_sln_one.to_numpy().ravel())

      nb_elements = data_sln_one.shape[0]
      elements = np.arange(nb_elements)

      significance_array = np.array([""] * nb_elements, dtype="<U21")

      from scipy.stats import ttest_1samp
      for i in np.arange(Z**2):
          sln_values = sln_zero_mean_noravel[:, i]
          sln_values = sln_values[~ np.isnan(sln_values)]
          _, pval = ttest_1samp(sln_values, data_sln_one[i], alternative="less")
          pval = pvalue2asterisks(pval)
          significance_array[i] = pval

      cmp = sns.color_palette("deep")
      asterisk_height = np.maximum(data_sln_one, sln_zero_mean) + 0.01

      scat_one = ax.scatter(elements, data_sln_one, color=cmp[0], label="data", alpha=0.8, s=5)
# 
      ax.scatter(elements, sln_zero_mean, s=5, color=cmp[1], alpha=0.8, label="random communities")
      scat_zero = ax.errorbar(
         elements, sln_zero_mean, yerr=sln_zero_std,
         ls="None", color=cmp[1], ecolor=cmp[1],
         capsize=3, capthick=1, label="random communities", alpha=0.8
      )

      ax.yaxis.tick_right()

      ax.legend(handles=[scat_one, scat_zero], fontsize=8)

      h_extra = np.zeros(elements.shape)
      # for i in np.arange(1, asterisk_height.shape[0]):
      #   if asterisk_height[i-1]/asterisk_height[i] > 0.95:
      #     if i % 2 == 0:
      #       h_extra[i] += 0.01
      #     else:
      #       h_extra[i] -= 0.01

      [ax.text(e + 0.5, h + 0.1, s, horizontalalignment="center", verticalalignment="center", rotation=90) for e, h, s, r in zip(elements, asterisk_height, significance_array, h_extra)]
      ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.1)
      ax.set_ylabel("Correlation")
      ax.yaxis.set_label_position("right")