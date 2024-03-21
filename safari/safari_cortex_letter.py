# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
# plt.style.use("dark_background")
from pathlib import Path
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

def fit_poly(x, y, grid, order, n_boot=1000, seed=None, units=None):
        import seaborn.algorithms as algo
        """Regression using numpy polyfit for higher-order trends."""
        def reg_func(_x, _y):
            return np.polyval(np.polyfit(_x, _y, order), grid)

        yhat = reg_func(x, y)

        yhat_boots = algo.bootstrap(x, y,
                                    func=reg_func,
                                    n_boot=n_boot,
                                    units=units,
                                    seed=seed)
        return yhat, yhat_boots

def omega(plot_path, common_features, mode="ZERO", iterations=500):
  path = "../pickle/RAN/distbase/{}/{}/{}/{}/{}/BIN_12/{}/{}/MIX_Hellinger2_trivial/discovery_7".format(
     common_features["subject"],
     common_features["version"],
     common_features["structure"],
     common_features["distance"],
     common_features["model_distbase"],
     common_features["subfolder"],
     mode
  )
  H_EDR = read_class(path, f"series_{iterations}")
  path = "../pickle/RAN/swaps/{}/{}/{}/{}/{}/{}/{}/MIX_Hellinger2_trivial/discovery_7".format(
     common_features["subject"],
     common_features["version"],
     common_features["structure"],
     common_features["distance"],
     common_features["model_swaps"],
     common_features["subfolder"],
     mode
  )
  H_CONG = read_class(path, f"series_{iterations}")
  edr_data =H_EDR.data
  cong_data = H_CONG.data
  edr_data["score"] = [s.replace("_", "") for s in edr_data["score"]]
  cong_data["score"] = [s.replace("_", "") for s in cong_data["score"]]
  edr_data = edr_data.loc[(edr_data.score == "S") & (edr_data.sim == "OMEGA") & (edr_data.direction == "both")]
  cong_data = cong_data.loc[(cong_data.score == "S") & (cong_data.sim == "OMEGA") & (cong_data.direction == "both")]

  data = pd.DataFrame(
    {
      "omega" : list(edr_data["values"]) + list(cong_data["values"]),
      "model" : ["EDR"] * edr_data.shape[0] + ["Configuration"] * cong_data.shape[0]
    }
  )

  print(data.groupby("model").mean())

  from scipy.stats import ttest_ind, ttest_1samp

  test = ttest_ind(data.omega.loc[data.model == "EDR"], data.omega.loc[data.model == "Configuration"], alternative="greater", equal_var=False)
  test_conf = ttest_1samp(data.omega.loc[data.model == "Configuration"], 0)

  # sns.set_context("talk")
  sns.set_style("whitegrid")
  # plt.style.use("dark_background")
  sns.histplot(
    data=data,
    x="omega",
    hue="model",
    stat="density",
    palette="deep",
    common_bins=False,
    common_norm=False
  )

  ax = plt.gca()

  if  not np.isnan(test.pvalue): 
    if test.pvalue > 0.05:
      a = "n.s."
    elif test.pvalue <= 0.05 and test.pvalue > 0.001:
      a = "*" 
    elif test.pvalue <= 0.001 and test.pvalue > 0.0001:
      a = "**" 
    else:
      a = "***"
  else:
    a = "nan"

  width_min = ax.get_xbound()[0]
  width_max = ax.get_xbound()[1]


  omega_mean = np.nanmean(data.omega.loc[data.model == "EDR"])
  omega_t = (omega_mean - width_min) / (width_max - width_min)
  ax.text(omega_t, 1.005, f"{omega_mean:.2f}\n{a}", transform=ax.transAxes, horizontalalignment="center")
  plt.axvline(omega_mean, linestyle="--", color="r")

  if  not np.isnan(test_conf.pvalue): 
    if test_conf.pvalue > 0.05:
      a = "ns"
    elif test_conf.pvalue <= 0.05 and test_conf.pvalue > 0.001:
      a = "*" 
    elif test_conf.pvalue <= 0.001 and test_conf.pvalue > 0.0001:
      a = "**" 
    else:
      a = "***"
  else:
    a = "nan" 

  omega_mean = np.nanmean(data.omega.loc[data.model == "Configuration"])
  omega_t = (omega_mean - width_min) / (width_max - width_min)
  ax.text(omega_t, 1.005, f"{omega_mean:.2f}\n{a}", transform=ax.transAxes, horizontalalignment="center")
  plt.axvline(omega_mean, linestyle="--", color="r")

  # ax.set_xlabel(r"Omega index $(\omega)$")
  ax.set_xlabel(r"$\omega$")

  fig = plt.gcf()
  fig.set_figheight(7)
  fig.set_figwidth(10)
  fig.tight_layout()
  
  plot_path = plot_path + "/cortex_letter/"
  # print(plot_path)
  Path(plot_path).mkdir(exist_ok=True, parents=True)
  plt.savefig(
    join(
      plot_path, "omega.svg"
    ),
    dpi=300, transparent=T
  )
  plt.close()

def entropy(plot_path, mode="ALPHA", iterations=500):
  path = f"../pickle/RAN/distbase/MAC/57d106/FLN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/{mode}/MIX_Hellinger2_trivial/b_0.0/discovery_7"
  H_EDR = read_class(path, f"series_{iterations}")
  path = f"../pickle/RAN/swaps/MAC/57d106/FLN/tracto16/1k/SINGLE_106_57_l10/{mode}/MIX_Hellinger2_trivial/b_0.0/discovery_7"
  H_CONG = read_class(path, f"series_{iterations}")
  edr_data = pd.concat([H_EDR.node_entropy, H_EDR.link_entropy])
  cong_data = pd.concat([H_CONG.node_entropy, H_CONG.link_entropy])
  edr_data = edr_data.loc[(edr_data.c == "node_hierarchy") | (edr_data.c == "link_hierarchy") & (edr_data.dir == "H")]
  cong_data = cong_data.loc[(cong_data.c == "node_hierarchy") | (cong_data.c == "link_hierarchy") & (cong_data.dir == "H")]
  cortex_data = edr_data.loc[edr_data.data == "1"]

  edr_data = edr_data.loc[(edr_data.dir == "H") & (edr_data.data == "0")]
  cong_data = cong_data.loc[(cong_data.dir == "H") & (cong_data.data == "0")]

  cortex_data = cortex_data.loc[cortex_data.dir == "H"]
  
  edr_data["model"] = "EDR"
  cong_data["model"] = "Configuration"
  cortex_data["model"] = "Data"

  data = pd.concat([edr_data, cong_data, cortex_data])

  sns.set_context("talk")

  g = sns.FacetGrid(
    data=data,
    col = "c",
    hue = "model",
    sharex=False,
    sharey=False
  )
  g.map_dataframe(
    sns.lineplot,
    x="level",
    y="S",
    style="data",
    alpha=0.4
  )

  g.add_legend()
  sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False)

  g.axes[0, 0].set_ylabel(r"$s_{H}$")
  g.axes[0, 1].set_ylabel(r"$s_{H}$")

  fig = plt.gcf()
  fig.set_figheight(5)
  fig.set_figwidth(13)
  fig.tight_layout()

  plot_path = plot_path + "cortex_letter/"
  Path(plot_path).mkdir(exist_ok=True, parents=True)
  plt.savefig(
    join(
      plot_path, "Entropy_levels.png"
    ),
    dpi=300, bbox_extra_artists=(g._legend,), bbox_inches='tight'
  )
  plt.close()

def read_scalefree_entropy(MAXI : int, mode="ALPHA"):
    import itertools
    from networks_serial.scalehrh import SCALEHRH
    topologies = ["MIX"]
    indices = ["Hellinger2"]
    nodes = [150]
    KAV = [7]
    MUT = [0.1]
    MUW = [0.01]
    NMIN = [5]
    NMAX = [25]
    list_of_lists = itertools.product(
      *[topologies, indices, nodes, KAV, MUT, MUW, NMIN, NMAX]
    )
    list_of_lists = np.array(list(list_of_lists))
    nlog10 = F
    lookup = F
    prob = F
    run = T
    maxk = 20
    beta = 3
    t1 = 2
    t2 = 1
    mapping = "trivial"
    __mode__ = mode
    alpha=0.
    cut = F
    LBF = pd.DataFrame()
    for topology, index, N, kav, mut, muw, nmin, nmax in list_of_lists:
      N = int(N)
      nmin = int(nmin)
      nmax = int(nmax)
      if nmin > nmax: continue
      # WDN paramters ----
      par = {
        "-N" : "{}".format(str(N)),
        "-k" : f"{kav}.0",
        "-maxk" : f"{maxk}",
        "-mut" : f"{mut}",
        "-muw" : f"{muw}",
        "-beta" : f"{beta}",
        "-t1" : f"{t1}",
        "-t2" : f"{t2}",
        "-nmin" : f"{nmin}",
        "-nmax" : f"{nmax}"
      }
      l10 = ""
      lup = ""
      _cut = ""
      if nlog10: l10 = "_l10"
      if lookup: lup = "_lup"
      if cut: _cut = "_cut"
      data = read_class(
        "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/{}/{}/{}/{}/".format(
          str(N),
          par["-k"], par["-maxk"],
          par["-mut"], par["-muw"],
          par["-beta"], par["-t1"], par["-t2"],
          par["-nmin"], par["-nmax"], MAXI-1,
          __mode__,__mode__,
          f"{topology}_{index}_{mapping}"
        ),
        "series_{}".format(MAXI)
      )
      if isinstance(data, SCALEHRH):
        scale_data = pd.concat([data.node_entropy, data.link_entropy])
        scale_data = scale_data.loc[
          ((scale_data.c == "node_hierarchy") | (scale_data.c == "link_hierarchy")) & (scale_data.dir == "H")
        ].groupby(["iter", "c"]).sum()["S"].reset_index()
        scale_data["model"] = f"LFR_{N}_{mut}"
        LBF = pd.concat([LBF, scale_data])
    return LBF


def entropy_networks_220830(plot_path, mode="ALPHA", iterations=500):
    path = f"../pickle/RAN/distbase/MAC/57d106/FLN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/{mode}/MIX_Hellinger2_trivial/b_0.0/discovery_7"
    H_EDR = read_class(path, f"series_{iterations}")
    path = f"../pickle/RAN/swaps/MAC/57d106/FLN/tracto16/1k/SINGLE_106_57_l10/{mode}/MIX_Hellinger2_trivial/b_0.0/discovery_7"
    H_CONG = read_class(path, f"series_{iterations}")
    edr_data = pd.concat([H_EDR.node_entropy, H_EDR.link_entropy])
    cong_data = pd.concat([H_CONG.node_entropy, H_CONG.link_entropy])
    edr_data = edr_data.loc[
       ((edr_data.c == "node_hierarchy") | (edr_data.c == "link_hierarchy")) &
       (edr_data.dir == "H") & (edr_data.data == "0")
    ].groupby(["iter", "c"]).sum()["S"].reset_index()
    edr_data["model"] = "EDR"
    cong_data = cong_data.loc[
       ((cong_data.c == "node_hierarchy") | (cong_data.c == "link_hierarchy")) &
       (cong_data.dir == "H") & (cong_data.data == "0")
    ].groupby(["iter", "c"]).sum()["S"].reset_index()
    cong_data["model"] = "Configuration"
    ###

    path = "../pickle/TOY/ER/SINGLE_100_100_cut/ZERO/MIX_Hellinger2_trivial/b_0.0/"
    H_ER_jacp = read_class(path, "series_50_0.60_100")

    path = "../pickle/TOY/HRG/SINGLE_640_640_cut/ZERO/MIX_Hellinger2_trivial/b_0.0/"
    H_HRG_jacp = read_class(path, "series_25")
    
    erjacp_data = pd.concat([H_ER_jacp.node_entropy, H_ER_jacp.link_entropy])
    hrgjacp_data = pd.concat([H_HRG_jacp.node_entropy, H_HRG_jacp.link_entropy])

    erjacp_data = erjacp_data.loc[
       ((erjacp_data.c == "node_hierarchy") |
       (erjacp_data.c == "link_hierarchy") )&
       (erjacp_data.dir == "H")
    ].groupby(["iter", "c"]).sum()["S"].reset_index()
    erjacp_data["model"] = "ER"
    hrgjacp_data = hrgjacp_data.loc[
       ((hrgjacp_data.c == "node_hierarchy") |
       (hrgjacp_data.c == "link_hierarchy")) &
       (hrgjacp_data.dir == "H")
    ].groupby(["iter", "c"]).sum()["S"].reset_index()
    hrgjacp_data["model"] = "HRG"

    # ###

    path = f"../pickle/RAN/distbase/MAC/57d106/FLN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/{mode}/MIX_Hellinger2_trivial/b_0.0/discovery_7"
    H_EDR = read_class(path, f"series_{iterations}")
    mac_data = pd.concat([H_EDR.node_entropy, H_EDR.link_entropy])
    mac_data = mac_data.loc[
       ((mac_data.c == "node_hierarchy") | (mac_data.c == "link_hierarchy")) &
       (mac_data.dir == "H") & (mac_data.data == "1")
    ].groupby("c").sum()["S"].reset_index()
    mac_data["model"] = "data"

    networks = np.array(["Zachary", "HSF", "Tractography"])
    hierarchy = np.array(["link_hierarchy", "node_hierarchy"])
    michelle = pd.DataFrame(
      {
        "model" : np.repeat(networks, 2),
        "S" : [0.2332, 0.1792, 0.4113, 0.4691, 0.4712, 0.2984],
        "c" : np.tile(hierarchy, 3)
      }
    )

    ##

    lfr = read_scalefree_entropy(50, mode=mode)

    lfr_models = list(np.unique(lfr.model))

    data = pd.concat(
       [erjacp_data, hrgjacp_data, edr_data, cong_data, mac_data, michelle, lfr]
    )
    fig = plt.gcf()
    order = ["Zachary", "ER", "HRG", "HSF"]+ lfr_models + ["Configuration", "EDR", "data", "Tractography"]
    sns.set_context("talk")
    g = sns.barplot(
       data=data,
       x="model",
       y="S",
       hue="c",
       estimator="median",
       errorbar="sd",
       order=order,
       alpha=0.4
    )

    g.legend_.remove()

    sns.violinplot(
       data=data,
       x="model",
       y="S",
       hue="c",
       errorbar="sd",
       order=order,
       ax=g
    )

    g.set_ylabel(r"$s_{H}$")
    g.set_xticklabels(g.get_xticklabels(), rotation=45, size=20)


    from scipy.stats import ttest_1samp

    

    yheight = {"link_hierarchy": 0.85, "node_hierarchy" : 0.9}
    xwidht = {"EDR" : 0.7, "Configuration" : 0.57}
    for i, m in enumerate(["EDR", "Configuration"]):
       for j, l in enumerate(["link_hierarchy", "node_hierarchy"]):
          d = data["S"].loc[(data.model == m) & (data.c == l)]
          da = data["S"].loc[(data.model == "data") & (data.c == l)]

          test = ttest_1samp(d, popmean=np.mean(da))

          if  not np.isnan(test.pvalue): 
            if test.pvalue > 0.05:
              a = "ns"
            elif test.pvalue <= 0.05 and test.pvalue > 0.001:
              a = "*" 
            elif test.pvalue <= 0.001 and test.pvalue > 0.0001:
              a = "**" 
            else:
              a = "***"
          else:
            a = "nan"
          if l == "link_hierarchy" : L = "LH"
          else: L = "NH"
          g.text(xwidht[m], yheight[l], f"{L}: {a}", transform=g.transAxes)


    sns.move_legend(g, "upper left", bbox_to_anchor=(0.01, 0.95), ncol=1, title=None)

    fig.set_figheight(9)
    fig.set_figwidth(12)
    fig.tight_layout()

    # import scipy.stats as stats

    # xlink = data["S"].loc[(data.model == "EDR") & (data.c == "link_hierarchy")]
    # xnode = data["S"].loc[(data.model == "EDR") & (data.c == "node_hierarchy")]


    # mac_link = mac_data["S"].loc[mac_data.c  == "link_hierarchy"]

    # mac_node = mac_data["S"].loc[mac_data.c  == "node_hierarchy"]

    # print(stats.ttest_1samp(xlink, mac_link))
    # print(stats.ttest_1samp(xnode, mac_node))
    
    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "Entropy_networks.png"
      ),
      dpi=300
    )
    plt.close()

def entropy_networks_40d91(plot_path):
    path = "../pickle/RAN/distbase/MAC/40d91/LN/MAP3D/EXPMLE/BIN_12/SINGLE_91_40_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
    H_EDR = read_class(path, "series_500")
    path = "../pickle/RAN/swaps/MAC/40d91/LN/MAP3D/1k/SINGLE_91_40_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
    H_CONG = read_class(path, "series_500")
    edr_data = pd.concat([H_EDR.node_entropy, H_EDR.link_entropy])
    cong_data = pd.concat([H_CONG.node_entropy, H_CONG.link_entropy])
    edr_data = edr_data.loc[
       ((edr_data.c == "node_hierarchy") | (edr_data.c == "link_hierarchy")) &
       (edr_data.dir == "H") & (edr_data.data == "0")
    ].groupby(["iter", "c"]).sum()["S"].reset_index()
    edr_data["model"] = "EDR_LN"
    cong_data = cong_data.loc[
       ((cong_data.c == "node_hierarchy") | (cong_data.c == "link_hierarchy")) &
       (cong_data.dir == "H") & (cong_data.data == "0")
    ].groupby(["iter", "c"]).sum()["S"].reset_index()
    cong_data["model"] = "Configuration_LN"

    ##
    bias = ["b_0.0", "b_1e-05", "b_0.1", "b_0.3"]
    edr_data_fln = []
    conf_data_fln = []
    null_names = []
    for b in bias:
      path = f"../pickle/RAN/distbase/MAC/40d91/FLN/MAP3D/EXPMLE/BIN_12/SINGLE_91_40_l10/ALPHA/MIX_jacw_R2/{b}/"
      H_EDR = read_class(path, "series_500")
      # print(H_EDR)a
      path = f"../pickle/RAN/swaps/MAC/40d91/FLN/MAP3D/1k/SINGLE_91_40_l10/ALPHA/MIX_jacw_R2/{b}/"
      H_CONG = read_class(path, "series_500")
      edr_data_ = pd.concat([H_EDR.node_entropy, H_EDR.link_entropy], ignore_index=True)
      cong_data_ = pd.concat([H_CONG.node_entropy, H_CONG.link_entropy], ignore_index=True)
      edr_data_ = edr_data_.loc[
        ((edr_data_.c == "node_hierarchy") | (edr_data_.c == "link_hierarchy")) &
        (edr_data_.dir == "H") & (edr_data_.data == "0")
      ].groupby(["iter", "c"]).sum()["S"].reset_index()
      edr_data_["model"] = f"EDR_FLN_{b}"
      cong_data_ = cong_data_.loc[
        ((cong_data_.c == "node_hierarchy") | (cong_data_.c == "link_hierarchy")) &
        (cong_data_.dir == "H") & (cong_data_.data == "0")
      ].groupby(["iter", "c"]).sum()["S"].reset_index()
      cong_data_["model"] = f"Configuration_FLN_{b}"
      null_names = null_names + [f"EDR_FLN_{b}", f"Configuration_FLN_{b}"]
      edr_data_fln.append(edr_data_)
      conf_data_fln.append(cong_data_)
    ###

    path = "../pickle/TOY/ER/SINGLE_128_128_cut/ALPHA/MIX_jacp_trivial/b_0.0/"
    H_ER_jacp = read_class(path, "series_500")

    path = "../pickle/TOY/HRG/SINGLE_640_640_cut/ALPHA/MIX_jacp_trivial/b_0.0/"
    H_HRG_jacp = read_class(path, "series_500")
    
    erjacp_data = pd.concat([H_ER_jacp.node_entropy, H_ER_jacp.link_entropy])
    hrgjacp_data = pd.concat([H_HRG_jacp.node_entropy, H_HRG_jacp.link_entropy])

    erjacp_data = erjacp_data.loc[
       ((erjacp_data.c == "node_hierarchy") |
       (erjacp_data.c == "link_hierarchy") )&
       (erjacp_data.dir == "H")
    ].groupby(["iter", "c"]).sum()["S"].reset_index()
    erjacp_data["model"] = "ER"
    hrgjacp_data = hrgjacp_data.loc[
       ((hrgjacp_data.c == "node_hierarchy") |
       (hrgjacp_data.c == "link_hierarchy")) &
       (hrgjacp_data.dir == "H")
    ].groupby(["iter", "c"]).sum()["S"].reset_index()
    hrgjacp_data["model"] = "HRG"

    # ###

    path = "../pickle/RAN/distbase/MAC/40d91/LN/MAP3D/EXPMLE/BIN_12/SINGLE_91_40_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
    H_EDR = read_class(path, "series_500")
    mac_data = pd.concat([H_EDR.node_entropy, H_EDR.link_entropy])
    mac_data = mac_data.loc[
       ((mac_data.c == "node_hierarchy") | (mac_data.c == "link_hierarchy")) &
       (mac_data.dir == "H") & (mac_data.data == "1")
    ].groupby("c").sum()["S"].reset_index()
    mac_data["model"] = "data_LN"

    networks = np.array(["Zachary", "HSF"])
    hierarchy = np.array(["link_hierarchy", "node_hierarchy"])
    michelle = pd.DataFrame(
      {
        "model" : np.repeat(networks, 2),
        "S" : [0.2062, 0.1543, 0.5053, 0.4080],
        "c" : np.tile(hierarchy, 2)
      }
    )

    bias = ["b_0.0", "b_1e-05", "b_0.1", "b_0.3"]
    mac_data_fln = []
    mac_fln_names = []
    for b in bias:
      path = f"../pickle/RAN/distbase/MAC/40d91/FLN/MAP3D/EXPMLE/BIN_12/SINGLE_91_40_l10/ALPHA/MIX_jacw_R2/{b}/"
      H_EDR = read_class(path, "series_500")
      mac_data_ = pd.concat([H_EDR.node_entropy, H_EDR.link_entropy])
      mac_data_ = mac_data_.loc[
        ((mac_data_.c == "node_hierarchy") | (mac_data_.c == "link_hierarchy")) &
        (mac_data_.dir == "H") & (mac_data_.data == "1")
      ].groupby("c").sum()["S"].reset_index()
      mac_data_["model"] = f"data_FLN_{b}"
      mac_data_fln.append(mac_data_)
      mac_fln_names.append(f"data_FLN_{b}")

    networks = np.array(["Zachary", "HSF"])
    hierarchy = np.array(["link_hierarchy", "node_hierarchy"])
    michelle = pd.DataFrame(
      {
        "model" : np.repeat(networks, 2),
        "S" : [0.2062, 0.1543, 0.5053, 0.4080],
        "c" : np.tile(hierarchy, 2)
      }
    )

    ##

    # lbf = read_scalefree_entropy(25)
    # lbf_models = list(np.unique(lbf.model))

    data = pd.concat([erjacp_data, hrgjacp_data, edr_data, cong_data, mac_data, michelle] + edr_data_fln + conf_data_fln + mac_data_fln)
    fig = plt.gcf()
    ax = plt.gca()
    order = ["Zachary", "ER", "HRG", "HSF", "Configuration_LN", "EDR_LN", "data_LN"] + null_names + mac_fln_names# + lbf_models
    sns.violinplot(
       data=data,
       x="model",
       y="S",
       hue="c",
       errorbar="sd",
       order=order,
       ax=ax
    )
    sns.barplot(
       data=data,
       x="model",
       y="S",
       hue="c",
       errorbar="sd",
       order=order,
       alpha=0.4,
       ax=ax
    )
    ax.set_ylabel(r"$s_{H}$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    fig.set_figheight(9)
    fig.set_figwidth(20)
    fig.tight_layout()
    
    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "Entropy_networks.png"
      ),
      dpi=300
    )
    plt.close()

def sim_histogram(plot_path, mode="ALPHA"):
    path = f"../pickle/MUS/19d47/FLN/original/MAP3D/19/SINGLE_47_19/{mode}/MIX_Hellinger2_trivial/b_0.0/discovery_7"
    H = read_class(path, "hanalysis")

    wlabel = r"$H^{2}$"

    # src = np.log(H.source_sim_matrix)
    # tgt = np.log(H.target_sim_matrix)

    src = 1 - H.source_sim_matrix
    tgt = 1 - H.target_sim_matrix
    
    src = adj2df(src)
    tgt = adj2df(tgt)

    src = src.loc[src.source > src.target]
    tgt = tgt.loc[tgt.source > tgt.target]
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus'] = False
    from matplotlib import rc
    rc('text', usetex=False)

    label_source = r"$H^{2}_{+}$"
    label_target = r"$H^{2}_{-}$"

    data = pd.DataFrame(
       {
        wlabel : list(src.weight) + list(tgt.weight) ,
        "direction" : [label_source] * src.shape[0] + [label_target] * tgt.shape[0]
       }
    )

    # sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.style.use("dark_background")

    sns.histplot(
       data=data,
       x=wlabel,
       hue="direction",
       stat="density",
       palette="pastel",
       common_norm=False
    )



    s = np.percentile(data[wlabel].loc[data.direction == label_source], 16.5)
    t = np.percentile(data[wlabel].loc[data.direction == label_target], 16.5)
    a = np.mean([s, t])

    fig = plt.gcf()
    ax = plt.gca() 
    ax.plot([a, a], [0.1, 0.5], 'red', linestyle="-", linewidth=5.5)
    ax.scatter(a, 0.2, marker="v", c="red", s=120)
    ax.text(0.3, 0.4, r"$p_{16.5}=$"+f"{a:.2f}", transform=ax.transAxes, size=30)
    ax.set_xlabel(wlabel)
    fig.set_figheight(7)
    fig.set_figwidth(10)
    fig.tight_layout()

    plot_path = plot_path + "cortex_letter/MUS/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "H2_bin_dark.svg"
      ),
      dpi=300, transparent=T
    )
    plt.close()

def sim_dist(plot_path, mode="ALPHA"):
    path = f"../pickle/MUS/19d47/FLN/original/MAP3D/19/SINGLE_47_19/{mode}/MIX_Hellinger2_trivial/b_0.0/discovery_7"
    H = read_class(path, "hanalysis")
    src = H.source_sim_matrix
    tgt = H.target_sim_matrix
    D = H.D[:NET.nodes, :]
    
    src = adj2df(src)
    tgt = adj2df(tgt)
    D = adj2df(D)

    src = src.loc[src.source > src.target]
    tgt = tgt.loc[tgt.source > tgt.target]
    D = D.loc[D.source > D.target]

    data = pd.DataFrame(
       {
        r"$D_{1/2}$" : list(1 / src.weight - 1) + list(1 / tgt.weight - 1) ,
        "direction" : ["outgoing"] * src.shape[0] + ["incoming"] * tgt.shape[0],
        "dist" : list(D.weight) + list(D.weight) 
       }
    )

    sns.scatterplot(
       data=data,
       x="dist",
       y=r"$D_{1/2}$",
       hue="direction",
       alpha = 0.7
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("tractography distance [mm]")
    fig.set_figheight(7)
    fig.set_figwidth(10)
    fig.tight_layout()

    plot_path = plot_path + "cortex_letter/MUS/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "D1_2_2_dist.png"
      ),
      dpi=300
    )
    plt.close()

def  fair_dist_bin(plot_path, pickle_path, ci=95, mode="ZERO", cmap="deep"):
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    from seaborn import utils
    import ctools as ct

    path = pickle_path
    H = read_class(path, "hanalysis")

    w_inset = r"$D_{1/2}$"

    src = np.sqrt(1 - H.source_sim_matrix)
    tgt = np.sqrt(1 - H.target_sim_matrix)
    D = H.D[:NET.nodes, :NET.nodes]

    src_inset = np.zeros(src.shape)
    tgt_inset = np.zeros(tgt.shape)

    for i in np.arange(src_inset.shape[0]):
       for j in np.arange(i + 1, src_inset.shape[1]):
          src_inset[i, j] = ct.D1_2_4(H.A[i, :], H.A[j, :], i, j)
          src_inset[j, i] = src_inset[i, j]

          tgt_inset[i, j] = ct.D1_2_4(H.A[:, i], H.A[:, j], i, j)
          tgt_inset[j, i] = tgt_inset[i, j]

    np.seterr(divide='ignore', invalid='ignore')
    src_inset = (1 / src_inset) - 1
    np.seterr(divide='ignore', invalid='ignore')
    tgt_inset = (1 / tgt_inset) - 1
    
    src = adj2df(src)
    tgt = adj2df(tgt)
    src_inset = adj2df(src_inset)
    tgt_inset = adj2df(tgt_inset)
    D = adj2df(D)

    src = src["weight"].loc[src.source > src.target].to_numpy()
    tgt = tgt["weight"].loc[tgt.source > tgt.target].to_numpy()
    src_inset = src_inset["weight"].loc[src_inset.source > src_inset.target].to_numpy()
    tgt_inset = tgt_inset["weight"].loc[tgt_inset.source > tgt_inset.target].to_numpy()
    D = D["weight"].loc[D.source > D.target].to_numpy()

    ### ----

    data = pd.DataFrame(
       {
        w_inset: list(src_inset) + list(tgt_inset),
        "direction" : ["source"] * src.shape[0] + ["target"] * tgt.shape[0],
        "interareal distance [mm]" : list(D) + list(D) 
       }
    )

    # Binarize ----
    nbin = 15
    D_min, D_max = np.min(D), np.max(D)
    d_range = np.linspace(D_min, D_max, nbin)
    dD = d_range[1] - d_range[0]
    d_range[-1] += 0.0001
    src_bin = np.array([np.nanmean(src[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)])
    tgt_bin = np.array([np.nanmean(tgt[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)])
    src_inset_bin = np.array([np.nanmean(src_inset[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)])
    tgt_inset_bin = np.array([np.nanmean(tgt_inset[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)])
    d_range[-1] -= 0.0001
    d_range = d_range[:-1]
    d_range = d_range + dD / 2
    data_bin = pd.DataFrame(
       {
          w_inset : list(src_inset_bin) + list(tgt_inset_bin),
          "direction" : ["source"] * len(src_bin) + ["target"] * len(src_bin),
          "interareal distance [mm]" : list(d_range) + list(d_range) 
       }
    )

    data_ref = data.loc[data[w_inset] < np.Inf]

    data_ref_src = data_ref["interareal distance [mm]"].loc[data_ref["direction"] == "source"].to_numpy()
    data_ref_tgt = data_ref["interareal distance [mm]"].loc[data_ref["direction"] == "target"].to_numpy()

    scale= StandardScaler()
    # mu_d_range = np.mean(d_range)
    sd_d_range_src = np.std(data_ref_src)
    sd_d_range_tgt = np.std(data_ref_tgt)
    Xsrc = scale.fit_transform(data_ref_src.reshape(-1, 1))
    Xsrc = sm.add_constant(Xsrc)
    Xtgt = scale.fit_transform(data_ref_tgt.reshape(-1, 1))
    Xtgt = sm.add_constant(Xtgt)
    est_src = sm.OLS(data_ref[w_inset].loc[data_ref["direction"] == "source"], Xsrc).fit()
    est_tgt = sm.OLS(data_ref[w_inset].loc[data_ref["direction"] == "target"], Xtgt).fit()
    print(est_src.params[1] / sd_d_range_src)
    print(est_tgt.params[1] / sd_d_range_tgt)

    labmda_src = np.round(est_src.params[1] / sd_d_range_src, 2)
    labmda_tgt = np.round(est_tgt.params[1] / sd_d_range_tgt, 2)

    # par_tgt = est_tgt.params[1] / sd_d_range
    # par_tgt /= 2
    # char_length = (2 + est_tgt.params[0]) * sd_d_range / est_tgt.params[1] - mu_d_range

    # a = np.sqrt(1 - np.exp(-1))

    # line of H(d) using the one in D_1/2(d) ----
    # grid = np.linspace(d_range[0], d_range[-1], 100)

    # yhat_src, yhat_src_boots = fit_poly(d_range, src_inset_bin, grid, order=1)
    # yhat_tgt, yhat_tgt_boots = fit_poly(d_range, tgt_bin, grid, order=3)

    # err_src_bands = utils.ci(yhat_src_boots, ci, axis=0)
    # err_tgt_bands = utils.ci(yhat_tgt_boots, ci, axis=0)

    # yhat_src = np.sqrt(1-np.exp(-0.5 * yhat_src))
    # yhat_tgt = np.sqrt(1-np.exp(-0.5 * yhat_tgt))

    # err_src_bands = np.sqrt(1-np.exp(-0.5 * err_src_bands))
    # err_tgt_bands = np.sqrt(1-np.exp(-0.5 * err_tgt_bands))

    # char_length_lowess = [d for d, y in zip(grid, yhat_tgt) if y >= a - 0.001 and y < a + 0.001][0]

    sns.set_context("talk")
    sns.set_style("whitegrid")
    # plt.style.use("dark_background")

    fig, ax = plt.subplots(1, 1)
    
    # sns.scatterplot(
    #   data=data,
    #   x="interareal distance [mm]",
    #   y= wlabel,
    #   hue="direction",
    #   hue_order=["source", "target"],
    #   alpha=0.4,
    #   palette=cmap,
    #   s=6,
    #   ax=ax
    # )

    # ax.scatter(d_range, src_bin, color=sns.color_palette(cmap, as_cmap=True)[0], s=30)
    # ax.scatter(d_range, tgt_bin, color=sns.color_palette(cmap, as_cmap=True)[1], s=30)

    # ax.plot(grid, yhat_src, color=sns.color_palette(cmap, as_cmap=True)[0])
    # ax.fill_between(grid, *err_src_bands, facecolor=sns.color_palette(cmap, as_cmap=True)[0], alpha=.15)

    # ax.plot(grid, yhat_tgt, color=sns.color_palette(cmap, as_cmap=True)[1])
    # ax.fill_between(grid, *err_tgt_bands, facecolor=sns.color_palette(cmap, as_cmap=True)[1], alpha=.15)



    # sns.regplot(
    #    data=data_bin.loc[data_bin.direction == "target"],
    #    x="interareal distance [mm]",
    #    y = wlabel,
    #    scatter_kws={"alpha":0.6, "s":30},
    #    line_kws={"linewidth":1},
    #   #  lowess=True,
    #   order=3,
    #    color=sns.color_palette(cmap, as_cmap=True)[1],
    #    ax=ax
    # )

    # sns.regplot(
    #    data=data_bin.loc[data_bin.direction == "source"],
    #    x="interareal distance [mm]",
    #    y = wlabel,
    #    scatter_kws={"alpha":0.6, "s":30},
    #    line_kws={"linewidth":1},
    #   #  lowess=True,
    #   order=3,
    #    color=sns.color_palette(cmap, as_cmap=True)[0],
    #    ax=ax
    # )

    # axinset = ax.inset_axes([0.5, 0.1, 0.45, 0.4], transform=ax.transAxes)

    sns.scatterplot(
      data=data,
      x="interareal distance [mm]",
      y= w_inset,
      hue="direction",
      hue_order=["source", "target"],
      alpha=0.4,
      palette=cmap,
      s=35,
      ax=ax
    )

    # sns.lineplot(
    #    data=data_bin.loc[data_bin.direction == "target"],
    #    x="interareal distance [mm]",
    #    y = w_inset,
    #   #  lowess=True,
    #   linestyle = "--",
    #    linewidth=1,
    #    color=sns.color_palette(cmap, as_cmap=True)[1],
    #    ax=ax
    # )

    # sns.lineplot(
    #    data=data_bin.loc[data_bin.direction == "source"],
    #    x="interareal distance [mm]",
    #    y = w_inset,
    #   #  lowess=True,
    #   linewidth=1,
    #   linestyle = "--",
    #    color=sns.color_palette(cmap, as_cmap=True)[0],
    #    ax=ax
    # )

    sns.regplot(
       data=data.loc[data.direction == "target"],
       x="interareal distance [mm]",
       y = w_inset,
       scatter_kws={"s":0},
       line_kws={"linewidth":2},
      #  lowess=True,
       color=sns.color_palette(cmap, as_cmap=True)[1],
       ax=ax
    )

    sns.regplot(
       data=data.loc[data.direction == "source"],
       x="interareal distance [mm]",
       y = w_inset,
       scatter_kws={"s":0},
       line_kws={"linewidth":2},
      #  lowess=True,
       color=sns.color_palette(cmap, as_cmap=True)[0],
       ax=ax
    )

    sns.scatterplot(
       data=data_bin.loc[data_bin.direction == "target"],
       x="interareal distance [mm]",
       y = w_inset,
       s =60,
       color=sns.color_palette(cmap, as_cmap=True)[1],
       ax=ax
    )

    sns.scatterplot(
       data=data_bin.loc[data_bin.direction == "source"],
       x="interareal distance [mm]",
       y = w_inset,
       s= 60,
       color=sns.color_palette(cmap, as_cmap=True)[0],
       ax=ax
    )
    # axinset.get_legend().remove()
    # axinset.set_xlabel("")

    # sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False)

    # if char_length > char_length_lowess:
    #   marker_high = char_length
    #   marker_low = char_length_lowess
    # else:
    #   marker_high = char_length_lowess
    #   marker_low = char_length
    # ax.plot([marker_low, marker_high], [a, a], linestyle="-", linewidth=3, alpha=0.4, c=sns.color_palette("deep")[3])
    # ax.plot([marker_low, marker_low], [0, a], linestyle="-", linewidth=3, alpha=0.4, c=sns.color_palette("deep")[3])
    # ax.plot([marker_high, marker_high], [0, a], linestyle="-", linewidth=3, alpha=0.4, c=sns.color_palette("deep")[3])
    # # ax.scatter(marker_low, a, marker="<", s=120, alpha=0.4, c=sns.color_palette("deep")[3])
    # ax.text(0.1, 0.15, f"L: {np.round(marker_low, 1)}", size=20, transform=ax.transAxes)
    # ax.text(0.1, 0.1, f"R: {np.round(marker_high, 1)}", size=20, transform=ax.transAxes)
    # ax.text(0.5, a-0.01, r"$H=$"+f"{a:.2f}", size=20)

    # g = sns.lmplot(
    #    data=data_resid,
    #    x=r"$D_{1/2}$" ,
    #    y="residuals",
    #    hue="direction",
    #    scatter_kws= {"s" : 3, "alpha" : 0.5},
    #    lowess=True
    # )
    # ax.set_xlabel("tractography distance [mm]")
    # # ax[1] = g.axes[0, 0]

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    ax.text(
       0.5, 1.02, r"$\lambda_{-}=$" + f"{labmda_tgt}," + "   " +  r"$\lambda_{+}=$" + f"{labmda_src}" + "   " + "[nats/mm]", transform=ax.transAxes,
       horizontalalignment='center',
      #  verticalalignment = "top"
    )

    fig.set_figheight(7)
    fig.set_figwidth(10)
    fig.tight_layout()

    # plt.show()
 
    plot_path = plot_path + "/cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "fair_dist_bin.svg"
      ),
      dpi=300, transparent=T
    )
    plt.close()

def  sim_dist_bin(plot_path, pickle_path, ci=95, mode="ALPHA", cmap="deep"):
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    from seaborn import utils
    import ctools as ct

    path = pickle_path
    H = read_class(path, "hanalysis")

    wlabel = r"$H$"
    w_inset = r"$D_{1/2}$"

    src = np.sqrt(1 - H.source_sim_matrix)
    tgt = np.sqrt(1 - H.target_sim_matrix)
    D = H.D[:NET.nodes, :NET.nodes]

    src_inset = np.zeros(src.shape)
    tgt_inset = np.zeros(tgt.shape)

    for i in np.arange(src_inset.shape[0]):
       for j in np.arange(i + 1, src_inset.shape[1]):
          src_inset[i, j] = ct.D1_2_4(H.A[i, :], H.A[j, :], i, j)
          src_inset[j, i] = src_inset[i, j]

          tgt_inset[i, j] = ct.D1_2_4(H.A[:, i], H.A[:, j], i, j)
          tgt_inset[j, i] = tgt_inset[i, j]

    np.seterr(divide='ignore', invalid='ignore')
    src_inset = (1 / src_inset) - 1
    np.seterr(divide='ignore', invalid='ignore')
    tgt_inset = (1 / tgt_inset) - 1
    
    src = adj2df(src)
    tgt = adj2df(tgt)
    src_inset = adj2df(src_inset)
    tgt_inset = adj2df(tgt_inset)
    D = adj2df(D)

    src = src["weight"].loc[src.source > src.target].to_numpy()
    tgt = tgt["weight"].loc[tgt.source > tgt.target].to_numpy()
    src_inset = src_inset["weight"].loc[src_inset.source > src_inset.target].to_numpy()
    tgt_inset = tgt_inset["weight"].loc[tgt_inset.source > tgt_inset.target].to_numpy()
    D = D["weight"].loc[D.source > D.target].to_numpy()

    ### ----

    data = pd.DataFrame(
       {
        wlabel : list(src) + list(tgt) ,
        w_inset: list(src_inset) + list(tgt_inset),
        "direction" : ["source"] * src.shape[0] + ["target"] * tgt.shape[0],
        "interareal distance [mm]" : list(D) + list(D) 
       }
    )

    # Binarize ----
    nbin = 15
    D_min, D_max = np.min(D), np.max(D)
    d_range = np.linspace(D_min, D_max, nbin)
    dD = d_range[1] - d_range[0]
    d_range[-1] += 0.0001
    src_bin = np.array([np.nanmean(src[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)])
    tgt_bin = np.array([np.nanmean(tgt[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)])
    src_inset_bin = np.array([np.nanmean(src_inset[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)])
    tgt_inset_bin = np.array([np.nanmean(tgt_inset[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)])
    d_range[-1] -= 0.0001
    d_range = d_range[:-1]
    d_range = d_range + dD / 2
    data_bin = pd.DataFrame(
       {
          wlabel : list(src_bin) + list(tgt_bin) ,
          w_inset : list(src_inset_bin) + list(tgt_inset_bin),
          "direction" : ["source"] * len(src_bin) + ["target"] * len(src_bin),
          "interareal distance [mm]" : list(d_range) + list(d_range) 
       }
    )

    data_bin = data_bin.loc[data_bin[w_inset] < np.Inf]

    d_range_src = data_bin["interareal distance [mm]"].loc[data_bin["direction"] == "source"].to_numpy()
    d_range_tgt = data_bin["interareal distance [mm]"].loc[data_bin["direction"] == "target"].to_numpy()

    scale= StandardScaler()
    mu_d_range = np.mean(d_range)
    sd_d_range = np.std(d_range)
    Xsrc = scale.fit_transform(d_range_src.reshape(-1, 1))
    Xsrc = sm.add_constant(Xsrc)
    Xtgt = scale.fit_transform(d_range_tgt.reshape(-1, 1))
    Xtgt = sm.add_constant(Xtgt)
    est_src = sm.OLS(data_bin[w_inset].loc[data_bin["direction"] == "source"], Xsrc).fit()
    est_tgt = sm.OLS(data_bin[w_inset].loc[data_bin["direction"] == "target"], Xtgt).fit()
    print(est_src.params[1] / sd_d_range)
    print(est_tgt.params[1] / sd_d_range)

    par_tgt = est_tgt.params[1] / sd_d_range
    par_tgt /= 2
    char_length = (2 + est_tgt.params[0]) * sd_d_range / est_tgt.params[1] - mu_d_range

    a = np.sqrt(1 - np.exp(-1))

    # line of H(d) using the one in D_1/2(d) ----
    grid = np.linspace(d_range[0], d_range[-1], 100)

    # yhat_src, yhat_src_boots = fit_poly(d_range, src_inset_bin, grid, order=1)
    yhat_tgt, yhat_tgt_boots = fit_poly(d_range, tgt_bin, grid, order=3)

    # err_src_bands = utils.ci(yhat_src_boots, ci, axis=0)
    # err_tgt_bands = utils.ci(yhat_tgt_boots, ci, axis=0)

    # yhat_src = np.sqrt(1-np.exp(-0.5 * yhat_src))
    # yhat_tgt = np.sqrt(1-np.exp(-0.5 * yhat_tgt))

    # err_src_bands = np.sqrt(1-np.exp(-0.5 * err_src_bands))
    # err_tgt_bands = np.sqrt(1-np.exp(-0.5 * err_tgt_bands))

    char_length_lowess = [d for d, y in zip(grid, yhat_tgt) if y >= a - 0.01 and y < a + 0.01][-1]

    # sns.set_context("talk")
    sns.set_style("whitegrid")
    # plt.style.use("dark_background")

    fig, ax = plt.subplots(1, 1)
    
    sns.scatterplot(
      data=data,
      x="interareal distance [mm]",
      y= wlabel,
      hue="direction",
      hue_order=["source", "target"],
      alpha=0.4,
      palette=cmap,
      s=6,
      ax=ax
    )

    # ax.scatter(d_range, src_bin, color=sns.color_palette(cmap, as_cmap=True)[0], s=30)
    # ax.scatter(d_range, tgt_bin, color=sns.color_palette(cmap, as_cmap=True)[1], s=30)

    # ax.plot(grid, yhat_src, color=sns.color_palette(cmap, as_cmap=True)[0])
    # ax.fill_between(grid, *err_src_bands, facecolor=sns.color_palette(cmap, as_cmap=True)[0], alpha=.15)

    # ax.plot(grid, yhat_tgt, color=sns.color_palette(cmap, as_cmap=True)[1])
    # ax.fill_between(grid, *err_tgt_bands, facecolor=sns.color_palette(cmap, as_cmap=True)[1], alpha=.15)



    sns.regplot(
       data=data_bin.loc[data_bin.direction == "target"],
       x="interareal distance [mm]",
       y = wlabel,
       scatter_kws={"alpha":0.6, "s":30},
       line_kws={"linewidth":1},
      #  lowess=True,
      order=3,
       color=sns.color_palette(cmap, as_cmap=True)[1],
       ax=ax
    )

    sns.regplot(
       data=data_bin.loc[data_bin.direction == "source"],
       x="interareal distance [mm]",
       y = wlabel,
       scatter_kws={"alpha":0.6, "s":30},
       line_kws={"linewidth":1},
      #  lowess=True,
      order=3,
       color=sns.color_palette(cmap, as_cmap=True)[0],
       ax=ax
    )

    axinset = ax.inset_axes([0.5, 0.1, 0.45, 0.4], transform=ax.transAxes)

    sns.scatterplot(
      data=data,
      x="interareal distance [mm]",
      y= w_inset,
      hue="direction",
      hue_order=["source", "target"],
      alpha=0.4,
      palette=cmap,
      s=6,
      ax=axinset
    )

    sns.regplot(
       data=data_bin.loc[data_bin.direction == "target"],
       x="interareal distance [mm]",
       y = w_inset,
       scatter_kws={"alpha":0.6, "s":30},
       line_kws={"linewidth":1},
      #  lowess=True,
       color=sns.color_palette(cmap, as_cmap=True)[1],
       ax=axinset
    )

    sns.regplot(
       data=data_bin.loc[data_bin.direction == "source"],
       x="interareal distance [mm]",
       y = w_inset,
       scatter_kws={"alpha":0.6, "s":30},
       line_kws={"linewidth":1},
      #  lowess=True,
       color=sns.color_palette(cmap, as_cmap=True)[0],
       ax=axinset
    )
    axinset.get_legend().remove()
    axinset.set_xlabel("")

    sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False)

    if char_length > char_length_lowess:
      marker_high = char_length
      marker_low = char_length_lowess
    else:
      marker_high = char_length_lowess
      marker_low = char_length
    # ax.plot([marker_low, marker_high], [a, a], linestyle="-", linewidth=3, alpha=0.4, c=sns.color_palette("deep")[3])
    # ax.plot([marker_low, marker_low], [0, a], linestyle="-", linewidth=3, alpha=0.4, c=sns.color_palette("deep")[3])
    # ax.plot([marker_high, marker_high], [0, a], linestyle="-", linewidth=3, alpha=0.4, c=sns.color_palette("deep")[3])
    # ax.scatter(marker_low, a, marker="<", s=120, alpha=0.4, c=sns.color_palette("deep")[3])
    # ax.text(0.1, 0.15, f"L: {np.round(marker_low, 1)}", size=20, transform=ax.transAxes)
    # ax.text(0.1, 0.1, f"R: {np.round(marker_high, 1)}", size=20, transform=ax.transAxes)
    # ax.text(0.5, a-0.01, r"$H=$"+f"{a:.2f}", size=20)

    # g = sns.lmplot(
    #    data=data_resid,
    #    x=r"$D_{1/2}$" ,
    #    y="residuals",
    #    hue="direction",
    #    scatter_kws= {"s" : 3, "alpha" : 0.5},
    #    lowess=True
    # )
    # ax.set_xlabel("tractography distance [mm]")
    # # ax[1] = g.axes[0, 0]

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    fig.set_figheight(7)
    fig.set_figwidth(10)
    fig.tight_layout()

    # plt.show()
 
    plot_path = plot_path + "/cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "H_dist_bin.png"
      ),
      dpi=300#, transparent=T
    )
    plt.close()


def overlap(plot_path, common_features, mode="ALPHA", iterations=500):
    
    path = "../pickle/RAN/distbase/{}/{}/FLN/{}/EXPMLE/BIN_12/{}/{}/MIX_Hellinger2_trivial/discovery_7".format(
      common_features["subject"],
      common_features["version"],
      common_features["distance"],
      common_features["subfolder"],
      mode
    )
    H_EDR = read_class(path, f"series_{iterations}")
    path = "../pickle/RAN/swaps/{}/{}/FLN/{}/1k/{}/{}/MIX_Hellinger2_trivial/discovery_7".format(
      common_features["subject"],
      common_features["version"],
      common_features["distance"],
      common_features["subfolder"],
      mode
    )
    H_CONG = read_class(path, f"series_{iterations}")

    edr_overlap = H_EDR.data_overlap.loc[
      (H_EDR.data_overlap.score == "_S") & (H_EDR.data_overlap.direction == "both")
    ]
    cong_overlap = H_CONG.data_overlap.loc[
      (H_CONG.data_overlap.score == "_S") & (H_CONG.data_overlap.direction == "both")
    ]

    mac_overlap = edr_overlap.loc[edr_overlap.data == "1"]
    edr_overlap = edr_overlap.loc[edr_overlap.data == "0"]
    edr_overlap["model"] = 'EDR'

    order = edr_overlap["Areas"].value_counts().sort_values(ascending=False)
    # edr_overlap["Areas"] = pd.Categorical(edr_overlap['Areas'], list(order.index))

    cong_overlap = cong_overlap.loc[cong_overlap.data == "0"]
    cong_overlap["model"] = "Configuration"

    order_conf = [s for s in np.unique(cong_overlap.Areas) if s not in order.index]

    # cong_overlap["Areas"] = pd.Categorical(cong_overlap['Areas'], list(order.index) + order_conf)

    # sns.set_context("talk")
    # plt.style.use("dark_background")
    sns.set_style("whitegrid")

    data = pd.concat([cong_overlap, edr_overlap], ignore_index=True)
    data["Areas"] = pd.Categorical(data['Areas'], list(order.index) + order_conf)

    fig, ax = plt.subplots(1, 1)

    sns.histplot(
        data = data.loc[data.direction == "both"],
        x = "Areas",
        hue = "model",
        stat = "count",
        discrete=True,
        hue_order=["EDR", "Configuration"],
        common_norm=False,
        palette="pastel",
        ax=ax
    )

    l = list(order.index) + order_conf
    [t.set_color("red") for i, t in enumerate(ax.xaxis.get_ticklabels()) if np.isin(l[i], mac_overlap.Areas)]

    plt.xticks(rotation=90)
   
    
    plt.ylabel("NOC probability")

    fig.set_figheight(9)
    fig.set_figwidth(15)
    fig.tight_layout()

    plot_path = plot_path + "/cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "overlap.png"
      ),
      dpi=300#,
      # transparent=T
    )
    plt.close()

def complete_network(plot_path):
    R = adj2df(np.log(1 + NET.C))
    R = R.loc[R.weight != 0]
    R["source_label"] = NET.struct_labels[R.source]
    R["target_label"] = NET.struct_labels[R.target]

    R = R.pivot("target_label", "source_label", "weight")
    sns.heatmap(
       data=R,
       cmap="crest", xticklabels=True, yticklabels=True
    )
    ax = plt.gca()
    ax.set_xlabel("source areas")
    ax.set_ylabel("target areas")
    fig = plt.gcf()
    fig.set_figwidth(17)
    fig.set_figheight(9)
    fig.tight_layout()
    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "network_heatmap.png"
      ),
      dpi=300
    )
    plt.close()

def Entropy_LBF_size(plot_path):
    lbf = read_scalefree_entropy(25)
    fig = plt.gcf()
    ax = plt.gca()
    sns.violinplot(
      data=lbf,
      x="model",
      y="S",
      hue="c",
      errorbar="sd",
      ax=ax
    )
    sns.barplot(
      data=lbf,
      x="model",
      y="S",
      hue="c",
      errorbar="sd",
      alpha=0.4,
      ax=ax
    )
    ax.set_ylabel(r"$s_{H}$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    fig.tight_layout()
    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "HEntropy_LBF_size.png"
      ),
      dpi=300
    )
    plt.close()

def entropy_ER(plot_path):
    path = "../pickle/TOY/ER/SINGLE_128_128_cut/ALPHA/MIX_jacp_trivial/b_0.0/"
    H_ER_jacp_128_06 = read_class(path, "series_500")
    H_ER_jacp_128_02 = read_class(path, "series_500_0.20_128")
    H_ER_jacp_128_03 = read_class(path, "series_500_0.30_128")
    H_ER_jacp_128_04 = read_class(path, "series_500_0.40_128")

    path = "../pickle/TOY/ER/SINGLE_250_250_cut/ALPHA/MIX_jacp_trivial/b_0.0/"
    H_ER_jacp_250_01 = read_class(path, "series_500_0.10_250")

    path = "../pickle/TOY/ER/SINGLE_50_50_cut/ALPHA/MIX_jacp_trivial/b_0.0/"
    H_ER_jacp_50_01 = read_class(path, "series_500_0.10_50")
    H_ER_jacp_50_02 = read_class(path, "series_500_0.20_50")

    ##

    er_128_06_data = pd.concat([H_ER_jacp_128_06.node_entropy, H_ER_jacp_128_06.link_entropy])
    er_128_06_data["model"] = "ER_128_06"

    er_128_02_data = pd.concat([H_ER_jacp_128_02.node_entropy, H_ER_jacp_128_02.link_entropy])
    er_128_02_data["model"] = "ER_128_02"

    er_128_03_data = pd.concat([H_ER_jacp_128_03.node_entropy, H_ER_jacp_128_03.link_entropy])
    er_128_03_data["model"] = "ER_128_03"

    er_128_04_data = pd.concat([H_ER_jacp_128_04.node_entropy, H_ER_jacp_128_04.link_entropy])
    er_128_04_data["model"] = "ER_128_04"

    er_250_01_data = pd.concat([H_ER_jacp_250_01.node_entropy, H_ER_jacp_250_01.link_entropy])
    er_250_01_data["model"] = "ER_250_01"

    er_50_01_data = pd.concat([H_ER_jacp_50_01.node_entropy, H_ER_jacp_50_01.link_entropy])
    er_50_01_data["model"] = "ER_50_01"

    er_50_02_data = pd.concat([H_ER_jacp_50_02.node_entropy, H_ER_jacp_50_02.link_entropy])
    er_50_02_data["model"] = "ER_50_02"

    data = pd.concat(
       [
          er_128_06_data, er_128_02_data, er_128_03_data, er_128_04_data,
          er_250_01_data, er_50_01_data, er_50_02_data
       ],
       ignore_index=True
    )
    data = data.loc[
       ((data.c == "node_hierarchy") |
       (data.c == "link_hierarchy") )&
       (data.dir == "H")
    ].groupby(["iter", "c", "model"]).sum()["S"].reset_index()

    ##

    fig = plt.gcf()
    ax = plt.gca()
    sns.violinplot(
      data=data,
      x="model",
      y="S",
      hue="c",
      errorbar="sd",
      ax=ax
    )
    sns.barplot(
      data=data,
      x="model",
      y="S",
      hue="c",
      errorbar="sd",
      alpha=0.4,
      ax=ax
    )
    ax.set_ylabel(r"$s_{H}$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    fig.tight_layout()
    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "HEntropy_ER.png"
      ),
      dpi=300
    )
    plt.close()

def target_bar_plot(PATH, NET):
  df = adj2df(NET.C)
  df["SOURCE"] = NET.struct_labels[df.source]
  df["TARGET"] = NET.struct_labels[df.target]
  data = df.loc[np.isin(df.TARGET, ["v1c", "v2c", "8l"])]
  data["Neural count"] = data.weight

  g = sns.catplot(
      data=data,
      x="SOURCE",
      y="Neural count",
      row="TARGET",
      kind="bar",
      color="b",
      aspect=4
  )
  for ax in g.axes.flat:
    labels = ax.get_yticklabels()
    ax.set_yticklabels(labels, size=20)
  plt.xticks(rotation=90)
  plt.yscale('log')
  plt.tight_layout()
  plt.savefig(
      PATH + "/cortex_letter/areal_count.png",
      dpi=300
  )

def directed_average_count(path, NET):
  D = NET.D.copy()[:, :__nodes__]
  C = NET.C.copy()

  labels = NET.struct_labels

  dC = adj2df(C)
  dC["SOURCE_INJ"] = labels[dC.source]
  dC["TARGET_INJ"] = labels[dC.target]
  dC = dC.loc[(dC.source < __nodes__) & (dC.target < __nodes__)]

  dD = adj2df(D)
  dD = dD.loc[(dD.source < __nodes__) & (dD.target < __nodes__)]

  dC["D"] = dD.weight

  dCs = dC.loc[np.isin(dC.SOURCE_INJ, ["v1c", "v2c", "10", "8l"])]
  dCs["dir"] = "source"
  dCs["Area_X"] = dCs.SOURCE_INJ
  dCs["Area_Y"] = dCs.TARGET_INJ 
  dCt = dC.loc[np.isin(dC.TARGET_INJ, ["v1c", "v2c", "10", "8l"])]
  dCt["dir"] = "target"
  dCt["Area_X"] = dCt.TARGET_INJ
  dCt["Area_Y"] = dCt.SOURCE_INJ

  CC = pd.concat(
    [dCs, dCt], ignore_index=True
  )

  CC = CC.loc[CC.weight > 0]
  CC["ANLNe"] = CC.weight

  CC = CC.sort_values("D", ascending=True)

  g = sns.FacetGrid(
      data=CC,
      col="Area_X",
      hue="dir",
      sharex=False,
      col_wrap=2
  )

  g.map_dataframe(
      sns.scatterplot,
      x="Area_Y",
      y="ANLNe"
  )

  g.map_dataframe(
      sns.lineplot,
      x="Area_Y",
      y="ANLNe"
  )

  g.add_legend()

  g.set_xticklabels(rotation=90)
  plt.yscale("log")
  plt.ylabel("ANLNe")
  fig = plt.gcf()
  fig.set_figheight(8)
  fig.set_figwidth(15)
  fig.tight_layout()
  plt.savefig(
    path + "/cortex_letter/directed_average_count.png",
    dpi=300
)

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
subject = "MAC"
structure = "FLNe"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
discovery = "discovery_7"
mapping = "trivial"
index  = "Hellinger2"
bias = float(0)
alpha = 0.
__nodes__ = 40
__inj__ = 40
version = f"{__nodes__}" + "d" + "91"
model_distbase = "M"
model_swaps = "TWOMX_FULL"

if __name__ == "__main__":
    NET = STR[f"{subject}{__inj__}"](
      linkage, mode,
      nlog10 = nlog10,
      structure = structure,
      lookup = lookup,
      version = version,
      nature = nature,
      model = imputation_method,
      distance = distance,
      inj = __inj__,
      discovery = discovery,
      topology = topology,
      index = index,
      mapping = mapping,
      cut = cut,
      b = bias,
      alpha = alpha
    )
    pickle_path = NET.pickle_path
    cortex_letter_path = os.path.join(
       "../plots", subject, version,
       structure, nature, distance, f"{__inj__}", NET.analysis
    )
    conf = {
       "subject" : NET.subject,
       "structure" : NET.structure,
       "version" : NET.version,
       "distance" : NET.distance,
       "subfolder" : NET.analysis,
       "model_distbase" : model_distbase,
       "model_swaps" : model_swaps
    }
    # entropy(cortex_letter_path, mode=mode, iterations=1000)
    omega(cortex_letter_path, conf, mode=mode, iterations=1000)
    # overlap(cortex_letter_path, conf,  mode=mode, iterations=1000)
    # entropy_networks_220830(cortex_letter_path, mode=mode, iterations=1000)
    # sim_dist(cortex_letter_path, mode=mode)
    # sim_dist_bin(cortex_letter_path, pickle_path, mode=mode)
    fair_dist_bin(cortex_letter_path, pickle_path, mode=mode)
    # sim_histogram(cortex_letter_path, mode=mode)
    # directed_average_count(cortex_letter_path, NET)
    