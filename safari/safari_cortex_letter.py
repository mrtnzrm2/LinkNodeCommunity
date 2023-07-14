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
from pathlib import Path
# Personal libraries ----
from networks.structure import MAC
from various.network_tools import *

def omega(plot_path, mode="ALPHA", iterations=500):
  path = f"../pickle/RAN/distbase/MAC/57d106/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
  H_EDR = read_class(path, f"series_{iterations}")
  path = f"../pickle/RAN/swaps/MAC/57d106/LN/tracto16/1k/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
  H_CONG = read_class(path, f"series_{iterations}")
  edr_data =H_EDR.data
  cong_data = H_CONG.data
  edr_data["score"] = [s.replace("_", "") for s in edr_data["score"]]
  cong_data["score"] = [s.replace("_", "") for s in cong_data["score"]]
  edr_data = edr_data.loc[(edr_data.score == "S") & (edr_data.sim == "OMEGA")]
  cong_data = cong_data.loc[(cong_data.score == "S") & (cong_data.sim == "OMEGA")]

  data = pd.DataFrame(
    {
      "omega" : list(edr_data["values"]) + list(cong_data["values"]),
      "model" : ["EDR"] * edr_data.shape[0] + ["Configuration"] * cong_data.shape[0]
    }
  )

  print(data.groupby("model").mean())

  sns.histplot(
    data=data,
    x="omega",
    hue="model",
    stat="probability",
    common_bins=False,
    common_norm=False
  )

  ax = plt.gca()
  ax.set_xlabel("Omega index")
  fig = plt.gcf()
  fig.set_figheight(7)
  fig.set_figwidth(10)
  fig.tight_layout()
  plot_path = plot_path + "cortex_letter/"
  Path(plot_path).mkdir(exist_ok=True, parents=True)
  plt.savefig(
    join(
      plot_path, "omega.png"
    ),
    dpi=300
  )
  plt.close()

def entropy(plot_path, mode="ALPHA", iterations=500):
  path = f"../pickle/RAN/distbase/MAC/57d106/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
  H_EDR = read_class(path, f"series_{iterations}")
  path = f"../pickle/RAN/swaps/MAC/57d106/LN/tracto16/1k/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
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

  g = sns.FacetGrid(
    data=data,
    row = "c",
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
  sns.move_legend( g, "lower center", bbox_to_anchor=(0.5, 0), ncol=3, title=None, frameon=False)
  g.axes[0, 0].set_ylabel(r"$s_{H}$")
  g.axes[1, 0].set_ylabel(r"$s_{H}$")
  fig = plt.gcf()
  fig.tight_layout()
  fig.set_figheight(8)
  fig.set_figwidth(8)
  plot_path = plot_path + "cortex_letter/"
  Path(plot_path).mkdir(exist_ok=True, parents=True)
  plt.savefig(
    join(
      plot_path, "Entropy_levels.png"
    ),
    dpi=300
  )
  plt.close()

def read_scalefree_entropy(MAXI : int, mode="ALPHA"):
    import itertools
    from networks_serial.scalehrh import SCALEHRH
    topologies = ["MIX"]
    indices = ["D1_2_4"]
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
    path = f"../pickle/RAN/distbase/MAC/57d106/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
    H_EDR = read_class(path, f"series_{iterations}")
    path = f"../pickle/RAN/swaps/MAC/57d106/LN/tracto16/1k/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
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

    path = f"../pickle/RAN/distbase/MAC/57d106/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
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
        "S" : [0.2645, 0.2645, 0.4299, 0.4734, 0.4712, 0.2984],
        "c" : np.tile(hierarchy, 3)
      }
    )

    ##

    # lbf = read_scalefree_entropy(50, mode=mode)

    # lbf_models = list(np.unique(lbf.model))

    data = pd.concat(
       [erjacp_data, hrgjacp_data, edr_data, cong_data, mac_data, michelle]
    )
    fig = plt.gcf()
    ax = plt.gca()
    order = ["Zachary", "ER", "HRG", "HSF", "Configuration", "EDR", "data", "Tractography"] #+ lbf_models
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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, size=20)
    fig.set_figheight(9)
    fig.set_figwidth(12)
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
    path = f"../pickle/MAC/57d106/LN/original/tracto16/57/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
    H = read_class(path, "hanalysis")
    src = H.source_sim_matrix
    tgt = H.target_sim_matrix
    
    src = adj2df(src)
    tgt = adj2df(tgt)

    src = src.loc[src.source > src.target]
    tgt = tgt.loc[tgt.source > tgt.target]

    data = pd.DataFrame(
       {
        r"$D_{1/2}$" : list(np.log(1 / src.weight - 1)) + list(np.log(1 / tgt.weight - 1)) ,
        "direction" : ["outgoing"] * src.shape[0] + ["incoming"] * tgt.shape[0]
       }
    )

    sns.histplot(
       data=data,
       x=r"$D_{1/2}$",
       hue="direction",
       stat="density",
       common_norm=False
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel(r"$\log(D_{1/2})$")
    fig.set_figheight(7)
    fig.set_figwidth(10)
    fig.tight_layout()

    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "D1_2_2_dir.png"
      ),
      dpi=300
    )
    plt.close()

def sim_dist(plot_path, mode="ALPHA"):
    path = f"../pickle/MAC/57d106/LN/original/tracto16/57/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
    H = read_class(path, "hanalysis")
    src = H.source_sim_matrix
    tgt = H.target_sim_matrix
    D = NET.D[:__nodes__, :__nodes__]
    
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

    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "D1_2_2_dist.png"
      ),
      dpi=300
    )
    plt.close()

def sim_dist_bin(plot_path, mode="ALPHA"):
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    path = f"../pickle/MAC/57d106/LN/original/tracto16/57/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
    H = read_class(path, "hanalysis")
    src = H.source_sim_matrix
    tgt = H.target_sim_matrix
    D = NET.D[:__nodes__, :__nodes__]
    
    src = adj2df(src)
    tgt = adj2df(tgt)
    D = adj2df(D)

    src = 1 / src.loc[src.source > src.target].weight.to_numpy() - 1
    src[src > 100] = np.nan
    tgt = 1 / tgt.loc[tgt.source > tgt.target].weight.to_numpy() - 1
    D = D.loc[D.source > D.target].weight.to_numpy()

    ## linear regression statistics ----
    scale= StandardScaler()
    X = NET.D.copy()
    X[X <= 0] = np.nan
    n = X.shape[0]
    X = scale.fit_transform(X.reshape(-1, 1))
    X = X.reshape(n, n)
    X = X[:__nodes__, :__nodes__]
    X = adj2df(X)
    X = X.loc[X.source > X.target].weight.to_numpy()
    yup_d = ~np.isnan(X)
    Ysrc = src.copy()
    yup_src = (~np.isnan(Ysrc)) & (Ysrc != np.inf) & (yup_d)
    Ysrc = Ysrc[yup_src]
    Ytgt = tgt.copy()
    yup_tgt = (~np.isnan(Ytgt)) & (Ytgt != np.inf) & (yup_d)
    Ytgt = Ytgt[yup_tgt]
    Xsrc = X[yup_src].reshape(-1,1)
    Xtgt = X[yup_tgt].reshape(-1,1)
    # Xsrc = np.hstack([Xsrc, np.power(Xsrc, 3.)])
    # Xtgt = np.hstack([Xtgt, np.power(Xtgt, 3.)])
    Xsrc = sm.add_constant(Xsrc)
    Xtgt = sm.add_constant(Xtgt)
    est_src = sm.OLS(Ysrc, Xsrc).fit()
    est_tgt = sm.OLS(Ytgt, Xtgt).fit()
    print(est_src.summary())
    print(est_tgt.summary())

    data_resid =pd.DataFrame(
       {
          "residuals" : list(est_src.resid) + list(est_tgt.resid),
          r"$D_{1/2}$" : list(Ysrc) + list(Ytgt),
          "direction" : ["outgoing"] * len(Ysrc) + ["incoming"] * len(Ytgt)
       }
    )

    ### ----

    data = pd.DataFrame(
       {
        r"$D_{1/2}$" : list(src) + list(tgt) ,
        "direction" : ["outgoing"] * src.shape[0] + ["incoming"] * tgt.shape[0],
        "interareal tractography distance [mm]" : list(D) + list(D) 
       }
    )

    # Binarize ----
    nbin = 15
    dD = (np.max(D) - np.min(D)) / nbin
    d_range = np.arange(np.min(D), np.max(D), dD)
    d_range[-1] += 0.0001
    src_bin = [np.nanmean(src[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)]
    tgt_bin = [np.nanmean(tgt[np.where((D >= d_range[i]) & (D < d_range[i+1]))[0]]) for i in np.arange(nbin - 1)]
    d_range[-1] -= 0.0001
    d_range = d_range[:-1]
    d_range = d_range + dD / 2
    data_bin = pd.DataFrame(
       {
          r"$D_{1/2}$" : list(src_bin) + list(tgt_bin) ,
          "direction" : ["outgoing"] * len(src_bin) + ["incoming"] * len(src_bin),
          "interareal tractography distance [mm]" : list(d_range) + list(d_range) 
       }
    )

    fig, ax = plt.subplots(1, 1)
    
    sns.lmplot(
       data=data_bin,
       x="interareal tractography distance [mm]",
       y = r"$D_{1/2}$",
       hue="direction",
       scatter_kws={"alpha":0.9, "s":8},
       line_kws={"linewidth":1},
       legend=False,
       aspect=1.3
    )
    sns.scatterplot(
       data=data,
       x="interareal tractography distance [mm]",
      y= r"$D_{1/2}$",
      hue="direction",
      alpha=0.3,
      s=4
    )
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
    # fig.set_figheight(5)
    # fig.set_figwidth(9.5)
    # fig.tight_layout()

    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "D1_2_2_dist_bin.png"
      ),
      dpi=300
    )
    plt.close()


def overlap(plot_path, mode="ALPHA", iterations=500):
    path = f"../pickle/RAN/distbase/MAC/57d106/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
    H_EDR = read_class(path, f"series_{iterations}")
    path = f"../pickle/RAN/swaps/MAC/57d106/LN/tracto16/1k/SINGLE_106_57_l10/{mode}/MIX_D1_2_4_trivial/b_0.0/"
    H_CONG = read_class(path, f"series_{iterations}")

    edr_overlap = H_EDR.data_overlap.loc[
      H_EDR.data_overlap.score == "_S"
    ]
    cong_overlap = H_CONG.data_overlap.loc[
      H_CONG.data_overlap.score == "_S"
    ]

    mac_overlap = edr_overlap.loc[edr_overlap.data == "1"]
    edr_overlap = edr_overlap.loc[edr_overlap.data == "0"]
    edr_overlap["model"] = 'EDR'
    cong_overlap = cong_overlap.loc[cong_overlap.data == "0"]
    cong_overlap["model"] = "Configuration"

    data = pd.concat([edr_overlap, cong_overlap])

    sns.histplot(
        data = data,
        x = "Areas",
        hue = "model",
        stat = "count",
        common_norm=False,
    )
    fig = plt.gcf()
    fig.set_figheight(7)
    fig.set_figwidth(12)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    [t.set_color("red") for i, t in enumerate(ax.xaxis.get_ticklabels()) if np.isin(t.get_text(), mac_overlap.Areas)]
    plt.ylabel("NOC count")
    plot_path = plot_path + "cortex_letter/"
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      join(
        plot_path, "overlap.png"
      ),
      dpi=300
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
  CC["Average neural count"] = CC.weight

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
      y="Average neural count"
  )

  g.map_dataframe(
      sns.lineplot,
      x="Area_Y",
      y="Average neural count"
  )

  g.add_legend()

  g.set_xticklabels(rotation=90)
  plt.yscale("log")
  plt.ylabel("Average neural count")
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
bias = float(0)
alpha = 0.
opt_score = ["_X", "_S"]
save_data = T
version = "57d106"
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
    NET = MAC[f"MAC{__inj__}"](
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
    cortex_letter_path = "../plots/MAC/57d106/LN/original/tracto16/57/SINGLE_106_57_l10/"
    # entropy(cortex_letter_path, mode=mode)
    # omega(cortex_letter_path, mode=mode)
    # overlap(cortex_letter_path, mode=mode)
    entropy_networks_220830(cortex_letter_path, mode=mode)
    # sim_dist(cortex_letter_path, mode=mode)
    # sim_dist_bin(cortex_letter_path, mode=mode)
    # sim_histogram(cortex_letter_path, mode=mode)
    # directed_average_count(cortex_letter_path, NET)
    