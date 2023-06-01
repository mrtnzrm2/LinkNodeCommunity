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
# Personal libraries ----
from networks.structure import MAC
from various.network_tools import *

def omega():
  path = "../pickle/RAN/distbase/MAC/220830/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
  H_EDR = read_class(path, "series_500")
  path = "../pickle/RAN/swaps/MAC/220830/LN/tracto16/1k/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
  H_CONG = read_class(path, "series_500")
  edr_data =H_EDR.data
  cong_data = H_CONG.data
  edr_data["score"] = [s.replace("_", "") for s in edr_data["score"]]
  cong_data["score"] = [s.replace("_", "") for s in cong_data["score"]]
  edr_data = edr_data.loc[(edr_data.score == "maxmu") & (edr_data.sim == "OMEGA")]
  cong_data = cong_data.loc[(cong_data.score == "maxmu") & (cong_data.sim == "OMEGA")]

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
    common_norm=False
  )

  ax = plt.gca()
  ax.set_xlabel("Omega index")
  fig = plt.gcf()
  fig.set_figheight(7)
  fig.set_figwidth(10)
  fig.tight_layout()
  plot_path = "../plots/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/cortex_letter/"
  plt.savefig(
    join(
      plot_path, "omega.png"
    ),
    dpi=300
  )
  plt.close()

def entropy():
  path = "../pickle/RAN/distbase/MAC/220830/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
  H_EDR = read_class(path, "series_500")
  path = "../pickle/RAN/swaps/MAC/220830/LN/tracto16/1k/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
  H_CONG = read_class(path, "series_500")
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
  plot_path = "../plots/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/cortex_letter/"
  plt.savefig(
    join(
      plot_path, "Entropy_levels.png"
    ),
    dpi=300
  )
  plt.close()

def read_scalefree_entropy(MAXI : int):
    import itertools
    from networks_serial.scalehrh import SCALEHRH
    topologies = ["SOURCE"]
    indices = ["bsim"]
    KAV = [10]
    MUT = [0.1, 0.3, 0.5]
    MUW = [0.3]
    NMIN = [10, 50]
    NMAX = [20, 100]
    list_of_lists = itertools.product(
      *[topologies, indices, KAV, MUT, MUW, NMIN, NMAX]
    )
    list_of_lists = np.array(list(list_of_lists))
    __nodes__ = 1000
    nlog10 = F
    lookup = F
    prob = F
    run = T
    maxk = 50
    beta = 3
    t1 = 2
    t2 = 1
    mapping = "trivial"
    __mode__ = "ALPHA"
    cut = F
    LBF = pd.DataFrame()
    for topology, index, kav, mut, muw, nmin, nmax in list_of_lists:
      nmin = int(nmin)
      nmax = int(nmax)
      if nmin > nmax: continue
      # WDN paramters ----
      par = {
        "-N" : "{}".format(str(__nodes__)),
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
        "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/{}/{}/{}/{}".format(
          str(__nodes__),
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
        scale_data["model"] = f"LBF_{mut}"
        LBF = pd.concat([LBF, scale_data])
    return LBF


def entropy_networks():
    path = "../pickle/RAN/distbase/MAC/220830/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
    H_EDR = read_class(path, "series_500")
    path = "../pickle/RAN/swaps/MAC/220830/LN/tracto16/1k/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
    H_CONG = read_class(path, "series_500")
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

    path = "../pickle/RAN/distbase/MAC/220830/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
    H_EDR = read_class(path, "series_500")
    mac_data = pd.concat([H_EDR.node_entropy, H_EDR.link_entropy])
    mac_data = mac_data.loc[
       ((mac_data.c == "node_hierarchy") | (mac_data.c == "link_hierarchy")) &
       (mac_data.dir == "H") & (mac_data.data == "1")
    ].groupby("c").sum()["S"].reset_index()
    mac_data["model"] = "data"

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

    lbf = read_scalefree_entropy(25)

    data = pd.concat([erjacp_data, hrgjacp_data, lbf, edr_data, cong_data, mac_data, michelle])
    fig = plt.gcf()
    ax = plt.gca()
    order = ["Zachary", "ER", "HRG", "HSF", "Configuration", "EDR", "data", "LBF_0.1", "LBF_0.3", "LBF_0.5"]
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
    fig.set_figwidth(12)
    fig.tight_layout()
    
    plot_path = "../plots/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/cortex_letter/"
    plt.savefig(
      join(
        plot_path, "Entropy_networks.png"
      ),
      dpi=300
    )
    plt.close()

def sim_histogram():
    path = "../pickle/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0"
    H = read_class(path, "hanalysis")
    src = H.source_sim_matrix
    tgt = H.target_sim_matrix
    
    src = adj2df(src)
    tgt = adj2df(tgt)

    src = src.loc[src.source > src.target]
    tgt = tgt.loc[tgt.source > tgt.target]

    data = pd.DataFrame(
       {
        "Jaclog" : list(src.weight) + list(tgt.weight) ,
        "direction" : ["outgoing"] * src.shape[0] + ["incoming"] * tgt.shape[0]
       }
    )

    sns.histplot(
       data=data,
       x="Jaclog",
       hue="direction",
       stat="probability"
    )
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_figheight(7)
    fig.set_figwidth(10)
    fig.tight_layout()

    plot_path = "../plots/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/cortex_letter/"
    plt.savefig(
      join(
        plot_path, "jaclog_dir.png"
      ),
      dpi=300
    )
    plt.close()

def sim_dist():
    path = "../pickle/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0"
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
        "Jaclog" : list(src.weight) + list(tgt.weight) ,
        "direction" : ["outgoing"] * src.shape[0] + ["incoming"] * tgt.shape[0],
        "dist" : list(D.weight) + list(D.weight) 
       }
    )

    sns.scatterplot(
       data=data,
       x="dist",
       y="Jaclog",
       hue="direction",
       alpha = 0.7
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("tractography distance [mm]")
    fig.set_figheight(7)
    fig.set_figwidth(10)
    fig.tight_layout()

    plot_path = "../plots/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/cortex_letter/"
    plt.savefig(
      join(
        plot_path, "jaclog_dist.png"
      ),
      dpi=300
    )
    plt.close()


def overlap():
    path = "../pickle/RAN/distbase/MAC/220830/LN/tracto16/EXPMLE/BIN_12/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
    H_EDR = read_class(path, "series_500")
    path = "../pickle/RAN/swaps/MAC/220830/LN/tracto16/1k/SINGLE_106_57_l10/ALPHA/MIX_simple2_trivial/b_0.0/"
    H_CONG = read_class(path, "series_500")

    edr_overlap = H_EDR.data_overlap.loc[
      H_EDR.data_overlap.score == "_maxmu"
    ]
    cong_overlap = H_CONG.data_overlap.loc[
      H_CONG.data_overlap.score == "_maxmu"
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
    
    plot_path = "../plots/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/cortex_letter/"
    plt.savefig(
      join(
        plot_path, "overlap.png"
      ),
      dpi=300
    )
    plt.close()

def complete_network():
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
    plot_path = "../plots/MAC/220830/LN/original/tracto16/57/SINGLE_106_57_l10/cortex_letter/"
    plt.savefig(
      join(
        plot_path, "network_heatmap.png"
      ),
      dpi=300
    )
    plt.close()


# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
structure = "LN"
mode = "ALPHA"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "simple2"
bias = float(0)
opt_score = ["_maxmu", "_X", "_D"]
save_data = T
version = 220830
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
    NET = MAC(
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
      b = bias
    )
    entropy()

