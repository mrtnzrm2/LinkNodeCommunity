# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Import libraries ----
## Standard libs ----
import itertools
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
## Personal libs ----
from various.network_tools import *

# Declare iter variables ----
benchmark = ["WDN"]
number_of_nodes = [1000]
topologies = ["MIX"]
indices = ["Hellinger2"]
# MUT = [0.1, 0.2, 0.3, 0.5]
MUT = np.linspace(0.1, 0.8, 10)
NMIN = [10]
NMAX = [50]
ON = [0.1]
OM = [3, 5, 8]
AUX = [10]
AUXLABEL = r"$\langle k \rangle$"
discovery = ["discovery_7"]
maxk = 50
list_of_lists = itertools.product(
  *[benchmark, number_of_nodes, topologies, indices, MUT, NMIN, NMAX, ON, OM, discovery, AUX]
)
list_of_lists = np.array(list(list_of_lists))
# Constant parameters ---
MAXI = 10
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
mapping = "trivial"
__mode__ = "ZERO"
l10 = ""
lup = ""
if nlog10: l10 = "_l10"
if lookup: lup = "_lup"
# opt_score = ["_S"]
if __name__ == "__main__":
  # Extract data ----
  THE_DF = pd.DataFrame()
  for mark, __nodes__, topology, index, mut, nmin, nmax, on, om, dis, aux in list_of_lists:
    __nodes__ = int(__nodes__)
    nmin = int(nmin)
    nmax = int(nmax)
    mut = float(mut)
    on = float(on)
    on = int(__nodes__ * on)
    om = int(om)
    if nmax < nmin: continue
    # WDN paramters ----
    par = {
      "-N" : "{}".format(str(__nodes__)),
      "-k" : f"{aux}.0",
      "-maxk" : f"{maxk}",
      "-mut" : f"{mut}",
      "-muw" : "0.01",
      "-beta" : "2",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : f"{nmin}",
      "-nmax" : f"{nmax}",
      "-on" : f"{on}",
      "-om" : f"{om}"
    }
    data = read_class(
      "../pickle/RAN/scalefree/{}/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/-on_{}/-om_{}/-fixed_range_True/{}/{}/{}/{}/{}".format(
        mark,
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        par["-nmin"], par["-nmax"],
        par["-on"], par["-om"], MAXI-1,
        __mode__, dis, __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )

    if isinstance(data, int): continue

    omega_label = r"$\omega$"
    mut_label = r"$\mu_{t}$"
    

    filtered_data = data.data.loc[
      (data.data["c"] != "_S2") &
      (data.data["sim"] == "OMEGA") & 
      (data.data["direction"] == "both")
    ]

    nrows = filtered_data.shape[0]

    THE_DF = pd.concat(
      [
        THE_DF, 
        pd.DataFrame(
          {
            omega_label : filtered_data["values"].to_numpy().astype(float),
            mut_label : [np.round(mut, 2)] * (nrows),
            "Om" : [f"{om}"] * (nrows),
            "nodes" : [int(__nodes__)] * (nrows),
            "benchmark" : [mark] * (nrows),
            "c" : [f[1:] for f in filtered_data["c"]],
            AUXLABEL : [aux] * (nrows)
          }
        )
      ], ignore_index=T
    )

  # Prepare plot ----
  IM_ROOT =  "../plots/RAN/scalefree/"

  # sns.set_context("talk")
  sns.set_style("ticks")
 
  g=sns.relplot(
    data=THE_DF,
    kind="line",
    col="Om",
    # row=AUXLABEL,
    # col_wrap=3,
    hue="c", x=mut_label, y=omega_label,
    style="c",
    style_order=["S", "D"],
    hue_order=["D", "S"],
    legend=False,
    alpha=0.8
  )

  # par = [("WN", 100), ("WN", 150), ("WDN", 100), ("WDN", 150)]
  # par = [("_S", 100), ("_S2", 100), ("_D", 100)]
  # par = [(2, 5), (3, 7), (4, 10)]


  # par = [("WDN", 100)]

  for i, ax in enumerate(g.axes.flatten()):
    # b, n = par[i]
    ax.minorticks_on()
    title = ax.get_title()
    # title = title.split(" | ")
    # n = title[0].split(" = ")[-1]
    # b = title[1].split(" = ")[-1]
    # subdata = THE_DF.loc[(THE_DF["Om"] == b) & (THE_DF[AUXLABEL] == n)]
    # subdata = subdata.groupby([mut_label, "Om", "c"])[omega_label].mean().reset_index()

    # title = title.split(" | ")
    n = title.split(" = ")[-1]
    # b = title[1].split(" = ")[-1]
    subdata = THE_DF.loc[(THE_DF["Om"] == n)]
    subdata = subdata.groupby([mut_label, "Om", "c"])[omega_label].mean().reset_index()
    sns.scatterplot(
      data=subdata,
      x=mut_label, y=omega_label,
      hue="c",
      style="c",
      markers=["o", "^"],
      hue_order=["D", "S"],
      style_order=["S", "D"],
      s=50,
      legend=False,
      ax=ax
    )

  cmap = sns.color_palette("deep")
  import matplotlib.lines as mlines

  orange_line = mlines.Line2D( 
      [], [], color=cmap[1], label=r"$S_{L}$", lw=1.5, marker="o", alpha=0.8, markeredgewidth=0.1, markersize=7
  )

  blue_line = mlines.Line2D( 
      [], [], color=cmap[0], label=r"$\langle D \rangle$", lw=1.5 , marker="^", alpha=0.8, markeredgewidth=0.1, markersize=7
  )

  legend = ax.legend(
    handles=[orange_line, blue_line],
    bbox_to_anchor=(0.7, 0.8),
    loc="center left",
    fontsize=12
  )

  legend.set_title("")
  frame = legend.get_frame()
  frame.set_linewidth(0.75)


  fig=plt.gcf()
  fig.set_figheight(3.5)
  fig.set_figwidth(7.08661)
  fig.tight_layout()

  Path(IM_ROOT).mkdir(exist_ok=True, parents=True)
  plt.savefig(
    os.path.join(
      IM_ROOT, f"omega_{AUXLABEL}_maxk{maxk}_{__nodes__}.svg"
    ),
    # dpi = 300,
    transparent=T
  )
  plt.savefig(
    os.path.join(
      IM_ROOT, f"omega_{AUXLABEL}_maxk{maxk}_{__nodes__}.png"
    ),
    dpi = 300,
    # transparent=T
  )
  plt.close()