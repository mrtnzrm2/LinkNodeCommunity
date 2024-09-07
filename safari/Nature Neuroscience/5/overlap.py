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
from pathlib import Path
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

def overlap(common_features, ax : plt.Axes, mode="ALPHA", iterations=500, cmap="deep"):
    
    path = "../pickle/RAN/distbase/{}/{}/FLNe/{}/MTRUNC/BIN_12/{}/{}/MIX_Hellinger2_trivial/discovery_7".format(
      common_features["subject"],
      common_features["version"],
      common_features["distance"],
      common_features["subfolder"],
      mode
    )
    H_EDR = read_class(path, f"series_{iterations}")

    path = "../pickle/RAN/swaps/{}/{}/FLNe/{}/TWOMX_FULL/{}/{}/MIX_Hellinger2_trivial/discovery_7".format(
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

    data = pd.concat([cong_overlap, edr_overlap], ignore_index=True)
    data["Areas"] = pd.Categorical(data['Areas'], list(order.index) + order_conf)

    # ax.hist(
        
    # )

    sns.histplot(
        data = data.loc[data.direction == "both"],
        x = "Areas",
        hue = "model",
        stat = "count",
        multiple="dodge",
        hue_order=["EDR", "Configuration"],
        common_norm=False,
        palette=cmap,
        ax=ax
    )

    l = list(order.index) + order_conf
    
    ax_tick_loc = np.arange(40, step=2)
    ax.set_xticks(ax_tick_loc)
    ax.set_xticklabels([a for i, a in enumerate(l) if i % 2 == 0], rotation=90, fontsize=10)

    [t.set_color("red") for i, t in enumerate(ax.xaxis.get_ticklabels()) if np.isin(t.get_text(), mac_overlap.Areas)]


    ax2 = ax.twiny()
    ax2_tick_loc = np.linspace(0.05, 0.95, 20) + 0.05/2
    ax2.set_xticks(ax2_tick_loc)
    ax2.set_xticklabels([a for i, a in enumerate(l) if i % 2 == 1], rotation=90, fontsize=10)

    [t.set_color("red") for i, t in enumerate(ax2.xaxis.get_ticklabels()) if np.isin(t.get_text(), mac_overlap.Areas)]

    ax.tick_params(axis="x", which="minor", bottom=False)

    plt.ylabel("NOC probability")