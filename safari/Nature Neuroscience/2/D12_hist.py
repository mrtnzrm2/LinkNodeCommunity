# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.abspath(SCRIPT_DIR))
SCRIPT_DIR = os.path.dirname(os.path.abspath(SCRIPT_DIR))
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

def  D12_hist_plot(nodes, pickle_path, ax : plt.Axes, cmap="deep"):


    path = pickle_path
    H = read_class(path, "hanalysis")

    label = r"$D_{1/2}^{\pm}$"
    # label_inset = "Interareal distances [mm]"

    src = -2 * np.log(H.source_sim_matrix)
    tgt = -2 * np.log(H.target_sim_matrix)
    D = H.D[:nodes, :nodes]

    src = adj2df(src)
    tgt = adj2df(tgt)
    D = adj2df(D)

    src = src["weight"].loc[src.source > src.target].to_numpy()
    tgt = tgt["weight"].loc[tgt.source > tgt.target].to_numpy()
    D = D["weight"].loc[D.source > D.target].to_numpy()

    # H_thr = 1-np.exp(-1)
    
    n = src.shape[0]
    data_main = pd.DataFrame({
        label : np.hstack([src, tgt]),
        "dataset" : ["+ (out)"] * n + ["- (in)"] * n
    })

    from scipy.stats import weibull_min as dist
    # from scipy.stats import gamma as dist
    # from scipy.stats import lognorm as dist
    k=2

    x = tgt[tgt < np.Inf]
    xmean = np.min(x)
    xstd = np.std(x)

    print(xmean, xstd)
    res = dist.fit(x[x>xmean], floc=xmean, scale=xstd)

    print(res)
    
    log_likelihood = dist.logpdf(np.sort(x[x>res[1]]), res[0], loc=res[1], scale=res[2])

    log_likelihood = log_likelihood.sum()
    AIC = 2*k - 2 *log_likelihood
    print(">>>", AIC)

    cmp = sns.color_palette(cmap)

    sns.histplot(
        data=data_main,
        x=label,
        palette=cmp[:2],
        hue="dataset",
        hue_order=["+ (out)", "- (in)"],
        stat="density",
        common_norm=False,
        alpha=1,
        multiple="dodge",
        ax=ax 
    )

    range_tgt = np.linspace(0, 16, 1000)[1:]
    y = dist.pdf(range_tgt, res[0], loc=res[1], scale=res[2])
    ax.plot(range_tgt, y, color="k", linewidth=2, alpha=0.7)
    ax.set_ylim(bottom=10**-3)
    
    ax.set_xlabel(label)
    ax.text(0.75, 0.5, r"$cx^{c-1}e^{-x^{c}}$", fontsize=15,
        color="k",
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes
    )

    import matplotlib.lines as mlines
    from matplotlib.patches import Patch

    orange_line = mlines.Line2D( 
        [], [], color="k", label='Weibull', lw=2
    )

    blue_rect = Patch(facecolor=cmp[0], edgecolor=None, label='+ (out)')
    orange_rect = Patch(facecolor=cmp[1], edgecolor=None, label='- (in)')

    ax.legend(handles=[blue_rect, orange_rect, orange_line], bbox_to_anchor=(0.5, 0.73), loc="center left")

    ax.get_legend().set_title("")


    # nbins = 200
    # n, bins, _ = plt.hist(x, nbins, density=True); 
    # centers = (0.5*(bins[1:]+bins[:-1]))
    # from scipy.optimize import curve_fit
    # pars, cov = curve_fit(lambda x, c, sig : dist.pdf(x[x>xmean], c, loc=xmean, scale=sig), centers, n, p0=[res[0], res[2]])
    # print(pars[0],np.sqrt(cov[0,0]), pars[1], np.sqrt(cov[1,1 ]))

    # known_params = {"c" : res[0], "loc" : res[1], "scale" : res[2]}
    # from scipy.stats import goodness_of_fit
    # res2 = goodness_of_fit(dist, x[x>xmean], statistic="cvm", known_params={"loc" : xmean})
    # print(res2)

    # par = ax.twinx()

    # Sf = 1/(1-np.exp(-0.5 * range_tgt[range_tgt>0]))

    # par.plot(range_tgt[range_tgt>0], Sf, color="k", linewidth=2, alpha=0.7)
    # par.set_ylabel(r"$N^{\delta}$")
    # par.set_yscale("log")

    # sns.move_legend(ax, "center left", bbox_to_anchor =(0.5, 0.65), ncol=1, title=None, frameon=True)    

