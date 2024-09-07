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

def  H2_bin_plot(nodes, pickle_path, ax : plt.Axes, cmap="deep"):
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    from seaborn import utils
    import ctools as ct

    path = pickle_path
    H = read_class(path, "hanalysis")

    wlabel = r"$H^{2}_{\pm}$"
    w_inset = r"$D_{1/2}_{\pm}$"

    # src = np.sqrt(1 - H.source_sim_matrix)
    # tgt = np.sqrt(1 - H.target_sim_matrix)
    src = (1 - H.source_sim_matrix)
    tgt = (1 - H.target_sim_matrix)
    D = H.D[:nodes, :nodes]

    src_inset = np.zeros(src.shape)
    tgt_inset = np.zeros(tgt.shape)

    for i in np.arange(src_inset.shape[0]):
       for j in np.arange(i + 1, src_inset.shape[1]):
          src_inset[i, j] = -2 * np.log(ct.Hellinger2(H.A[i, :], H.A[j, :], i, j))
          src_inset[j, i] = src_inset[i, j]

          tgt_inset[i, j] = -2 * np.log(ct.Hellinger2(H.A[:, i], H.A[:, j], i, j))
          tgt_inset[j, i] = tgt_inset[i, j]

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

    ordD = np.argsort(D)
    D = D[ordD]
    src = src[ordD]
    tgt = tgt[ordD]
    src_inset = src_inset[ordD]
    tgt_inset = tgt_inset[ordD]

    Dmu = D.copy()
    mu_D = np.mean(D)
    std_D = np.std(D)
    Dmu -= mu_D
    Dmu /= std_D

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
    nbin = 10

    D_min, D_max = np.min(D), np.max(D)
    dD = (D_max - D_min) / nbin
    boundaries = np.arange(D_min, D_max + 1e-5, dD)
    centers = boundaries.copy()[:-1]
    centers += dD / 2
    boundaries[-1] += 1e-5

    src_bin = [np.nanmean(src[np.where((D >= boundaries[i]) & (D < boundaries[i+1]))[0]]) for i in np.arange(nbin)]
    src_bin = np.array(src_bin)

    tgt_bin = [np.nanmean(tgt[np.where((D >= boundaries[i]) & (D < boundaries[i+1]))[0]]) for i in np.arange(nbin)]
    tgt_bin = np.array(tgt_bin)

    src_inset_bin = [np.nanmean(src_inset[np.where((D >= boundaries[i]) & (D < boundaries[i+1]))[0]]) for i in np.arange(nbin)]
    src_inset_bin = np.array(src_inset_bin)

    tgt_inset_bin = [np.nanmean(tgt_inset[np.where((D >= boundaries[i]) & (D < boundaries[i+1]))[0]]) for i in np.arange(nbin)]
    tgt_inset_bin = np.array(tgt_inset_bin)

    var_src_inset_bin = [np.nanvar(src_inset[np.where((D >= boundaries[i]) & (D < boundaries[i+1]))[0]]) for i in np.arange(nbin)]
    var_src_inset_bin = np.array(var_src_inset_bin)

    var_tgt_inset_bin = [np.nanvar(tgt_inset[np.where((D >= boundaries[i]) & (D < boundaries[i+1]))[0]]) for i in np.arange(nbin)]
    var_tgt_inset_bin = np.array(var_tgt_inset_bin)

    boundaries[-1] -= 1e-5

    D_min_mu, D_max_mu = np.min(Dmu), np.max(Dmu)
    dD = (D_max_mu - D_min_mu) / nbin
    boundaries_mu = np.arange(D_min_mu, D_max_mu + 1e-5, dD)
    centers_mu = boundaries_mu.copy()[:-1]
    centers_mu += dD / 2

    data_bin = pd.DataFrame(
       {
          wlabel : list(src_bin) + list(tgt_bin) ,
          w_inset : list(src_inset_bin) + list(tgt_inset_bin),
          "direction" : ["source"] * len(src_bin) + ["target"] * len(src_bin),
          "interareal distance [mm]" : list(centers) + list(centers) 
       }
    )

    data_bin = data_bin.loc[data_bin[w_inset] < np.Inf]

    Xsrc = centers_mu.copy().reshape(-1, 1)
    Xsrc = sm.add_constant(Xsrc)
    Xtgt = centers_mu.copy().reshape(-1, 1)
    Xtgt = sm.add_constant(Xtgt)

    # llf_src =  -np.Inf
    # llf_tgt = -np.Inf

    # for i in np.linspace(0.1, 1, 100):

    #   est_src = sm.WLS(src_inset, Xsrc, weights=1/np.power(src_inset, i)).fit()
    #   est_tgt = sm.WLS(tgt_inset, Xtgt, weights=1/np.power(tgt_inset, i)).fit()

    #   if llf_src < est_src.llf:
    #     llf_src = est_src.llf
    #     isrc = i
    #     print("*", llf_src, i)

    #   if llf_tgt < est_tgt.llf:
    #     llf_tgt = est_tgt.llf
    #     itgt = i
    #     print("**", llf_tgt, i)

    est_src = sm.WLS(src_inset_bin, Xsrc, weights=1/var_src_inset_bin).fit()
    
    # print(np.abs(est_src.conf_int()[1:] - est_src.params[1])[0] / std_D)
    alpha_src_err = np.abs(est_src.conf_int()[1:] - est_src.params[1])[0] / std_D
    alpha_src_err = alpha_src_err[0]
    # print(src_inset_bin, "\n", tgt_inset_bin)

    pred_wls_src = est_src.get_prediction()
    iv_l_src = pred_wls_src.summary_frame()["obs_ci_lower"]
    iv_u_src = pred_wls_src.summary_frame()["obs_ci_upper"]
    # print(iv_u_src - iv_l_src)

    est_tgt = sm.WLS(tgt_inset_bin, Xtgt, weights=1/var_tgt_inset_bin).fit()

    # print(np.abs(est_tgt.conf_int()[1:] - est_tgt.params[1])[0] / std_D)
    alpha_tgt_err = np.abs(est_tgt.conf_int()[1:] - est_tgt.params[1])[0] / std_D
    alpha_tgt_err = alpha_tgt_err[0]
    # print( var_src_inset_bin, "\n", var_tgt_inset_bin)

    pred_wls_tgt = est_tgt.get_prediction()
    # print(pred_wls_tgt.summary_frame()) 
    iv_l_tgt = pred_wls_tgt.summary_frame()["obs_ci_lower"]
    iv_u_tgt = pred_wls_tgt.summary_frame()["obs_ci_upper"]
    # print(iv_u_tgt - iv_l_tgt)

    span = np.linspace(D_min, D_max, 100)
    span_2 = np.linspace(D_min_mu, D_max_mu, 100)
    span_2 = sm.add_constant(span_2.reshape(-1, 1))

    D12_from_src = est_src.predict(span_2)
    D12_from_tgt = est_tgt.predict(span_2)


    # print(est_src.bse)
    # print(est_src.params)
    # print(est_tgt.bse)
    # print(est_tgt.params)

    est_src_bse = est_src.bse
    est_src_bse[1] /= std_D
    est_tgt_bse = est_tgt.bse
    est_tgt_bse[1] /= std_D
    
    est_src = est_src.params
    est_tgt = est_tgt.params


    print("source lambda:\n", est_src[1] / std_D)
    print("target lambda:\n", est_tgt[1] / std_D)

    est_src[1] /= std_D
    est_src[0] -= est_src[1] * mu_D

    print("source characteristic length:", (2-est_src[0]) / est_src[1])
    
    est_tgt[1] /= std_D
    est_tgt[0] -= est_tgt[1] * mu_D

    print("target characteristic length:", (2-est_tgt[0]) / est_tgt[1])
    
    # line of H(d) using the one in D_1/2(d) ----
    
    # H_from_linear_src = np.sqrt(1 - np.exp(-0.5 *( est_src[1] * span + est_src[0])))
    # H_iv_l_src = np.sqrt(1 - np.exp(-0.5 * iv_l_src))
    # H_iv_l_src[np.isnan(H_iv_l_src)] = 0
    # H_iv_u_src = np.sqrt(1 - np.exp(-0.5 * iv_u_src))

    # H_from_linear_tgt = np.sqrt(1 - np.exp(-0.5 *( est_tgt[1] * span + est_tgt[0])))
    # H_iv_l_tgt = np.sqrt(1 - np.exp(-0.5 * iv_l_tgt))
    # H_iv_l_tgt[np.isnan(H_iv_l_tgt)] = 0
    # H_iv_u_tgt = np.sqrt(1 - np.exp(-0.5 * iv_u_tgt))

    H_from_linear_src = (1 - np.exp(-0.5 *( est_src[1] * span + est_src[0])))
    H_iv_l_src = (1 - np.exp(-0.5 * iv_l_src))
    H_iv_l_src[np.isnan(H_iv_l_src)] = 0
    H_iv_u_src = (1 - np.exp(-0.5 * iv_u_src))

    H_from_linear_tgt = (1 - np.exp(-0.5 *( est_tgt[1] * span + est_tgt[0])))
    H_iv_l_tgt = (1 - np.exp(-0.5 * iv_l_tgt))
    H_iv_l_tgt[np.isnan(H_iv_l_tgt)] = 0
    H_iv_u_tgt = (1 - np.exp(-0.5 * iv_u_tgt))


    cmp = sns.color_palette(cmap)

    ax.fill_between(
        centers, H_iv_l_tgt, H_iv_u_tgt, color=cmp[1],
        alpha=0.2, edgecolor=cmp[1], linewidth=3, linestyle="--"
    )

    ax.fill_between(
        centers, H_iv_l_src, H_iv_u_src, color=cmp[0],
        alpha=0.2, edgecolor=cmp[0], linewidth=3, linestyle="--"
    )

    sns.scatterplot(
      data=data,
      x="interareal distance [mm]",
      y= wlabel,
      hue="direction",
      hue_order=["source", "target"],
      style="direction",
      markers=["^", "o"],
      alpha=0.4,
      palette=cmp[:2],
      s=6,
      ax=ax
    )

    sns.lineplot(
        x=span,
        y=H_from_linear_src,
        color=cmp[0],
        linestyle="-",
        alpha=0.7,
        linewidth=3,
        ax=ax
    )

    sns.lineplot(
        x=span,
        y=H_from_linear_tgt,
        color=cmp[1],
        linestyle="-",
        linewidth=3,
        alpha=0.7,
        ax=ax
    )

    sns.scatterplot(
       data=data_bin.loc[data_bin.direction == "target"],
       x="interareal distance [mm]",
       y = wlabel,
       color=cmp[1],
       alpha=0.8,
       edgecolor="k",
       ax=ax
    )

    sns.scatterplot(
       data=data_bin.loc[data_bin.direction == "source"],
       x="interareal distance [mm]",
       y = wlabel,
       markers=["^"],
       style="direction",
       color=cmp[0],
       alpha=0.8,
       edgecolor="k",
       ax=ax
    )

    import matplotlib.lines as mlines

    blue_line = mlines.Line2D(
        [], [], color=cmp[0], marker='^',
        markersize=10, label='+ (out)', markeredgecolor="k", markeredgewidth=0.5
    )
    orange_line = mlines.Line2D(
        [], [], color=cmp[1], marker='o',
        markersize=10, label='- (in)', markeredgecolor="k", markeredgewidth=0.5
    )

    ax.legend(handles=[blue_line, orange_line], bbox_to_anchor=(0.2, 0.14), loc="center left")
    plt.xlabel("Interareal distance [mm]")
    # sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5), ncol=1, title=None, frameon=False)

    # plt.savefig(
    #   "../Publication/Nature Neuroscience/Figures/2/dis_funct_v1.pdf"
    # )

    # plt.show()
 

# linkage = "single"
# nlog10 = F
# lookup = F
# prob = T
# cut = F
# subject = "MAC"
# structure = "FLNe"
# mode = "ZERO"
# nature = "original"
# imputation_method = ""
# topology = "MIX"
# discovery = "discovery_7"
# mapping = "trivial"
# index  = "Hellinger2"
# bias = float(0)
# alpha = 0.
# __nodes__ = 40
# __inj__ = 40
# distance = "MAP3D"
# version = f"{__nodes__}" + "d" + "91"
# model_distbase = "M"
# model_swaps = "TWOMX_FULL" # 1K_DENSE

# if __name__ == "__main__":
#     NET = STR[f"{subject}{__inj__}"](
#       linkage, mode,
#       nlog10 = nlog10,
#       structure = structure,
#       lookup = lookup,
#       version = version,
#       nature = nature,
#       model = imputation_method,
#       distance = distance,
#       inj = __inj__,
#       discovery = discovery,
#       topology = topology,
#       index = index,
#       mapping = mapping,
#       cut = cut,
#       b = bias,
#       alpha = alpha
#     )
#     pickle_path = NET.pickle_path
#     cortex_letter_path = os.path.join(
#        "../plots", subject, version,
#        structure, nature, distance, f"{__inj__}", NET.analysis
#     )
#     conf = {
#        "subject" : NET.subject,
#        "structure" : NET.structure,
#        "version" : NET.version,
#        "distance" : NET.distance,
#        "subfolder" : NET.analysis,
#        "model_distbase" : model_distbase,
#        "model_swaps" : model_swaps
#     }

#     sns.set_style("ticks")
#     sns.set_context("talk")

#     fig, ax = plt.subplots(1)
#     ax.minorticks_on()
#     sim_dist_bin_plot(pickle_path, ax)

    