# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
from pathlib import Path
# Personal libraries ----
from networks.structure import MAC
from various.network_tools import *
import ctools as ct


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
discovery = "discovery_6"
bias = 0.
alpha = 0.
opt_score = ["_X", "_S"]
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
      alpha = alpha,
      discovery = discovery
    )

    data = pd.DataFrame()
    pred = pd.DataFrame()

    X = NET.D.copy()
    np.fill_diagonal(X, np.nan)
    mux = np.nanmean(X)
    sdx = np.nanstd(X)
    X = (X - mux) / sdx
    np.fill_diagonal(X, 0.)
    A = ["stpc", "8l", "f1", "1", "teo", "v1fplf"]

    for a in A:
        iA = match([a], NET.labels)

    

        x= X[iA, :].ravel()
        xtarget = X[:, iA].ravel()
        dx = np.linspace(np.min(x), np.max(x), 100)

        x = x[:NET.nodes]
        Y = np.log(1 + NET.C[iA, :]).ravel()
        Ytarget = np.log(1 + NET.C[:, iA]).ravel()

        yno = Y == 0
        x = x[~yno]
        Y = Y[~yno]

        ynotarget = Ytarget == 0
        xtarget = xtarget[~ynotarget]
        Ytarget = Ytarget[~ynotarget]
        Ntarget = Ytarget.shape[0]

        k = 5
        N = x.shape[0]
        ntrain = int(N * (k-1) / k)
        ntest = N - ntrain

        perm = np.random.permutation(np.arange(N))

        Xtrain = x[perm[:ntrain]]
        Xtest = x[perm[ntrain:]]

        Ytrain = Y[perm[:ntrain]]
        Ytest = Y[perm[ntrain:]]

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        noise_std = 1.5
        kernel = 1 * RBF()
        gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
        gaussian_process.fit(Xtrain.reshape(-1, 1), Ytrain)

        mean_prediction, std_prediction = gaussian_process.predict(dx.reshape(-1, 1), return_std=True)

        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    {
                        "tractography distance [mm]" : list(Xtrain * sdx + mux) + list(Xtest * sdx + mux) + list(xtarget * sdx + mux),
                        "log(1+N)" : list(Ytrain) + list(Ytest) + list(Ytarget),
                        "type" : ["train"] * ntrain + ["test"] * ntest + ["target"] * Ntarget,
                        "SOURCE" : [a] * (N + Ntarget)
                    }
                )
            ], ignore_index=True
        )

        pred = pd.concat(
            [
                pred,
                pd.DataFrame(
                    {
                        "tractography distance [mm]" : dx * sdx + mux,
                        "log(1+N)" : mean_prediction,
                        "sd" : std_prediction,
                        "SOURCE" : [a] * dx.shape[0]
                    }
                )
            ], ignore_index=True
        )

    
    g = sns.FacetGrid(
        data=data,
        col="SOURCE",
        col_wrap=3,
        hue="type",
        aspect=1.5,
        height=4
    )
    g.map_dataframe(
        sns.scatterplot,
        x="tractography distance [mm]",
        y="log(1+N)",
        s=30
    )
    for i, axes in enumerate(g.axes.flat):
        sns.lineplot(
            data=pred.loc[pred.SOURCE == A[i]],
            x="tractography distance [mm]",
            y="log(1+N)",
            color="r",
            ax=axes
        )
        axes.fill_between(
            pred["tractography distance [mm]"].loc[pred.SOURCE == A[i]],
            pred["log(1+N)"].loc[pred.SOURCE == A[i]] + pred["sd"].loc[pred.SOURCE == A[i]] / 2,
            pred["log(1+N)"].loc[pred.SOURCE == A[i]] - pred["sd"].loc[pred.SOURCE == A[i]] / 2,
            color = "r", alpha=0.2
        )

    cortex_letter_path = "../plots/MAC/57d106/LN/original/tracto16/57/SINGLE_106_57_l10/"
    plt.savefig(
        f"{cortex_letter_path}/gaussian_process_regression.png", dpi=300
    )





    
    