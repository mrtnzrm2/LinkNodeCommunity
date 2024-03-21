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

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

from pathlib import Path
import itertools
sns.set_theme()
# Personal libraries ----
from networks.MAC.mac57 import MAC57
from various.network_tools import *
import ctools as ct

def plot_error_histogram():
    results = pd.read_pickle("../pickle/PRED/GP/MATERN/NU_1_5/DSIM/gaussian_process_regression_validation.pk")
    results = results.loc[(results.Y > 0)]
    result_marginal = results.groupby(["SOURCE", "set", "iteration"])["error"].mean().reset_index()

    g = sns.FacetGrid(
        data=result_marginal,
        col="SOURCE",
        col_wrap=3,
        hue="set"
    )

    g.map_dataframe(
        sns.histplot,
        x="error",
        common_bins=False,
        kde=True,
        alpha=0.3
    )
    g.add_legend()
    plt.show()

def plot_reg_overview():
    from scipy.stats import pearsonr
    DATA = pd.read_pickle("../pickle/PRED/GP/RBF/DSIM/10.pk")
    DATA = DATA.groupby(["SOURCE", "X", "Y", "set"])["YPRED"].mean().reset_index()
    areas = np.unique(DATA["SOURCE"])

    g = sns.FacetGrid(
        data=DATA,
        col="SOURCE",
        col_wrap=3,
        hue="set",
        col_order=areas
    )

    g.map_dataframe(
        sns.scatterplot,
        x="YPRED",
        y="Y",
        alpha=0.7
    )

    pos = {"train" : 0.92, "test" : 0.82}
    for i, axes in enumerate(g.axes.flat):
        for k in ["train", "test"]:
            x = DATA.YPRED.loc[(DATA.SOURCE == areas[i]) & (DATA.set == k)]
            y = DATA.Y.loc[(DATA.SOURCE == areas[i]) & (DATA.set == k)]
            cor = pearsonr(x, y)
            pv = cor.pvalue
            if pv <= 0.05 and pv > 0.001:
                a = "*"
            elif pv <= 0.001 and pv > 0.0001:
                a = '**'
            elif pv <= 0.0001:
                a = "***"
            else:
                a = "ns"
            axes.text(0.1, pos[k], f"{k}: {cor.statistic:.2f} {a}", transform=axes.transAxes)

    g.add_legend()

    plt.show()


    da1 = DATA[["SOURCE", "X", "YPRED"]].loc[DATA.set == "test"]
    da1.columns = ["SOURCE", "X", "Y"]
    da1["set"] = "pred"
    da2 = DATA[["SOURCE", "X", "Y"]].loc[DATA.set == "test"]
    da2["set"] = "ref"

    da = pd.concat([da1, da2], ignore_index=True)

    g = sns.FacetGrid(
        data=da,
        col="SOURCE",
        hue="set",
        col_wrap=3,
        col_order=areas
    )

    g.map_dataframe(
        sns.scatterplot,
        x="X",
        y="Y",
        alpha=0.7
    )

    plt.show()

def plot_error_violin():

    # results = pd.read_pickle("../pickle/PRED/GP/MATERN/NU_0_5/DSIM/gaussian_process_regression_validation.pk")
    results = pd.read_pickle("../pickle/PRED/GP/RBF/DSIM/gaussian_process_regression_validation.pk")
    results = results.loc[results.Y > 0]
    result_marginal = results.groupby(["SOURCE", "set", "iteration"])["error"].mean().reset_index()

    result_marginal = result_marginal.loc[result_marginal.set == "test"]

    order = result_marginal.groupby(["SOURCE", "set"])["error"].var().reset_index().sort_values("error", ascending=False)

    sns.catplot(
        data=result_marginal,
        x="SOURCE",
        y="error",
        hue="set",
        kind="violin",
        order=order["SOURCE"]
    )

    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    plt.show()

def plot_residuals():
    results = pd.read_pickle("../pickle/gaussian_process_regression_validation.pk")
    results = results.loc[results.Y > 0]
    result_marginal = results.groupby(["SOURCE", "X", "set", "iteration"])["error"].mean().reset_index()

    g = sns.FacetGrid(
        data=result_marginal,
        col="SOURCE",
        col_wrap=3,
        hue="set"
    )

    g.map_dataframe(
        sns.scatterplot,
        x="X",
        y="error"
    )
    g.add_legend()
    plt.show()

def plot_residuals_all():
    results = pd.read_pickle("../pickle/PRED/GP/RBF/D/gaussian_process_regression_validation.pk")
    results = results.loc[results.Y > 0]
    result_marginal = results.groupby(["SOURCE", "X", "set", "iteration"])["error"].mean().reset_index()

    result_marginal = result_marginal.loc[result_marginal.set == "test"]

    sns.scatterplot(
        data=result_marginal,
        x="X",
        y="error",
        hue="set",
        alpha=0.4,
        s=10
    )
    plt.show()

def DSIM(NET, perm_rows, perm_cols, in_rows, in_cols, l_in_train):
    X = np.zeros((1 + 1, NET.rows, NET.rows))
    nfeatures = X.shape[0]

    X[0, :, :] = NET.D.copy()[perm_rows, :][:, perm_rows][in_rows, :][:, in_rows]
    C = NET.C.copy()[perm_rows, :][:, perm_cols][in_rows, :][:, in_cols]

    for ii in np.arange(NET.rows):
        for jj in np.arange(ii + 1, NET.rows):
            X[1, ii, jj] = ct.Hellinger(C[ii, :l_in_train], C[jj, :l_in_train], ii, jj)
            X[1, ii, jj] = 1 - X[1, ii, jj]
            X[1, jj, ii] = X[1, ii, jj]

    for i in [0, 1]: np.fill_diagonal(X[i, :, :], np.nan)

    mux = np.nanmean(X, axis=(1, 2))
    sdx = np.nanstd(X, axis=(1, 2))

    for i in [0, 1]:
        X[i] = (X[i] - mux[i]) / sdx[i]
        np.fill_diagonal(X[i], 0.)
    
    return X, C, nfeatures, sdx[0], mux[0]

def SIMfeature(NET, perm_rows, perm_cols, in_rows, in_cols, l_in_train):
    X = np.zeros((1, NET.rows, NET.rows))
    nfeatures = X.shape[0]

    C = NET.C.copy()[perm_rows, :][:, perm_cols][in_rows, :][:, in_cols]
    for ii in np.arange(NET.rows):
        for jj in np.arange(ii + 1, NET.rows):
            X[0, ii, jj] = ct.Hellinger(C[ii, :l_in_train], C[jj, :l_in_train], ii, jj)
            X[0, ii, jj] = 1 - X[0, ii, jj]
            X[0, jj, ii] = X[0, ii, jj]

    for i in [0]: np.fill_diagonal(X[i, :, :], np.nan)

    mux = np.nanmean(X, axis=(1, 2))
    sdx = np.nanstd(X, axis=(1, 2))

    for i in [0]:
        X[i] = (X[i] - mux[i]) / sdx[i]
        np.fill_diagonal(X[i], 0.)
    
    return X, C, nfeatures, sdx[0], mux[0]

def Dfeature(NET, perm_rows, perm_cols, in_rows, in_cols, l_in_train):
    X = np.zeros((1, NET.rows, NET.rows))
    nfeatures = X.shape[0]
    C = NET.C.copy()[perm_rows, :][:, perm_cols][in_rows, :][:, in_cols]

    X[0, :, :] = NET.D.copy()[perm_rows, :][:, perm_rows][in_rows, :][:, in_rows]
    for i in [0]: np.fill_diagonal(X[i, :, :], np.nan)

    mux = np.nanmean(X, axis=(1, 2))
    sdx = np.nanstd(X, axis=(1, 2))

    for i in [0]:
        X[i] = (X[i] - mux[i]) / sdx[i]
        np.fill_diagonal(X[i], 0.)
    
    return X, C, nfeatures, sdx[0], mux[0]

def choose_feature(feature):
    if feature == "DSIM":
        return DSIM
    elif feature == "D":
        return Dfeature
    elif feature == "SIM":
        return SIMfeature
    else: raise ValueError("That feature does not exist")

@ignore_warnings(category=ConvergenceWarning)
def iterations(listargs):

    NET = listargs[0]
    areas = listargs[1]
    k = listargs[2]
    lw = int(listargs[3])
    up = int(listargs[4])
    ker, feature, nu = listargs[5]


    K = np.floor(NET.nodes / k).astype(int)

    data = { a : pd.DataFrame() for a in areas}

    for it in np.arange(lw, up):

        indices = np.arange(NET.nodes)
        perm_cols = np.random.permutation(NET.nodes)
        perm_rows = list(np.random.permutation(np.arange(NET.nodes, NET.rows)))
        perm_rows = list(perm_cols) + perm_rows

        TARGET_TR = {a: [] for a in areas}
        TARGET_TS = {a: [] for a in areas}

        Y_TR = {a: [] for a in areas}
        YPRED_TR = {a: [] for a in areas}

        Y_TS = {a: [] for a in areas}
        YPRED_TS = {a: [] for a in areas}

        X_TR = {a: [] for a in areas}
        X_TS = {a: [] for a in areas}

        for j in range(k):

            if j < k - 1 : in_test = list(np.arange(j * (K), (j + 1 ) * (K), dtype=int))
            else: in_test = list(np.arange(j * (K), NET.nodes, dtype=int))
            in_train = [ii for ii in indices if ii not in in_test]

            l_in_train = len(in_train)

            in_cols = in_train + in_test
            in_rows = in_train + in_test + list(np.arange(NET.nodes, NET.rows))

            labels = list(NET.labels[perm_rows][in_rows])

            feat = choose_feature(feature)
            X, C, nfeatures, sdx, mux = feat(NET, perm_rows, perm_cols, in_rows, in_cols, l_in_train)

            for a in areas:

                iA = match([a], labels)

                Xtrain= X[:, iA, :l_in_train].reshape(nfeatures, -1).T
                Ytrain = np.log(1 + C[iA, :l_in_train]).ravel()

                yno = Ytrain == 0
                Xtrain = Xtrain[~yno, :]
                Ytrain = Ytrain[~yno]

                Xtest = X[:, iA, l_in_train:NET.nodes].reshape(nfeatures, -1).T
                Ytest = np.log(1 + C[iA, l_in_train:NET.nodes]).ravel()

                if ker == "RBF":
                  kernel = RBF(length_scale=2,length_scale_bounds=(1, 10)) + WhiteKernel(noise_level=5)
                elif ker == "Matern":
                  kernel = Matern(length_scale=2,length_scale_bounds=(1, 10), nu=float(nu)) + WhiteKernel(noise_level=5)

                model = GaussianProcessRegressor(
                    kernel=kernel, n_restarts_optimizer=100, normalize_y=True
                )
                model.fit(Xtrain, Ytrain)

                Ytrain_pred = model.predict(Xtrain)
                Ytest_pred = model.predict(Xtest)

                Y_TR[a] += list(Ytrain)
                Y_TS[a] += list(Ytest)

                X_TR[a] += list(Xtrain[:, 0] * sdx + mux)
                X_TS[a] += list(Xtest[:, 0] * sdx + mux)

                YPRED_TR[a] += list(Ytrain_pred)
                YPRED_TS[a] += list(Ytest_pred)

                TARGET_TR[a] += [s for i, s in enumerate(labels[:l_in_train]) if not yno[i]]
                TARGET_TS[a] += labels[l_in_train:NET.nodes]
        
        for a in areas:
            mse_train = mean_squared_error(Y_TR[a], YPRED_TR[a])
            mse_test = mean_squared_error(Y_TS[a], YPRED_TS[a])
            print(it, a, mse_train, mse_test)
            data[a] = pd.concat(
                [
                    data[a],
                    pd.DataFrame(
                        {
                            "Y" : Y_TR[a] + Y_TS[a],
                            "YPRED" : YPRED_TR[a] + YPRED_TS[a],
                            "X" : X_TR[a] + X_TS[a],
                            "error" : list(np.array(Y_TR[a]) - np.array(YPRED_TR[a])) + list(np.array(Y_TS[a]) - np.array(YPRED_TS[a])),
                            "set" : ["train"] * len(Y_TR[a]) + ["test"] * len(Y_TS[a]),
                            "SOURCE" : [a] * (len(Y_TR[a])+ len(Y_TS[a])),
                            "TARGET" : TARGET_TR[a] + TARGET_TS[a],
                            "k" : [j] * (len(Y_TR[a]) + len(Y_TS[a])) ,
                            "iteration" : [it] * (len(Y_TR[a]) + len(Y_TS[a]))
                        }
                    )
                ], ignore_index=True
            ) 
    
    DATA = pd.DataFrame()
    for _, dat in data.items(): DATA = pd.concat([DATA, dat], ignore_index=True)

    DATA = DATA.loc[DATA.Y > 0]

    return DATA

def gaussian_process_kfold(NET, areas,  t, k=10, MAXIT=100, ker="RBF", feature="D", nu=1.5):
    import multiprocessing as mp

    De = np.floor(MAXIT / t).astype(int)

    lw = 0
    bounds = []

    for i in np.arange(t - 1):
        bounds.append((lw, lw + De))
        lw += De
    bounds.append((lw, MAXIT))

    paralist = []
    for lw, up in bounds: paralist.append([NET, areas, k, lw, up, (ker, feature, nu)])

    with mp.Pool(t) as p:
      process = p.map(iterations, paralist)
    
    DATA = pd.DataFrame()

    for p in process: DATA = pd.concat([DATA, p], ignore_index=True)

    if ker == "RBF":
      folderpath = f"../pickle/PRED/GP/{ker}/{feature}"
    elif ker == "Matern":
      folderpath = f"../pickle/PRED/GP/{ker}/N_{float(nu)}/{feature}"
    Path(folderpath).mkdir(parents=True, exist_ok=True)
    DATA.to_pickle(f"{folderpath}/{k}.pk")

def model_params():
    features = ["DSIM", "SIM", "D"]
    # K = np.arange(2, 12, 2, dtype=int)
    K = [4]
    KERNELS = ["RBF", "Matern"]
    NU = [0.5, 1.5, 2.5]
    NU = [str(s) for s in NU]
    NU += [""]

    list_of_lists = itertools.product(*[features, K, KERNELS, NU])
    list_of_lists = np.array(list(list_of_lists))
    array_model = pd.DataFrame(
        {
            "worker" : ["REG"] * list_of_lists.shape[0],
            "feature" : list_of_lists[:, 0].astype(str),
            "k" : list_of_lists[:, 1].astype(int),
            "kernel" : list_of_lists[:, 2].astype(str),
            "nu" : list_of_lists[:, 3]
        }
    )
    
    array_model = array_model.loc[
        ((array_model.kernel == "RBF") & (array_model.nu == "")) |
        ((array_model.kernel == "Matern") & (array_model.nu != ""))
    ]

    array_model.index = np.arange(array_model.shape[0])

    return array_model

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
index  = "Hellinger"
discovery = "discovery_7"
bias = 0.
alpha = 0.
version = "57d106"
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
    NET = MAC57(
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

    # plot_reg_overview()
    # areas = ["v4c", "v1c", "8l", "tepd", "mip", "stpc"]

    # t = int(sys.argv[1])
    for t in np.arange(1, 13):
        parms = model_params()
        # print(parms.loc[parms.kernel == "RBF"])
        print(parms.shape[0])
        parms = parms.iloc[t-1]
        print(parms)
        
        gaussian_process_kfold(
            NET, NET.labels, 4, k=int(parms["k"]), MAXIT=100, feature=parms["feature"],
            ker = parms["kernel"], nu = parms["nu"]
        )

