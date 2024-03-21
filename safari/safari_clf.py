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
from pathlib import Path
import itertools
sns.set_theme()
# Personal libraries ----
from networks.MAC.mac57 import MAC57
from various.network_tools import *
import ctools as ct

def plot_ROC():
    results = pd.read_pickle("../pickle/gaussian_process_classification_validation_2.pk")
    result_marginal = results.groupby(["SOURCE", "set", "iteration", "FPR"])[["TPR"]].mean().reset_index()

    g = sns.FacetGrid(
        data=result_marginal,
        col="SOURCE",
        col_wrap=3,
        hue="set"
    )

    g.map_dataframe(
        sns.lineplot,
        x="FPR",
        y="TPR",
        alpha=0.5
    )
    x = np.linspace(0, 1, 25)
    for axes in g.axes.flat:
        sns.lineplot(
            x=x,
            y=x,
            linestyle="--",
            color="black",
            ax=axes
        )
    g.add_legend()
    plt.show()

def plot_ACC(variable="FPR"):
    results = pd.read_pickle("../pickle/PRED/XGBOOST/DSIM/10.pk")
    result_marginal = results.groupby(["SOURCE", "set", "iteration"])[[variable]].mean().reset_index()

    result_marginal = result_marginal.loc[result_marginal.set == "test"]

    order = result_marginal.groupby("SOURCE")[variable].median().reset_index().sort_values(variable)
    

    _, ax = plt.subplots(1, 1)
    sns.violinplot(
        data=result_marginal,
        x="SOURCE",
        y=variable,
        hue="set",
        ax=ax,
        order=order.SOURCE
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

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
def xgboost_bin_kfold(NET, areas, k=10, MAXIT=100, feature="D"):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, confusion_matrix

    # param = {"objective" : "binary:logistic"}
    # # param["max_depth"] = 7
    # param['tree_method'] = 'hist'
    # # param["min_child_weight"] = 2
    # param["gamma"] = 0.1
    # param["lambda"] = 3
    # param["eta"] = 1
    
    indices = np.arange(NET.nodes)
    K = np.floor(NET.nodes / k).astype(int)

    data = {a : pd.DataFrame() for a in areas}

    it = 0
    while it < MAXIT:
        perm_cols = np.random.permutation(NET.nodes)
        perm_rows = list(np.random.permutation(np.arange(NET.nodes, NET.rows)))
        perm_rows = list(perm_cols) + perm_rows

        TARGET_TR = {a: [] for a in areas}
        TARGET_TS = {a: [] for a in areas}

        Y_TR = {a: [] for a in areas}
        YPRED_TR = {a: [] for a in areas}
        YPRED_PROBA_TR = {a: [] for a in areas}

        Y_TS = {a: [] for a in areas}
        YPRED_TS = {a: [] for a in areas}
        YPRED_PROBA_TS= {a: [] for a in areas}

        X_TR = {a: [] for a in areas}
        X_TS = {a: [] for a in areas}

        flag = False

        for j in range(k):

            if j < k -1 : in_test = list(np.arange(j * (K), (j + 1 ) * (K), dtype=int))
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
                Ytrain = np.array([C[iA, :l_in_train] > 0]).ravel().astype(int)

                Xtest = X[:, iA, l_in_train:NET.nodes].reshape(nfeatures, -1).T
                Ytest = np.array(C[iA, l_in_train:NET.nodes] > 0).ravel().astype(int)

                if iA < l_in_train:
                    keep = [i for i in np.arange(l_in_train) if i != iA]
                    Xtrain  = Xtrain[keep, :]
                    Ytrain = Ytrain[keep]

                if np.unique(Ytrain).shape[0] < 2:
                    flag=True
                    break

                model = xgb.XGBClassifier()
                model.fit(Xtrain, Ytrain)

                Ytrain_pred = model.predict(Xtrain)
                Ytest_pred = model.predict(Xtest)

                Ytrain_proba_pred = model.predict_proba(Xtrain)
                Ytest_proba_pred = model.predict_proba(Xtest)

                Y_TR[a] += list(Ytrain)
                YPRED_TR[a] += list(Ytrain_pred)
                YPRED_PROBA_TR[a] += list(Ytrain_proba_pred[:, 1])

                Y_TS[a] += list(Ytest)
                YPRED_TS[a] += list(Ytest_pred)
                YPRED_PROBA_TS[a] += list(Ytest_proba_pred[:, 1])

                X_TR[a] += list(Xtrain[:, 0] * sdx + mux)
                X_TS[a] += list(Xtest[:, 0] * sdx + mux)

                if iA < NET.nodes:
                    TARGET_TR[a] += [s for ki, s in enumerate(labels[:l_in_train]) if ki != iA]
                else:
                    TARGET_TR[a] += labels[:l_in_train]
                TARGET_TS[a] += labels[l_in_train:NET.nodes]

            if flag: break

        if not flag:
            
            for a in areas:
                acc_train = accuracy_score(Y_TR[a], YPRED_TR[a])
                acc_test = accuracy_score(Y_TS[a], YPRED_TS[a])

                tn, fp_tr, fn, tp_tr = confusion_matrix(Y_TR[a], YPRED_TR[a]).ravel()
                tpr_tr = tp_tr / (tp_tr + fn)
                fpr_tr = fp_tr / (fp_tr + tn)

                tn, fp_ts, fn, tp_ts = confusion_matrix(Y_TS[a], YPRED_TS[a]).ravel()
                tpr_ts = tp_ts / (tp_ts + fn)
                fpr_ts = fp_ts / (fp_ts + tn)

                print(it, a, acc_train, tpr_tr, fpr_tr)
                print(it, a, acc_test, tpr_ts, fpr_ts)

                data[a] = pd.concat(
                    [
                        data[a],
                        pd.DataFrame(
                            {
                                "set" : ["train"] * len(TARGET_TR[a]) + ["test"] * len(TARGET_TS[a]),
                                "Y" : Y_TR[a] + Y_TS[a],
                                "YPRED" : YPRED_TR[a] + YPRED_TS[a],
                                "YPRED_PROBA" : YPRED_PROBA_TR[a] + YPRED_PROBA_TS[a],
                                "X" : X_TR[a] + X_TS[a],
                                "SOURCE" : [a] * (len(TARGET_TR[a]) + len(TARGET_TS[a])),
                                "TARGET" : TARGET_TR[a] + TARGET_TS[a],
                                "iteration" : [it] * (len(TARGET_TR[a]) + len(TARGET_TS[a])),
                                "acc" : [acc_train] * len(TARGET_TR[a]) + [acc_test] * len(TARGET_TS[a]),
                                "TPR" : [tpr_tr] * len(Y_TR[a]) + [tpr_ts] * len(Y_TS[a]),
                                "FPR" : [fpr_tr] * len(Y_TR[a]) + [fpr_ts] * len(Y_TS[a])
                            }
                        )
                    ], ignore_index=True
                )
            it += 1
            
    DATA = pd.DataFrame()
    for _, dat in data.items(): DATA = pd.concat([DATA, dat], ignore_index=True)

    folderpath = f"../pickle/PRED/XGBOOST/{feature}"
    Path(folderpath).mkdir(parents=True, exist_ok=True)
    DATA.to_pickle(f"{folderpath}/{k}.pk")

def model_params():
    features = ["DSIM", "SIM", "D"]
    K = np.arange(2, 12, 2, dtype=int)

    list_of_lists = itertools.product(*[features, K])
    list_of_lists = np.array(list(list_of_lists))
    array_model = pd.DataFrame(
        {
            "worker" : ["CLF"] * list_of_lists.shape[0],
            "feature" : list_of_lists[:, 0].astype(str),
            "k" : list_of_lists[:, 1].astype(int)
        }
    )

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

    # plot_ACC("acc")

    # areas = ["v4c", "v1c", "8l", "tepd", "mip", "stpc"]

    # t = int(sys.argv[1])
    for t in np.arange(1, 16):
        parms = model_params()
        print(parms.shape[0])
        parms = parms.iloc[t-1]
        print(parms)

        xgboost_bin_kfold(
            NET, NET.labels, k=int(parms["k"]), MAXIT=100, feature=parms["feature"]
        )

