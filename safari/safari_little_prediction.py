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
sns.set_theme()
# Personal libraries ----
from networks.MAC.mac57 import MAC57
from various.network_tools import *
import ctools as ct

@ignore_warnings(category=ConvergenceWarning)
def predict_source_areas_kfold(listargs):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

    NET = listargs[0]
    a= listargs[1]
    k = listargs[2]
    MAXIT = listargs[3]

    print(listargs[4], a)

    pred_train = pd.DataFrame()
    pred_test = pd.DataFrame()

    K = np.floor(NET.nodes / k).astype(int)
    indices = np.arange(NET.nodes)

    for it in np.arange(MAXIT):

        perm_cols = np.random.permutation(NET.nodes)
        perm_rows = list(np.random.permutation(np.arange(NET.nodes, NET.rows)))
        perm_rows = list(perm_cols) + perm_rows

        for j in np.arange(k):
            if j < k -1 : in_test = list(np.arange(j * (K), (j + 1 ) * (K), dtype=int))
            else: in_test = list(np.arange(j * (K), NET.nodes, dtype=int))
            in_train = [ii for ii in indices if ii not in in_test]

            l_in_train = len(in_train)

            in_cols = in_train + in_test
            in_rows = in_train + in_test + list(np.arange(NET.nodes, NET.rows))

            labels = list(NET.labels[perm_rows][in_rows])

            X = np.zeros((2, NET.rows, NET.rows))
            X[0, :, :] = NET.D.copy()[perm_rows, :][:, perm_rows][in_rows, :][:, in_rows]
            C = NET.C.copy()[perm_rows, :][:, perm_cols][in_rows, :][:, in_cols]
            for ii in np.arange(NET.rows):
                for jj in np.arange(ii + 1, NET.rows):
                    X[1, ii, jj] = ct.D1_2_4(C[ii, :l_in_train], C[jj, :l_in_train], ii, jj)
                    np.seterr(divide='ignore', invalid='ignore')
                    X[1, ii, jj] = 1 / X[1, ii, jj] + 1
                    X[1, jj, ii] = X[1, ii, jj]
            X[1, :, :][X[1, :, :] == np.Inf] = np.max(X[1, :, :][X[1, :, :] < np.Inf]) * 1.05

            for i in [0, 1]: np.fill_diagonal(X[i, :, :], np.nan)
            mux = np.nanmean(X, axis=(1, 2))
            sdx = np.nanstd(X, axis=(1, 2))
            for i in [0, 1]:
                X[i] = (X[i] - mux[i]) / sdx[i]
                np.fill_diagonal(X[i], 0.)

            iA = match([a], labels)

            Xtrain= X[:, iA, :l_in_train].reshape(X.shape[0], -1)
            Ytrain = np.log(1 + C[iA, :l_in_train]).ravel()

            yno = Ytrain == 0
            Xtrain = Xtrain[:, ~yno]
            Ytrain = Ytrain[~yno]

            Xtest = X[:, iA, l_in_train:NET.nodes].reshape(X.shape[0], -1)
            Ytest = np.log(1 + C[iA, l_in_train:NET.nodes]).ravel()

            kernel = Matern(
                length_scale=1,
                length_scale_bounds=(0.7, 1.5),
                nu=0.5
            ) + WhiteKernel(
                noise_level=1, noise_level_bounds=(0.5, 4)
            )  

            gaussian_process = GaussianProcessRegressor(kernel=kernel,  n_restarts_optimizer=100, normalize_y=True)
            gaussian_process.fit(Xtrain.reshape(-1, Xtrain.shape[0]), Ytrain)

            mean_train = gaussian_process.predict(Xtrain.reshape(-1, Xtrain.shape[0]), return_std=False)
            mean_test = gaussian_process.predict(Xtest.reshape(-1, Xtest.shape[0]), return_std=False)

            pred_train = pd.concat(
                [
                    pred_train,
                    pd.DataFrame(
                        {
                            "error" : Ytrain - mean_train,
                            "k" : [j] * Ytrain.shape[0],
                            "SOURCE" : [a] * Ytrain.shape[0],
                            "TARGET" : [s for i, s in enumerate(labels[:l_in_train]) if not yno[i]],
                            "Y" : Ytrain,
                            "X" : Xtrain[0, :] * sdx[0] + mux[0],
                            "iteration" : [it] * Ytrain.shape[0],
                            "set" : ["train"] * Ytrain.shape[0]
                        }
                    )
                ], ignore_index=True
            )

            pred_test = pd.concat(
                [
                    pred_test,
                    pd.DataFrame(
                        {
                            "error" : Ytest - mean_test,
                            "k" : [j] * (Ytest.shape[0]),
                            "SOURCE" : [a] * (Ytest.shape[0]),
                            "TARGET" : labels[l_in_train:NET.nodes],
                            "Y" : Ytest,
                            "X" : Xtest[0, :] * sdx[0] + mux[0],
                            "iteration" : [it] * Ytest.shape[0],
                            "set" : ["test"] * Ytest.shape[0],
                        }
                    )
                ], ignore_index=True
            )

    pred = pd.concat([pred_train, pred_test], ignore_index=True)
    return pred

def parallel_predict_areas(NET, areas, k=10, MAXIT=100):
    import multiprocessing as mp

    paralist = []
    for i, a in enumerate(areas): paralist.append([NET, a, k, MAXIT, i])

    with mp.Pool(6) as p:
      process = p.map(predict_source_areas_kfold, paralist)
    
    pred = pd.DataFrame()

    for p in process:
        pred = pd.concat([pred, p], ignore_index=True)
    
    pred.to_pickle("../pickle/PRED/GP/MATERN/NU_0_5/DSIM/gaussian_process_regression_validation.pk")

def predict_source_areas_bin_kfold(listargs):
    from sklearn.metrics import roc_curve, accuracy_score
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

    NET = listargs[0]
    a= listargs[1]
    k = listargs[2]
    MAXIT = listargs[3]

    pred_train = pd.DataFrame()
    pred_test = pd.DataFrame()

    K = np.floor(NET.nodes / k).astype(int)
    indices = np.arange(NET.nodes)

    for it in np.arange(MAXIT):

        perm_cols = np.random.permutation(NET.nodes)
        perm_rows = list(np.random.permutation(np.arange(NET.nodes, NET.rows)))
        perm_rows = list(perm_cols) + perm_rows

        for j in np.arange(k):
            if j < k -1 : in_test = list(np.arange(j * (K), (j + 1 ) * (K), dtype=int))
            else: in_test = list(np.arange(j * (K), NET.nodes, dtype=int))
            in_train = [ii for ii in indices if ii not in in_test]

            l_in_train = len(in_train)

            in_cols = in_train + in_test
            in_rows = in_train + in_test + list(np.arange(NET.nodes, NET.rows))

            X = np.zeros((2, NET.rows, NET.rows))
            X[0, :, :] = NET.D.copy()[perm_rows, :][:, perm_rows][in_rows, :][:, in_rows]
            C = NET.C.copy()[perm_rows, :][:, perm_cols][in_rows, :][:, in_cols]
            for ii in np.arange(NET.rows):
                for jj in np.arange(ii + 1, NET.rows):
                    X[1, ii, jj] = ct.D1_2_4(C[ii, :l_in_train], C[jj, :l_in_train], ii, jj)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        X[1, ii, jj] = 1 / X[1, ii, jj] + 1
                    X[1, jj, ii] = X[1, ii, jj]
            X[1, :, :][X[1, :, :] == np.Inf] = np.max(X[1, :, :][X[1, :, :] < np.Inf]) * 1.95
            
            for i in np.arange(2): np.fill_diagonal(X[i, :, :], np.nan)
            mux = np.nanmean(X, axis=(1, 2))
            sdx = np.nanstd(X, axis=(1, 2))
            for i in np.arange(2):
                X[i] = (X[i] - mux[i]) / sdx[i]
                np.fill_diagonal(X[i], 0.)

            iA = match([a], NET.labels[perm_rows][in_rows])

            Xtrain= X[:, iA, :l_in_train].reshape(X.shape[0], -1)
            Ytrain = np.array(C[iA, :l_in_train] > 0).ravel().astype(float)

            Xtest = X[:, iA, l_in_train:NET.nodes].reshape(X.shape[0], -1)
            Ytest = np.array(C[iA, l_in_train:NET.nodes] > 0).ravel().astype(float)

            if np.unique(Ytest).shape[0] < 2 or np.unique(Ytrain).shape[0] < 2: continue

            kernel = Matern(length_scale=2, length_scale_bounds=(1, 3), nu=1.5)

            gaussian_process = GaussianProcessClassifier(kernel=kernel,  n_restarts_optimizer=100)
            gaussian_process.fit(Xtrain.reshape(-1, Xtrain.shape[0]), Ytrain)

            p_train = gaussian_process.predict_proba(Xtrain.reshape(-1, Xtrain.shape[0]))[:, 1]
            p_test = gaussian_process.predict_proba(Xtest.reshape(-1, Xtest.shape[0]))[:, 1]

            fpr_tr, tpr_tr, _ = roc_curve(Ytrain, p_train)
            fpr_ts, tpr_ts, _ = roc_curve(Ytest, p_test)

            acc_tr = accuracy_score(Ytrain, gaussian_process.predict(Xtrain.reshape(-1, Xtrain.shape[0])))
            acc_ts = accuracy_score(Ytest, gaussian_process.predict(Xtest.reshape(-1, Xtest.shape[0])))

            ltr = len(tpr_tr)
            lts = len(tpr_ts)

            pred_train = pd.concat(
                [
                    pred_train,
                    pd.DataFrame(
                        {
                            "FPR" : fpr_tr,
                            "TPR" : tpr_tr,
                            "k" : [j] * ltr,
                            "SOURCE" : [a] * ltr,
                            "iteration" : [it] * ltr,
                            "set" : ["train"] * ltr,
                            "acc" : [acc_tr] * ltr
                        }
                    )
                ], ignore_index=True
            )

            pred_test = pd.concat(
                [
                    pred_test,
                    pd.DataFrame(
                        {
                            "FPR" : fpr_ts,
                            "TPR" : tpr_ts,
                            "k" : [j] * lts,
                            "SOURCE" : [a] * lts,
                            "iteration" : [it] * lts,
                            "set" : ["test"] * lts,
                            "acc" : [acc_ts] * lts
                        }
                    )
                ], ignore_index=True
            )

    pred = pd.concat([pred_train, pred_test], ignore_index=True)
    return pred

def parallel_predict_areas_bin(NET, areas, k=10, MAXIT=100):
    import multiprocessing as mp

    paralist = []
    for i, a in enumerate(areas): paralist.append([NET, a, k, MAXIT, i])

    with mp.Pool(6) as p:
      process = p.map(predict_xgboost_bin_kfold, paralist)
    
    pred = pd.DataFrame()
    for p in process:
        pred = pd.concat([pred, p], ignore_index=True)
    
    pred.to_pickle("../pickle/PRED/XGBOOST/DSIM/xgb_classification_validation.pk")

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
    DATA = pd.read_pickle("../pickle/PRED/GP/RBF/DSIM/gaussian_process_regression_validation.pk")
    DATA = DATA.groupby(["SOURCE", "X", "Y", "set"])["YPRED"].mean().reset_index()

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
            axes.text(0.1, pos[k], f"{k}:\t{cor.statistic:.2f} {a}", transform=axes.transAxes)

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
    results = pd.read_pickle("../pickle/PRED/XGBOOST/DSIM/xgb_classification_validation.pk")
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

def plot_gaussian_process_mcmc(NET, areas, k=10, l=100):
    from sklearn.metrics import mean_squared_error
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
    
    indices = np.arange(NET.nodes)
    perm_cols = np.random.permutation(NET.nodes)
    perm_rows = list(np.random.permutation(np.arange(NET.nodes, NET.rows)))
    perm_rows = list(perm_cols) + perm_rows

    K = np.floor(NET.nodes / k, dtype=float)

    data = { k : pd.DataFrame() for k in areas}
    hyper = {k : np.zeros(6 + 3) for k in areas}

    min_fitness = {k : 100 for k in areas}
    fitness_old = {k : 10 for k in areas}
    fitness = {k : 0 for k in areas}

    hyperparameters = {k: np.zeros(6) for k in areas}
    for a in areas:
        hyperparameters[a][0] = 0.5
        hyperparameters[a][1] = 0.1
        hyperparameters[a][2] = 1
        hyperparameters[a][3] = 0.7
        hyperparameters[a][4] = 0.5
        hyperparameters[a][5] = 2

    for j in range(1):

        if j < k -1 : in_test = list(np.arange(j * (K), (j + 1 ) * (K), dtype=int))
        else: in_test = list(np.arange(j * (K), NET.nodes, dtype=int))
        in_train = [ii for ii in indices if ii not in in_test]

        l_in_train = len(in_train)

        in_cols = in_train + in_test
        in_rows = in_train + in_test + list(np.arange(NET.nodes, NET.rows))

        X = np.zeros((2, NET.rows, NET.rows))
        X[0, :, :] = NET.D.copy()[perm_rows, :][:, perm_rows][in_rows, :][:, in_rows]
        C = NET.C.copy()[perm_rows, :][:, perm_cols][in_rows, :][:, in_cols]
        for ii in np.arange(NET.rows):
            for jj in np.arange(ii + 1, NET.rows):
                X[1, ii, jj] = ct.D1_2_4(C[ii, :l_in_train], C[jj, :l_in_train], ii, jj)
                X[1, ii, jj] = 1 / X[1, ii, jj] + 1
                X[1, jj, ii] = X[1, ii, jj]
        X[1, :, :][X[1, :, :] == np.Inf] = np.max(X[1, :, :][X[1, :, :] < np.Inf]) * 1.95

        for i in np.arange(2): np.fill_diagonal(X[i, :, :], np.nan)
        mux = np.nanmean(X, axis=(1, 2))
        sdx = np.nanstd(X, axis=(1, 2))
        for i in np.arange(2):
            X[i] = (X[i] - mux[i]) / sdx[i]
            np.fill_diagonal(X[i], 0.)
        
        for a in areas:

            print(a)
            iA = match([a], NET.labels[perm_rows][in_rows])

            Xtrain= X[:, iA, :l_in_train].reshape(X.shape[0], -1)
            Ytrain = np.log(1 + C[iA, :l_in_train]).ravel()

            yno = Ytrain == 0
            Xtrain = Xtrain[:, ~yno]
            Ytrain = Ytrain[~yno]

            Xtest = X[:, iA, l_in_train:NET.nodes].reshape(X.shape[0], -1)
            Ytest = np.log(1 + C[iA, l_in_train:NET.nodes]).ravel()

            T = 200
            DT = (T - 0.1)/l

            for _ in np.arange(l):   

                kernel = Matern(
                    length_scale=hyperparameters[a][0],
                    length_scale_bounds=(
                        hyperparameters[a][1],
                        hyperparameters[a][2]), 
                        nu=0.5
                    ) + WhiteKernel(
                    noise_level=hyperparameters[a][3],
                    noise_level_bounds=(
                        hyperparameters[a][4],
                        hyperparameters[a][5])
                    )  

                gaussian_process = GaussianProcessRegressor(
                    kernel=kernel, n_restarts_optimizer=100, normalize_y=True
                )

                gaussian_process.fit(Xtrain.reshape(-1, Xtrain.shape[0]), Ytrain)

                Ytrain_pred = gaussian_process.predict(Xtrain.reshape(-1, Xtrain.shape[0]))
                Ytest_pred = gaussian_process.predict(Xtest.reshape(-1, Xtest.shape[0]))

                mse_train = mean_squared_error(Ytrain, Ytrain_pred)
                mse_test = mean_squared_error(Ytest, Ytest_pred)

                fitness[a] = np.abs(mse_train - mse_test) + mse_train / 10

                print("score:\t", fitness[a])

                if min_fitness[a] > fitness[a]:
                    min_fitness[a] = fitness[a]
                    print("\n\nsaved", fitness[a], mse_train)
                    data[a] = pd.DataFrame(
                        {
                            "Y" : list(Ytrain) +  list(Ytest) ,
                            "Ypred" : list(Ytrain_pred) + list(Ytest_pred),
                            "X" : list(Xtrain[0, :] * sdx[0] + mux[0]) + list(Xtest[0, :] * sdx[0] + mux[0]),
                            "set" : ["train"] * Xtrain.shape[1] + ["test"] * Xtest.shape[1],
                            "SOURCE" : [a] * (Xtrain.shape[1] + Xtest.shape[1])
                        }
                    )
                    hyper[a][:6] = hyperparameters[a]
                    hyper[a][6] = mse_train
                    hyper[a][7] = mse_test
                    hyper[a][8] = fitness[a]

                if fitness[a] > fitness_old[a]:
                    fitness_old[a] = fitness[a]

                    hyperparameters[a][0] = np.random.randn() * 0.1 + hyperparameters[a][0]
                    if hyperparameters[a][0] < 0.07 or hyperparameters[a][0] > 1: hyperparameters[a][0] = np.random.uniform(0.07, 1)
                    hyperparameters[a][1] = np.random.randn() * 0.1 + hyperparameters[a][1]
                    if hyperparameters[a][1] < 0.05 or hyperparameters[a][1] > hyperparameters[a][0]: hyperparameters[a][1] = np.random.uniform(0.05, hyperparameters[a][0])
                    hyperparameters[a][2] = np.random.randn() * 0.1 + hyperparameters[a][2]
                    if hyperparameters[a][2] < 0.1 or hyperparameters[a][2] < hyperparameters[a][0]: hyperparameters[a][2] = np.random.uniform(hyperparameters[a][0], 3)

                    hyperparameters[a][3] = np.random.randn() * 0.1 + hyperparameters[a][3]
                    if hyperparameters[a][3] < 0.1 or hyperparameters[a][3] > 1: hyperparameters[a][3] = np.random.uniform(0.08, 0.5)
                    hyperparameters[a][4] = np.random.randn() * 0.1 + hyperparameters[a][4]
                    if hyperparameters[a][4] < 0.07 or hyperparameters[a][4] > hyperparameters[a][3]: hyperparameters[a][4] = np.random.uniform(0.07, hyperparameters[a][3])
                    hyperparameters[a][5] = np.random.randn() * 0.1 + hyperparameters[a][5]
                    if hyperparameters[a][5] < 0.1 or hyperparameters[a][4] < hyperparameters[a][4]: hyperparameters[a][5] = np.random.uniform(hyperparameters[a][3], 2)

                
                elif np.exp(-(fitness_old[a] - fitness[a]) / T) > np.random.rand():
                    fitness_old[a] = fitness[a]

                    hyperparameters[a][0] = np.random.randn() * 0.1 + hyperparameters[a][0]
                    if hyperparameters[a][0] < 0.07 or hyperparameters[a][0] > 1: hyperparameters[a][0] = np.random.uniform(0.07, 1)
                    hyperparameters[a][1] = np.random.randn() * 0.1 + hyperparameters[a][1]
                    if hyperparameters[a][1] < 0.05 or hyperparameters[a][1] > hyperparameters[a][0]: hyperparameters[a][1] = np.random.uniform(0.05, hyperparameters[a][0])
                    hyperparameters[a][2] = np.random.randn() * 0.1 + hyperparameters[a][2]
                    if hyperparameters[a][2] < 0.1 or hyperparameters[a][2] < hyperparameters[a][0]: hyperparameters[a][2] = np.random.uniform(hyperparameters[a][0], 3)

                    hyperparameters[a][3] = np.random.randn() * 0.1 + hyperparameters[a][3]
                    if hyperparameters[a][3] < 0.1 or hyperparameters[a][3] > 1: hyperparameters[a][3] = np.random.uniform(0.08, 0.5)
                    hyperparameters[a][4] = np.random.randn() * 0.1 + hyperparameters[a][4]
                    if hyperparameters[a][4] < 0.07 or hyperparameters[a][4] > hyperparameters[a][3]: hyperparameters[a][4] = np.random.uniform(0.07, hyperparameters[a][3])
                    hyperparameters[a][5] = np.random.randn() * 0.1 + hyperparameters[a][5]
                    if hyperparameters[a][5] < 0.1 or hyperparameters[a][4] < hyperparameters[a][4]: hyperparameters[a][5] = np.random.uniform(hyperparameters[a][3], 2)
                
                T -= DT

    for k, par in hyper.items():
        print(k)
        print(par)

    DATA = pd.DataFrame()
    for k, dat in data.items(): DATA = pd.concat([DATA, dat], ignore_index=True)


    na = len(areas)
    nr = int(na / 3)
    fig, axes = plt.subplots(nr, 3)
    i = -1
    j = -1
    for k in np.arange(na):
        if np.mod(k, 3) == 0:
            j += 1
        if np.mod(k, 3) == 0:
            i = -1
        i += 1
        
        axes[j, i].set_title(areas[k])
        ax2 = axes[j, i].twiny()
        sns.scatterplot(
            x = DATA.X.loc[DATA.SOURCE == areas[k]],
            y= DATA.Y.loc[DATA.SOURCE == areas[k]],
            hue=DATA.set.loc[DATA.SOURCE == areas[k]],
            ax = axes[j, i]
        )
        sns.scatterplot(
            x= DATA.Ypred.loc[DATA.SOURCE == areas[k]],
            y= DATA.Y.loc[DATA.SOURCE == areas[k]],
            hue=DATA.set.loc[DATA.SOURCE == areas[k]],
            ax = ax2,
            palette=sns.color_palette("deep")[2:4],
            legend=False
        )
        ax2.xaxis.set_ticks_position("top")
    fig.tight_layout()
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
def gaussian_process_kfold(NET, areas, k=10, MAXIT=100, ker="RBF", feature="D"):
    from sklearn.metrics import mean_squared_error
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

    if ker == "RBF":
        kernel = RBF
    elif ker == "Matern":
        kernel = Matern
    else: raise ValueError("That kernel does not exist")

    K = np.floor(NET.nodes / k).astype(int)

    data = { a : pd.DataFrame() for a in areas}

    for it in np.arange(MAXIT):

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

                kernel = kernel(
                    length_scale=2,
                    length_scale_bounds=(
                        0.5,
                        3)
                    ) + WhiteKernel(noise_level=5)  

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
    for k, dat in data.items(): DATA = pd.concat([DATA, dat], ignore_index=True)

    DATA = DATA.loc[DATA.Y > 0]

    folderpath = f"../pickle/PRED/GP/{ker}/{feature}"
    Path(folderpath).mkdir(parents=True, exist_ok=True)
    DATA.to_pickle(f"{folderpath}/{k}.pk")

def plot_xgboost(NET, areas, k=3, l=100):
    import xgboost as xgb

    param = {'max_depth': 3, 'objective': 'reg:squarederror'}
    param['eval_metric'] = 'mae'
    param['tree_method'] = 'hist'
    param["min_child_weight"] = 3
    param["gamma"] = 0.1
    param["eta"] = 1


    data = pd.DataFrame()
    pred = pd.DataFrame()

    X = NET.D.copy()
    np.fill_diagonal(X, np.nan)
    mux = 0.
    sdx = 1.
    np.fill_diagonal(X, 0.)

    for a in areas:
        iA = match([a], NET.labels)

        x= X[iA, :].ravel()
        dx = np.linspace(np.min(x), np.max(x), l)

        x = x[:NET.nodes]
        Y = np.log(1 + NET.C[iA, :]).ravel()

        yno = Y == 0
        x = x[~yno]
        Y = Y[~yno]

        N = x.shape[0]
        K = np.floor(N / k, dtype=float)

        indices = np.arange(N)
        perm = np.random.permutation(N)

        for j in np.arange(1):
            if j < k - 1:
                in_test = np.arange(j * (K), (j + 1 ) * (K), dtype=int)
            else:
                in_test = np.arange(j * (K), N, dtype=int)
            ntest = in_test.shape[0]
            in_train = np.array([ii for ii in indices if ii not in in_test])
            ntrain = in_train.shape[0]

            Xtrain = x[perm[in_train]]
            Xtest = x[perm[in_test]]

            Ytrain = Y[perm[in_train]]
            Ytest = Y[perm[in_test]]

            model = xgb.XGBRegressor(**param)
            model.fit(
                Xtrain.reshape(-1, 1), Ytrain,
                eval_set=[(Xtest.reshape(-1, 1), Ytest), (Xtrain.reshape(-1, 1), Ytrain)]
            )

            mean_pred = model.predict(dx.reshape(-1, 1))

            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "Y" : list(Ytrain) + list(Ytest),
                            "X" : list(Xtrain * sdx + mux) + list(Xtest * sdx + mux),
                            "set" : ["train"] * ntrain + ["test"] * ntest,
                            "SOURCE" : [a] * N
                        }
                    )
                ], ignore_index=True
            )
            pred = pd.concat(
                [
                    pred,
                    pd.DataFrame(
                        {
                            "Y" : mean_pred,
                            "X" : dx * sdx + mux,
                            "SOURCE" : [a] * l
                        }
                    )
                ], ignore_index=True
            )
    
    g = sns.FacetGrid(
        data=data,
        col="SOURCE",
        col_wrap=3,
        hue="set"
    )
    g.map_dataframe(
        sns.scatterplot,
        x="X",
        y="Y"
    )

    for i, axes in enumerate(g.axes.flat):
        sns.lineplot(
            data=pred.loc[pred.SOURCE == areas[i]],
            x="X",
            y="Y",
            color=sns.color_palette("deep")[2],
            ax=axes
        )

    plt.show()

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

                X_TR[a] += list(Xtrain[:, 0] * sdx[0] + mux[0])
                X_TS[a] += list(Xtest[:, 0] * sdx[0] + mux[0])

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
    for k, dat in data.items(): DATA = pd.concat([DATA, dat], ignore_index=True)

    folderpath = f"../pickle/PRED/XGBOOST/{feature}"
    Path(folderpath).mkdir(parents=True, exist_ok=True)
    DATA.to_pickle(f"{folderpath}/{k}.pk")

    # order = DATA.groupby(["SOURCE", "set"])["acc"].mean().reset_index()
    # order = order.loc[order.set == "test"].sort_values("acc", ascending=False)

    # sns.violinplot(
    #     data=DATA,
    #     x="SOURCE",
    #     y="acc",
    #     hue="set",
    #     order = order["SOURCE"]
    # )

    # plt.show()

    # DATA = DATA.loc[DATA.set == "test"]

    # da = DATA.groupby(["SOURCE", "Y", "X"])["YPRED_PROBA"].mean().reset_index()

    # da1 = da[["X", "Y", "SOURCE"]]
    # da1["set"] = "data"
    # da2 = da[["X", "YPRED_PROBA", "SOURCE"]]
    # da2.columns = ["X", "Y", "SOURCE"]
    # da2.Y.loc[da2.Y > 0.5] = 1
    # da2.Y.loc[da2.Y <= 0.5] = 0
    # da2["set"] = "pred"

    # da = pd.concat([da1, da2], ignore_index=True)
    # da["Y"] = pd.Categorical(da["Y"], [1, 0])


    # g=sns.FacetGrid(
    #     data=da,
    #     col="SOURCE",
    #     col_wrap=3
    # )

    # g.map_dataframe(
    #     sns.stripplot,
    #     x="X",
    #     y="Y",
    #     s=3,
    #     dodge=True,
    #     palette="deep",
    #     hue="set",
    #     alpha=1
    # )

    # g.add_legend()

    # plt.show()

def predict_xgboost_bin_kfold(listargs):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, confusion_matrix

    NET=listargs[0]
    a=listargs[1]
    k=listargs[2]
    MAXIT=listargs[3]
    kk=listargs[4]

    param = {'max_depth': 3, 'objective': 'binary:logistic'}
    param['tree_method'] = 'hist'
    param["min_child_weight"] = 7
    param["gamma"] = 10
    param["lambda"] = 8
    param["eta"] = 1
    
    indices = np.arange(NET.nodes)

    K = np.floor(NET.nodes / k).astype(int)

    data = pd.DataFrame()

    print(kk, a)

    it = 0
    while it < MAXIT:
        flag = False

        perm_cols = np.random.permutation(NET.nodes)
        perm_rows = list(np.random.permutation(np.arange(NET.nodes, NET.rows)))
        perm_rows = list(perm_cols) + perm_rows

        Y_TR = []
        YPRED_TR =[]
        YPRED_PROBA_TR=[]

        Y_TS = []
        YPRED_TS =[]
        YPRED_PROBA_TS=[]

        TARGET_TR = []
        TARGET_TS = []

        for j in range(k):
            if j < k -1 : in_test = list(np.arange(j * (K), (j + 1 ) * (K), dtype=int))
            else: in_test = list(np.arange(j * (K), NET.nodes, dtype=int))
            in_train = [ii for ii in indices if ii not in in_test]

            l_in_train = len(in_train)

            in_cols = in_train + in_test
            in_rows = in_train + in_test + list(np.arange(NET.nodes, NET.rows))

            labels = list(NET.labels[perm_rows][in_rows])

            X = np.zeros((2, NET.rows, NET.rows))
            X[0, :, :] = NET.D.copy()[perm_rows, :][:, perm_rows][in_rows, :][:, in_rows]
            C = NET.C.copy()[perm_rows, :][:, perm_cols][in_rows, :][:, in_cols]
            for ii in np.arange(NET.rows):
                for jj in np.arange(ii + 1, NET.rows):
                    X[1, ii, jj] = ct.D1_2_4(C[ii, :l_in_train], C[jj, :l_in_train], ii, jj)
                    np.seterr(divide='ignore', invalid='ignore')
                    X[1, ii, jj] = 1 / X[1, ii, jj] + 1
                    X[1, jj, ii] = X[1, ii, jj]
            X[1, :, :][X[1, :, :] == np.Inf] = np.max(X[1, :, :][X[1, :, :] < np.Inf]) * 1.05

            for i in [0, 1]: np.fill_diagonal(X[i, :, :], np.nan)
            mux = np.nanmean(X, axis=(1, 2))
            sdx = np.nanstd(X, axis=(1, 2))
            for i in [0, 1]:
                X[i] = (X[i] - mux[i]) / sdx[i]
                np.fill_diagonal(X[i], 0.)

            iA = match([a], labels)

            Xtrain= X[:, iA, :l_in_train].reshape(X.shape[0], -1)
            Ytrain = np.array([C[iA, :l_in_train] > 0]).ravel().astype(int)

            Xtest = X[:, iA, l_in_train:NET.nodes].reshape(X.shape[0], -1)
            Ytest = np.array(C[iA, l_in_train:NET.nodes] > 0).ravel().astype(int)

            if iA < l_in_train:
                keep = [i for i in np.arange(l_in_train) if i != iA]
                Xtrain  = Xtrain[:, keep]
                Ytrain = Ytrain[keep]

            if np.unique(Ytrain).shape[0] < 2:
                flag = True
                break

            model = xgb.XGBClassifier(**param)

            model.fit(Xtrain.reshape(-1, Xtrain.shape[0]), Ytrain)

            Ytrain_pred = model.predict(Xtrain.reshape(-1, Xtrain.shape[0]))
            Ytest_pred = model.predict(Xtest.reshape(-1, Xtest.shape[0]))

            Ytrain_pred_proba = model.predict_proba(Xtrain.reshape(-1, Xtrain.shape[0]))
            Ytest_pred_proba = model.predict_proba(Xtest.reshape(-1, Xtest.shape[0]))

            Y_TR += list(Ytrain)
            YPRED_TR += list(Ytrain_pred)
            YPRED_PROBA_TR += list(Ytrain_pred_proba)

            Y_TS += list(Ytest)
            YPRED_TS += list(Ytest_pred)
            YPRED_PROBA_TS += list(Ytest_pred_proba)

            if iA < NET.nodes:
                TARGET_TR += [s for ki, s in enumerate(labels[:l_in_train]) if ki != iA]
            else:
                TARGET_TR += labels[:l_in_train]
            TARGET_TS += labels[l_in_train:NET.nodes]

        if not flag:
            acc_train = accuracy_score(Y_TR, YPRED_TR)
            acc_test = accuracy_score(Y_TS, YPRED_TS)

            tn, fp_tr, fn, tp_tr = confusion_matrix(Y_TR, YPRED_TR).ravel()
            tpr_tr = tp_tr / (tp_tr + fn)
            fpr_tr = fp_tr / (fp_tr + tn)

            tn, fp_ts, fn, tp_ts = confusion_matrix(Y_TS, YPRED_TS).ravel()
            tpr_ts = tp_ts / (tp_ts + fn)
            fpr_ts = fp_ts / (fp_ts + tn)

            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "Y" : Y_TR + Y_TS,
                            "YPRED" : YPRED_TR + YPRED_TS,
                            "YPRED_PROBA" : YPRED_PROBA_TR + YPRED_PROBA_TS,
                            "set" : ["train"] * len(Y_TR) + ["test"] * len(Y_TS),
                            "SOURCE" : [a] * (len(Y_TR) + len(Y_TS)),
                            "TARGET" : TARGET_TR + TARGET_TS,
                            "iteration" : [it] * (len(Y_TR) + len(Y_TS)),
                            "acc" : [acc_train] * len(Y_TR) + [acc_test] * len(Y_TS),
                            "TPR" : [tpr_tr] * len(Y_TR) + [tpr_ts] * len(Y_TS),
                            "FPR" : [fpr_tr] * len(Y_TR) + [fpr_ts] * len(Y_TS)
                        }
                    )
                ], ignore_index=True
            )
            it += 1

    return data

def plot_xgboost_bin_mcmc(NET, areas, k=10, MAXIT=100):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    
    indices = np.arange(NET.nodes)

    K = np.floor(NET.nodes / k).astype(int)

    hyper = {a : {
            "min_child_weight" : 5,
            "gamma" : 8,
            "lambda" : 5,
            "acc_tr" : 0,
            "acc_ts" : 0,
            "fitness" : 0
        } for a in areas
    }

    min_fitness = {a : 100 for a in areas}
    fitness_old = {a : 10 for a in areas}
    fitness = {a : 0 for a in areas}

    hyperparameters = {a: {
            "min_child_weight" : 8,
            "gamma" : 8,
            "lambda" : 5,
            "eta" : 1,
            "max_depth" : 3,
            "objective" : "binary:logistic"
        }  for a in areas
    }

    params = ["min_child_weight", "gamma", "lambda"]

    for a in areas:
        print(a)
        T = 200
        DT = (T - 0.1)/MAXIT
        for _ in np.arange(MAXIT):
            perm_cols = np.random.permutation(NET.nodes)
            perm_rows = list(np.random.permutation(np.arange(NET.nodes, NET.rows)))
            perm_rows = list(perm_cols) + perm_rows

            Y_TR = []
            YPRED_TR =[]
            Y_TS = []
            YPRED_TS =[]
            for j in range(k):
                if j < k -1 : in_test = list(np.arange(j * (K), (j + 1 ) * (K), dtype=int))
                else: in_test = list(np.arange(j * (K), NET.nodes, dtype=int))
                in_train = [ii for ii in indices if ii not in in_test]

                l_in_train = len(in_train)

                in_cols = in_train + in_test
                in_rows = in_train + in_test + list(np.arange(NET.nodes, NET.rows))

                labels = list(NET.labels[perm_rows][in_rows])

                X = np.zeros((2, NET.rows, NET.rows))
                X[0, :, :] = NET.D.copy()[perm_rows, :][:, perm_rows][in_rows, :][:, in_rows]
                C = NET.C.copy()[perm_rows, :][:, perm_cols][in_rows, :][:, in_cols]
                for ii in np.arange(NET.rows):
                    for jj in np.arange(ii + 1, NET.rows):
                        X[1, ii, jj] = ct.D1_2_4(C[ii, :l_in_train], C[jj, :l_in_train], ii, jj)
                        np.seterr(divide='ignore', invalid='ignore')
                        X[1, ii, jj] = 1 / X[1, ii, jj] + 1
                        X[1, jj, ii] = X[1, ii, jj]
                X[1, :, :][X[1, :, :] == np.Inf] = np.max(X[1, :, :][X[1, :, :] < np.Inf]) * 1.95
                np.seterr(divide='ignore', invalid='ignore')
                X[1, :, :] = np.log(X[1, :, :])

                for i in np.arange(2): np.fill_diagonal(X[i, :, :], np.nan)
                mux = np.nanmean(X, axis=(1, 2))
                sdx = np.nanstd(X, axis=(1, 2))
                for i in np.arange(2):
                    X[i] = (X[i] - mux[i]) / sdx[i]
                    np.fill_diagonal(X[i], 0.)

                iA = match([a], labels)

                Xtrain= X[:, iA, :l_in_train].reshape(X.shape[0], -1)
                Ytrain = np.array([C[iA, :l_in_train] > 0]).ravel().astype(int)

                Xtest = X[:, iA, l_in_train:NET.nodes].reshape(X.shape[0], -1)
                Ytest = np.array(C[iA, l_in_train:NET.nodes] > 0).ravel().astype(int)

                if iA < l_in_train:
                    keep = [i for i in np.arange(l_in_train) if i != iA]
                    Xtrain  = Xtrain[:, keep]
                    Ytrain = Ytrain[keep]

                # if np.unique(Ytest).shape[0] < 2 or np.unique(Ytrain).shape[0] < 2: continue

                model = xgb.XGBClassifier(**hyperparameters[a])

                model.fit(Xtrain.reshape(-1, Xtrain.shape[0]), Ytrain)

                Ytrain_pred = model.predict(Xtrain.reshape(-1, Xtrain.shape[0]))
                Ytest_pred = model.predict(Xtest.reshape(-1, Xtest.shape[0]))

                Y_TR += list(Ytrain)
                YPRED_TR += list(Ytrain_pred)
                Y_TS += list(Ytest)
                YPRED_TS += list(Ytest_pred)

            acc_train = accuracy_score(Y_TR, YPRED_TR)
            acc_test = accuracy_score(Y_TS, YPRED_TS)

            fitness[a] = np.abs(acc_train - acc_test) + acc_train / 10

            print("score:\t", fitness[a])

            if min_fitness[a] > fitness[a]:
                min_fitness[a] = fitness[a]
                print("\n\nsaved", fitness[a], acc_train)
                for p in params:
                    hyper[a][p] = hyperparameters[a][p]
                hyper[a]["acc_tr"] = acc_train
                hyper[a]["acc_ts"] = acc_test
                hyper[a]["fitness"] = fitness[a]

            if fitness[a] > fitness_old[a]:
                fitness_old[a] = fitness[a]

                hyperparameters[a]["min_child_weight"] = np.random.randn() * 2 + hyperparameters[a]["min_child_weight"]
                if hyperparameters[a]["min_child_weight"] < 0 or hyperparameters[a]["min_child_weight"] > 10: hyperparameters[a]["min_child_weight"] = np.random.uniform(1, 10)
                hyperparameters[a]["gamma"] = np.random.randn() * 5 + hyperparameters[a]["gamma"]
                if hyperparameters[a]["gamma"] < 0.1 or hyperparameters[a]["gamma"] > 20: hyperparameters[a]["gamma"] = np.random.uniform(0.1, 20)
                hyperparameters[a]["lambda"] = np.random.randn() * 5 + hyperparameters[a]["lambda"]
                if hyperparameters[a]["lambda"] < 0.1 or hyperparameters[a]["lambda"] > 20: hyperparameters[a]["lambda"] = np.random.uniform(0.1, 20)
            
            elif np.exp(-(fitness_old[a] - fitness[a]) / T) > np.random.rand():
                fitness_old[a] = fitness[a]

                hyperparameters[a]["min_child_weight"] = np.random.randn() * 2 + hyperparameters[a]["min_child_weight"]
                if hyperparameters[a]["min_child_weight"] < 0 or hyperparameters[a]["min_child_weight"] > 10: hyperparameters[a]["min_child_weight"] = np.random.uniform(1, 10)
                hyperparameters[a]["gamma"] = np.random.randn() * 5 + hyperparameters[a]["gamma"]
                if hyperparameters[a]["gamma"] < 0.1 or hyperparameters[a]["gamma"] > 20: hyperparameters[a]["gamma"] = np.random.uniform(0.1, 20)
                hyperparameters[a]["lambda"] = np.random.randn() * 5 + hyperparameters[a]["lambda"]
                if hyperparameters[a]["lambda"] < 0.1 or hyperparameters[a]["lambda"] > 20: hyperparameters[a]["lambda"] = np.random.uniform(0.1, 20)
            
            T -= DT

    for k, v in hyper.items():
        print(k)
        for kv, vv in v.items():
            print("\t", kv, vv)
        print("\n")

def imputation(listargs):

    NET = listargs[0]
    a = listargs[1]
    MAXIT = listargs[2]

    print(a)

    from sklearn.metrics import mean_squared_error
    from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    imputation = pd.DataFrame()

    X = NET.D.copy()
    np.fill_diagonal(X, np.nan)
    # mux = np.nanmean(X)
    # sdx = np.nanstd(X)
    # X = (X - mux) / sdx
    mux = 0.
    sdx = 1.
    np.fill_diagonal(X, 0.)

    iA = match([a], NET.labels)

    for i in np.arange(MAXIT):
        ####

        x= X[iA, :].ravel()
        ximputation = x[NET.nodes:]

        x = x[:NET.nodes]
        Y = np.log(1 + NET.C[iA, :]).ravel()

        yno = Y == 0
        x = x[~yno]
        Y = Y[~yno]
        Y = np.log(Y)

        kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(0.1, 10)) * RBF(length_scale=12, length_scale_bounds=(10, 40)) +  WhiteKernel(noise_level=2, noise_level_bounds=(0.001, 5)) 
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)
        gaussian_process.fit(x.reshape(-1, 1), Y)

        print(f"reg {i}\t", mean_squared_error(Y, gaussian_process.predict(x.reshape(-1, 1))))

        pred = gaussian_process.predict(ximputation.reshape(-1, 1))
        pred = np.exp(pred) - 1
        pred[pred < 0] = 0
        
        ####

        x= X[iA, :].ravel()

        x = x[:NET.nodes]
        Y = NET.C[iA, :].ravel()

        yno = Y == 0
        Y[~yno] = 1

        if iA < NET.nodes:
            x = np.array([e for i, e in enumerate(x) if i != iA])
            Y = np.array([e for i, e in enumerate(Y) if i != iA])

        kernel = ConstantKernel(constant_value=10, constant_value_bounds=(0.01, 40)) * RBF(length_scale=12, length_scale_bounds=(10, 50)) #+ ConstantKernel(constant_value=0.01, constant_value_bounds=(0.001, 0.09)) *  WhiteKernel(noise_level=2, noise_level_bounds=(0.1, 20))

        gaussian_process = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10)
        gaussian_process.fit(x.reshape(-1, 1), Y)

        print(f"clf {i}\t", gaussian_process.score(x.reshape(-1, 1), Y))

        bin_pred = gaussian_process.predict(ximputation.reshape(-1, 1))


        imputation = pd.concat(
            [
                imputation,
                pd.DataFrame(
                    {
                        "distance" : ximputation * sdx + mux,
                        "N" : pred * bin_pred,
                        "SOURCE" : [a] * (NET.rows - NET.nodes),
                        "TARGET" : NET.labels[NET.nodes:],
                        "iteration" : [i] * (NET.rows - NET.nodes)
                    }
                )
            ], ignore_index=True
        )

    return imputation

def parallel_imputation(NET, areas, MAXIT=10):
    import multiprocessing as mp

    paralist = []
    for a in areas: paralist.append([NET, a, MAXIT])

    with mp.Pool(4) as p:
      process = p.map(imputation, paralist)
    
    pred = pd.DataFrame()
    for p in process:
        pred = pd.concat([pred, p], ignore_index=True)
    
    pred.to_pickle("../pickle/gaussian_process_imputation.pk")

def see_imputation_scatter(NET):

    IMPUTATION = pd.read_pickle("../pickle/gaussian_process_imputation.pk")
    IMPUTATION = IMPUTATION.groupby(["distance", "SOURCE", "TARGET"])["N"].mean().reset_index()
    IMPUTATION["set"] = ["imputation"] * IMPUTATION.shape[0]

    DATA = adj2df(NET.C)
    DATA.columns = ["source", "target", "N"]
    DATA["SOURCE"] = NET.labels[DATA.source]
    DATA["TARGET"] = NET.labels[DATA.target]
    D = adj2df(NET.D[:, :NET.nodes])
    DATA["distance"] = D.weight

    DATA["set"] = ["data"] * DATA.shape[0]

    DATA = pd.concat([DATA, IMPUTATION], ignore_index=True).sort_values("distance")
    DATA = DATA.loc[DATA.N >= 1]
    DATA = DATA.loc[DATA.SOURCE != DATA.TARGET]

    DATA["N"] = np.log(1+DATA["N"])

    DATAs = DATA.loc[np.isin(DATA.SOURCE, areas), ["distance", "SOURCE", "N"]]
    DATAs.columns = ["distance", "AREA", "N"]

    DATAt = DATA.loc[np.isin(DATA.TARGET, areas), ["distance", "TARGET", "N"]]
    DATAt.columns = ["distance", "AREA", "N"]

    DATAs["dir"] = "src"
    DATAt["dir"] = "tgt"

    data = pd.concat([DATAs, DATAt], ignore_index=True)
    
    sns.lmplot(
        data=data,
        x="distance",
        y="N",
        hue="dir",
        col="AREA",
        col_wrap=3,
        lowess=True
    )
    plt.show()

    IMPUTATION = pd.read_pickle("../pickle/gaussian_process_imputation.pk")
    IMPUTATION = IMPUTATION.groupby(["distance", "SOURCE", "TARGET"])["N"].mean().reset_index()
    IMPUTATION["set"] = ["imputation"] * IMPUTATION.shape[0]

    DATA = adj2df(NET.C)
    DATA.columns = ["source", "target", "N"]
    DATA["SOURCE"] = NET.labels[DATA.source]
    DATA["TARGET"] = NET.labels[DATA.target]
    D = adj2df(NET.D[:, :NET.nodes])
    DATA["distance"] = D.weight

    DATA["set"] = ["data"] * DATA.shape[0]

    DATA = pd.concat([DATA, IMPUTATION], ignore_index=True).sort_values("distance")
    DATA = DATA.loc[DATA.N >= 1]
    DATA = DATA.loc[DATA.SOURCE != DATA.TARGET]

    DATA["N"] = np.log(1+DATA["N"])

    _, ax = plt.subplots(1, 2)

    sns.scatterplot(
        data=DATA,
        x="distance",
        y="N",
        hue="set",
        s=10,
        ax=ax[0]
    )

    sns.heatmap(
        data=DATA.pivot(index="SOURCE", columns="TARGET", values="N"),
        cmap="viridis",
        ax=ax[1]
    )
    plt.show()

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
opt_score = ["_X", "_S"]
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

    areas = ["v4c", "v1c", "8l", "tepd", "mip", "stpc"]

    # plot_error_histogram()
    # plot_reg_overview()
    # plot_ACC()
    # plot_residuals()
    # plot_residuals_all()
    # plot_error_violin()
 
    gaussian_process_kfold(NET, areas, k=10, MAXIT=10)
    # plot_gaussian_process_mcmc(NET, areas, k=10,l=50)
    # plot_gaussian_process_bin(NET, areas, k=3)

    # plot_xgboost(NET, areas, k=10)
    xgboost_bin_kfold(NET, areas, k=10, MAXIT=100)
    # plot_xgboost_bin_mcmc(NET, areas, k=10, MAXIT=20)

    # parallel_predict_areas_bin(NET, areas, k=10, MAXIT=20)
    parallel_predict_areas(NET, NET.labels, MAXIT=100)

    # see_imputation_scatter(NET)
    # parallel_imputation(NET, NET.labels)