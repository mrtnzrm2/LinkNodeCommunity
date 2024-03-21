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
from networks.MAC.mac53 import SUPRA53, INFRA53
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
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "D1_2_4"
discovery = "discovery_6"
bias = 0.
alpha = 0.
opt_score = ["_X", "_S"]
version = "53d91"
__nodes__ = 53
__inj__ = 53

if __name__ == "__main__":
    NET = SUPRA53(
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

    NET2 = INFRA53(
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
    

    areas = ["v1", "8l", "vip", "7a", "v2", "stpc"]
    k = 3

    Supra = NET.C
    Infra = NET2.C
    D = NET.D
    np.fill_diagonal(D, np.nan)

    SSupra = np.zeros((NET.rows, NET.rows))
    SInfra = np.zeros((NET.rows, NET.rows))

    for i in np.arange(NET.rows):
        for j in np.arange(i + 1, NET.rows):
            SSupra[i, j] = ct.D1_2_4(Supra[i, :], Supra[j, :], i, j)
            SSupra[i, j] = 1 / SSupra[i, j] + 1
            SSupra[j, i] = SSupra[i, j]

            SInfra[i, j] = ct.D1_2_4(Infra[i, :], Infra[j, :], i, j)
            SInfra[i, j] = 1 / SInfra[i, j] + 1
            SInfra[j, i] = SInfra[i, j]

    SSupra[SSupra == np.Inf] = np.max(SSupra[SSupra < np.Inf]) * 1.95
    SInfra[SInfra == np.Inf] = np.max(SInfra[SInfra < np.Inf]) * 1.95

    SSupra = np.log(SSupra)
    SInfra = np.log(SInfra)
    np.fill_diagonal(SSupra, np.nan)
    np.fill_diagonal(SInfra, np.nan)

    A = np.zeros((3, NET.rows, NET.rows))
    A[0, :, :] = D
    A[1, :, :] = SSupra
    # A[2, :, :] = SInfra

    mu = np.nanmean(A, axis=(1, 2))
    sd = np.nanstd(A, axis=(1, 2))

    for i in [1, 0]:
      A[i, :, :] = (A[i, :, :] - mu[i]) / sd[i]
      np.fill_diagonal(A[i, :, :], 0.)
      # print(np.min(A[i, :, :][A[i, :, :] != np.min(A[i, :, :])]) - np.min(A[i, :, :]))

    from sklearn.metrics import mean_squared_error
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    data = pd.DataFrame()

    for a in areas:
        ia = match([a], NET.labels)

        x= A[:, ia, :]
        x = x[:, :, :NET.nodes].reshape(3, -1)
        Y = np.log(1 + NET2.C[ia, :]).ravel()

        yno = (Y == 0) | (Y == -np.Inf)
        x = x[:, ~yno]
        Y = Y[~yno]

        N = x.shape[1]
        K = np.floor(N / k, dtype=float)

        indices = np.arange(N)
        perm = np.random.permutation(N)
        # perm = np.arange(N)

        for j in np.arange(1):
            if j < k - 1:
                in_test = np.arange(j * (K), (j + 1 ) * (K), dtype=int)
            else:
                in_test = np.arange(j * (K), N, dtype=int)
            ntest = in_test.shape[0]
            in_train = np.array([ii for ii in indices if ii not in in_test])
            ntrain = in_train.shape[0]

            Xtrain = x[:, perm[in_train]]
            Xtest = x[:, perm[in_test]]

            Ytrain = Y[perm[in_train]]
            Ytest = Y[perm[in_test]]

            kernel = RBF(length_scale=0.5, length_scale_bounds=(0.1, 0.75))  + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 3))  

            gaussian_process = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=100, normalize_y=True
            )

            gaussian_process.fit(Xtrain.reshape(-1, 3), Ytrain)

            pred_train = gaussian_process.predict(Xtrain.reshape(-1, 3))
            pred_test = gaussian_process.predict(Xtest.reshape(-1, 3))


            print(a)
            print(np.exp(gaussian_process.kernel_.theta))
            print("Train", mean_squared_error(Ytrain, pred_train))
            print("Test", mean_squared_error(Ytest, pred_test))

            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                      {
                          "Y" : list(Ytrain) + list(Ytest) +list(pred_train) + list(pred_test),
                          "X" : list(Xtrain[0, :]) + list(Xtest[0, :]) + list(Xtrain[0, :]) + list(Xtest[0, :]),
                          "set" : ["train"] * ntrain + ["test"] * ntest + ["pred_train"] * ntrain + ["pred_test"] * ntest,
                          "set2" : ["1"] * ntrain + ["2"] * ntest + ["1"] * ntrain + ["2"] * ntest,
                          "SOURCE" : [a] * (ntrain + ntest) + [a] * (ntrain + ntest)
                      }
                  )
                ], ignore_index=True 
            ) 

    sns.lmplot(
        data=data,
        col="SOURCE",
        row="set2",
        y="Y",
        x="X",
        hue="set",
        fit_reg=False
    )
    plt.show()