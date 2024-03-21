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
from various.network_tools import *

def one_sample_ttest(data : pd.DataFrame):
   
    print("One sample ttest with mu_null = 0.")

    from scipy.stats import ttest_1samp

    data = data.groupby(["SOURCE", "set", "iteration"])["error"].mean().reset_index()

    areas = np.unique(data.SOURCE)
    onetest = {}

    for a in areas:
        onetest[a] = ttest_1samp(data["error"].loc[data.SOURCE == a], popmean=0.)

    return onetest

def compare_variances(data1 : pd.DataFrame, data2 : pd.DataFrame):
    data1 = data1.groupby(["SOURCE", "set", "iteration"])["error"].mean().reset_index()
    data1 = data1.groupby(["SOURCE", "set"])["error"].var().reset_index()

    data2 = data2.groupby(["SOURCE", "set", "iteration"])["error"].mean().reset_index()
    data2 = data2.groupby(["SOURCE", "set"])["error"].var().reset_index()

    data = pd.DataFrame(
        {
            "SOURCE" : data1.SOURCE,
            "r_variance": data2.error / data2.error
        }
    )

    data = data.loc[data.r_variance >= 2]
    print(data)

def ttest(data1 : pd.DataFrame, data2: pd.DataFrame, variable="error", alternative="two-sided"):
    
    """mu1 neq mu2"""
    
    print("T test for a two-tailed test")

    data1 = data1.groupby(["SOURCE", "iteration"])[variable].mean().reset_index()
    data2 = data2.groupby(["SOURCE", "iteration"])[variable].mean().reset_index()

    import scipy.stats as stats

    areas = np.unique(data1.SOURCE)

    ttest = {}

    for a in areas:
        x_1 = data1[variable].loc[data1.SOURCE == a]
        x_2 = data2[variable].loc[data2.SOURCE == a]

        ttest[a] = stats.ttest_ind(x_1, x_2, alternative=alternative)
    
    return ttest

def ftest(data1 : pd.DataFrame, data2 : pd.DataFrame, variable="error", alpha=0.05):

    """ F = sig1^2/sig2^2 and; null: sig1 === sig2; al: F > F(alpha, N1-1, N2-1) """

    print("F test for an upper one-tailed test")

    import scipy.stats as stats

    data1 = data1.groupby(["SOURCE", "iteration"])[variable].mean().reset_index()
    data2 = data2.groupby(["SOURCE", "iteration"])[variable].mean().reset_index()

    areas = np.unique(data1.SOURCE)

    ftest = {}

    for a in areas:
        f = np.var(data1[variable].loc[data1.SOURCE == a], ddof=1) / np.var(data2[variable].loc[data2.SOURCE == a], ddof=1)
        dun = data2.loc[data2.SOURCE == a].shape[0] - 1
        nun = data1.loc[data1.SOURCE == a].shape[0] - 1
        fp = stats.f.ppf(q=1-alpha, dfn=nun, dfd=dun)
        ftest[a] = f > fp

    return ftest

def reg_validation():

    paths = {
        "MATERN_D" : "../pickle/PRED/GP/MATERN/NU_0_5/D",
        "MATERN_DSIM" : "../pickle/PRED/GP/MATERN/NU_0_5/DSIM",
        "RBF_D" : "../pickle/PRED/GP/RBF/D",
        "RBF_SIM" : "../pickle/PRED/GP/RBF/SIM",
        "RBF_DSIM" : "../pickle/PRED/GP/RBF/DSIM"
    }

    name = "gaussian_process_regression_validation.pk"

    data = pd.read_pickle(os.path.join(paths["RBF_D"], name))
    data = data.loc[(data.set == "test") & (data.Y > 0)]

    areas = np.unique(data.SOURCE)
    al_models = ["RBF_SIM", "RBF_DSIM", "MATERN_D", "MATERN_DSIM"]

    data_al = {a : pd.read_pickle(os.path.join(paths[a], name)) for a in al_models}
    for a in data_al.keys():
        data_al[a] = data_al[a].loc[(data_al[a].set == "test") & (data_al[a].Y > 0)]

    area_reg_validation = pd.DataFrame(np.repeat(areas, 4), columns=["SOURCE"])
    area_reg_validation["test"] = np.tile(np.repeat(["t-test one-sample", "f-test upper one-tailed"], 2), areas.shape[0])
    area_reg_validation["pvalue"] = np.tile([0.05, 0.001], areas.shape[0] * 2)
    
    area_reg_validation["RBF_D"] = np.nan
    for al in al_models:
        area_reg_validation[al] = np.nan

    # Check how statistically significant the mean of the error distribution differs from
    # zero

    test1 = one_sample_ttest(data)
    for a in test1.keys():

        if test1[a].pvalue < 0.05:
            area_reg_validation["RBF_D"].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.05) & (area_reg_validation.test == "t-test one-sample")] = False
        else:
            area_reg_validation["RBF_D"].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.05) & (area_reg_validation.test == "t-test one-sample")] = True
        if test1[a].pvalue < 0.001:
            area_reg_validation["RBF_D"].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.001) & (area_reg_validation.test == "t-test one-sample")] = False
        else:
            area_reg_validation["RBF_D"].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.001) & (area_reg_validation.test == "t-test one-sample")] = True

    # Check how statistically significance the variance of the null model is greater
    # than the variance of the alternative models

    for m, model in data_al.items():
        test = one_sample_ttest(model)
        for a in test.keys():
            p_0_05 = area_reg_validation["RBF_D"].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.05) & (area_reg_validation.test == "t-test one-sample")]
            p_0_05 = np.unique(p_0_05)

            if p_0_05.shape[0] == 1 and not p_0_05[0]:
                if test[a].pvalue < 0.05:
                    area_reg_validation[m].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.05) & (area_reg_validation.test == "t-test one-sample")] = False
                else:
                    area_reg_validation[m].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.05) & (area_reg_validation.test == "t-test one-sample")] = True
            
            p_0_001 = area_reg_validation["RBF_D"].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.001) & (area_reg_validation.test == "t-test one-sample")]
            p_0_001 = np.unique(p_0_001)

            if p_0_001.shape[0] == 1 and not p_0_001[0]:
                if test[a].pvalue < 0.001:
                    area_reg_validation[m].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.001) & (area_reg_validation.test == "t-test one-sample")] = False
                else:
                    area_reg_validation[m].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.001) & (area_reg_validation.test == "t-test one-sample")] = True
        
        test = ftest(data, model)
        for a in test.keys():
            p_0_05 = area_reg_validation["RBF_D"].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.05) & (area_reg_validation.test == "t-test one-sample")]
            p_0_05 = np.unique(p_0_05)

            if p_0_05.shape[0] == 1 and p_0_05[0]:
                if test[a]:
                    area_reg_validation[m].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.05) & (area_reg_validation.test == "f-test upper one-tailed")] = True
                else:
                    area_reg_validation[m].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.05) & (area_reg_validation.test == "f-test upper one-tailed")] = False
        
        test = ftest(data, model, alpha=0.001)
        for a in test.keys():
            p_0_001 = area_reg_validation["RBF_D"].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.001) & (area_reg_validation.test == "t-test one-sample")]
            p_0_001 = np.unique(p_0_001)

            if p_0_001.shape[0] == 1 and p_0_001[0]:
                if test[a]:
                    area_reg_validation[m].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.001) & (area_reg_validation.test == "f-test upper one-tailed")] = True
                else:
                    area_reg_validation[m].loc[(area_reg_validation.SOURCE == a) & (area_reg_validation.pvalue == 0.001) & (area_reg_validation.test == "f-test upper one-tailed")] = False

    # Choosing the best model for each area, considering statistical significance and
    # test

    summary = pd.DataFrame()

    data_0_05 = area_reg_validation.loc[area_reg_validation.pvalue == 0.05]

    for a in areas:
        dat = data_0_05.loc[data_0_05.SOURCE == a]
        test = []
        mm = []

        if dat["test"].loc[dat.test == "t-test one-sample"].to_numpy()[0]:
            for m in al_models:
                if dat[m].loc[dat.test == "f-test upper one-tailed"].to_numpy()[0]:
                    mm.append(m)
                    test.append("f-test upper one-tailed")
            if len(mm) == 0:
                mm.append("RBF_D")
                test.append("t-test one-sample")
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        {
                            "SOURCE" : [a] * len(mm),
                            "model" : mm,
                            "test" : test,
                            "pvalue" : [0.05] * len(mm)
                        }
                    )
                ], ignore_index=True
            )
        else:
            for m in al_models:
                if dat[m].loc[dat.test == "t-test one-sample"].to_numpy()[0]:
                    mm.append(m)
                    test.append("t-test one-sample")
            if len(mm) > 0:
                summary = pd.concat(
                    [
                        summary,
                        pd.DataFrame(
                            {
                                "SOURCE" : [a] * len(mm),
                                "model" : mm,
                                "test" : test,
                                "pvalue" : [0.05] * len(mm)
                            }
                        )
                    ], ignore_index=True
            )

    data_0_001 = area_reg_validation.loc[area_reg_validation.pvalue == 0.001]

    for a in areas:
        dat = data_0_001.loc[data_0_001.SOURCE == a]
        test = []
        mm = []

        if dat["test"].loc[dat.test == "t-test one-sample"].to_numpy()[0]:
            for m in al_models:
                if dat[m].loc[dat.test == "f-test upper one-tailed"].to_numpy()[0]:
                    mm.append(m)
                    test.append("f-test upper one-tailed")
            if len(mm) == 0:
                mm.append("RBF_D")
                test.append("t-test one-sample")
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        {
                            "SOURCE" : [a] * len(mm),
                            "model" : mm,
                            "test" : test,
                            "pvalue" : [0.001] * len(mm)
                        }
                    )
                ], ignore_index=True
            )
        else:
            for m in al_models:
                if dat[m].loc[dat.test == "t-test one-sample"].to_numpy()[0]:
                    mm.append(m)
                    test.append("t-test one-sample")
            if len(mm) > 0:
                summary = pd.concat(
                    [
                        summary,
                        pd.DataFrame(
                            {
                                "SOURCE" : [a] * len(mm),
                                "model" : mm,
                                "test" : test,
                                "pvalue" : [0.001] * len(mm)
                            }
                        )
                    ], ignore_index=True
                )


    summary["model"] = pd.Categorical(summary["model"], ["RBF_D"] + al_models)

    g = sns.FacetGrid(
        data=summary,
        row="pvalue",
        hue="test"
    )

    g.map_dataframe(
        sns.scatterplot,
        x="SOURCE",
        y="model"
    )
    g.add_legend()

    plt.xticks(rotation=90)
    plt.gcf().tight_layout()

    plt.show()

def clf_validation():
    paths = {
        "D" : "../pickle/PRED/XGBOOST/D",
        "DSIM" : "../pickle/PRED/XGBOOST/DSIM",
        "SIM": "../pickle/PRED/XGBOOST/SIM",
    }

    name = "xgb_classification_validation.pk"

    data_null = pd.read_pickle(os.path.join(paths["D"], name))
    data_al = pd.read_pickle(os.path.join(paths["SIM"], name))

    data_null = data_null.loc[data_al.set == "test"]
    data_null["model"] = "D"
    data_al = data_al.loc[(data_al.set == "test")]
    data_al["model"] = "SIM"

    test = ttest(data_null, data_al, variable="acc", alternative="two-sided")

    for a, k in test.items():
        print(a, k)

if __name__ == "__main__":
    reg_validation()
    
    