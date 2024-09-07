import numpy as np
import pandas as pd
from various.network_tools import *

class wrap_fit_exp:
  def __init__(self, lb, loc=0) -> None:
    self.coef_ = np.zeros(1)
    self.coef_[0] = lb
    self.loc=loc
  
  def predict(self, x):
    y = -self.coef_[0] * (x-self.loc) + np.log(self.coef_[0])
    return y
  
  def log_likelihood(self, x, n):
    return np.sum(n) * np.log(self.coef_[0]) - self.coef_[0] * np.sum( (x-self.loc) * n)
  
class wrap_fit_pareto:
  def __init__(self, a, xm) -> None:
    self.coef_ = np.zeros(2)
    self.coef_[0] = a
    self.coef_[1] = xm

  def predict(self, x):
    y = np.log(self.coef_[0]) + self.coef_[0] * np.log(self.coef_[1]) - (self.coef_[0] + 1) * np.log(x)
    return y
  
  def log_likelihood(self, x, n):
    return np.sum(n) * np.log(self.coef_[0]) + np.sum(n) * self.coef_[0] * np.log(self.coef_[1]) - (self.coef_[0] + 1) * np.sum(n * np.log(x))

class wrap_fit_exp_trunc:
  def __init__(self, lb, xmin, xmax) -> None:
    self.coef_ = np.zeros(3)
    self.coef_[0] = lb
    self.coef_[1] = xmin
    self.coef_[2] = xmax
  
  def predict(self, x):
    y = -self.coef_[0] * x + np.log(self.coef_[0]) - np.log(np.exp(-self.coef_[0] * self.coef_[1]) - np.exp(-self.coef_[0] * self.coef_[2]))
    return y
  
  def log_likelihood(self, x, n):
    return np.sum(n) * np.log(self.coef_[0]) - self.coef_[0] * np.sum( x * n) - np.sum(n) * np.log(np.exp(-self.coef_[0] * self.coef_[1]) - np.exp(-self.coef_[0] * self.coef_[2]))
  
class wrap_fit_pareto_trunc:
  def __init__(self, a, xm, xmax) -> None:
    self.coef_ = np.zeros(3)
    self.coef_[0] = a
    self.coef_[1] = xm
    self.coef_[2] = xmax
  
  def predict(self, x):
    y = np.log(self.coef_[0]) + self.coef_[0] * np.log(self.coef_[1]) - (self.coef_[0] + 1) * np.log(x) - np.log(1 - np.power(self.coef_[1] / self.coef_[2], self.coef_[0]))
    return y
  
  def log_likelihood(self, x, n):
    return np.sum(n) * np.log(self.coef_[0]) + np.sum(n) * self.coef_[0] * np.log(self.coef_[1]) - (self.coef_[0] + 1) * np.sum(n * np.log(x)) - np.sum(n) * np.log(1 - np.power(self.coef_[1] / self.coef_[2], self.coef_[0]))
  
def linear_fit(x, y):
  # Regression ----
  # x = (x - np.mean(x)) / np.std(x)
  from sklearn.linear_model import LinearRegression
  line_poly = LinearRegression(fit_intercept=True).fit(
    x.reshape(-1, 1), y
  )
  print(f"> Linear slope:\t{line_poly.coef_[0]:.3f}")
  print(f"> Linear interception:\t{line_poly.intercept_:.3f}")
  return line_poly

def fit_pareto_STAN(D, C, nodes, *args):
  import stan
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set_theme()
  CC = C.copy()
  CC[:nodes, :nodes] = CC[:nodes, :nodes] + CC[:nodes, :nodes].T
  CC = adj2df(CC)
  CC = CC.loc[CC.source > CC.target]
  zeros = CC.weight == 0
  DD = D[:, :nodes]
  DD = adj2df(DD)
  DD = DD.loc[DD.source > DD.target]
  DD = DD.weight.loc[~zeros].to_numpy().ravel()
  CC = CC.weight.loc[~zeros].to_numpy().ravel()
  order = np.argsort(DD)
  DD = DD[order]
  CC = CC[order]
  CC = np.array([CC[0]]+[CC[i] + np.sum(CC[:i]) for i in np.arange(1, CC.shape[0])])
  CC = CC / np.sum(C)
  # cc = np.log(CC)
  x = np.linspace(np.min(D[D > 0]), np.max(D), 100).reshape(-1, 1)
  ## fit pareto ----
  # Create data ----
  data = {
    "N" : DD.shape[0] - 1,
    "xm" : DD[0],
    "x" : DD[1:],
    "y" : CC[1:]
  }
  stan_cum_pareto_fit = """
  data {
    int N;
    real xm;
    vector[N] x;
    vector[N] y;
  }

  parameters {
    real<lower=0.1, upper=5> a;
    real<lower=0> sig;

  }

  model {
    a ~ inv_gamma(1, 1);
    y ~ normal(1 - pow(xm / x, a), sig);
  }

  """
  posterior = stan.build(stan_cum_pareto_fit, data=data)
  fit = posterior.sample(num_chains=4, num_warmup=1000, num_samples=1000).to_frame()
  a = fit["a"].mean()
  sig = fit["a"].std()
  xmin = DD[0]
  print("\na:\t", a, "\nsigma:\t", fit["sig"].mean())
  pred = wrap_fit_pareto(a, xmin)
  prob = pred.predict(x)
  ###
  # cum_pareto = lambda a, xm, x:  1 - np.power(xm / x, a)
  # fig, ax = plt.subplots(1, 1)
  # data = pd.DataFrame(
  #   {
  #     "x" : np.hstack([DD, x.ravel()]),
  #     "CCDF" : np.hstack([1 - CC, 1 - cum_pareto(a, xmin, x.ravel())]),
  #     "model" : ["data"] * CC.shape[0] + ["PARETO_STAN"] * x.shape[0] 
  #   }
  # )
  # sns.lineplot(
  #   data=data,
  #   x="x",
  #   y="CCDF",
  #   hue="model",
  #   ax=ax
  # )
  # ax.set_title(r"$\alpha$:" + f" {a:.4f} "+r"$\pm$ " + r"$\sigma_{\alpha}$:" + f" {sig:.5f} " + r"$x_{min}$:" + f" {xmin:.2f}")
  # ax.set_yscale("log")
  # fig.tight_layout()
  # plt.show()
  ##
  return [], x.reshape(-1), prob, np.zeros(prob.shape), pred

def fit_pareto_MLE(D, C, nodes, *args):
  import stan
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set_theme()
  CC = C.copy()
  CC[:nodes, :nodes] = CC[:nodes, :nodes] + CC[:nodes, :nodes].T
  # CC = CC[:nodes, :nodes]
  CC = adj2df(CC)
  CC = CC.loc[CC.source > CC.target]
  zeros = CC.weight == 0
  # DD = D[:nodes, :nodes]
  DD = D[:, :nodes]
  DD = adj2df(DD)
  DD = DD.loc[DD.source > DD.target]
  DD = DD.weight.loc[~zeros].to_numpy().ravel()
  CC = CC.weight.loc[~zeros].to_numpy().ravel()
  order = np.argsort(DD)
  DD = DD[order]
  CC = CC[order]
  # Statistical inference ----
  xmin = np.min(DD)
  a = np.log(DD / xmin)
  a = np.sum(CC) / np.sum(CC * a)
  print("\na:\t", a)
  pred = wrap_fit_pareto(a, xmin)
  print("llhood", pred.log_likelihood(DD, CC))
  x = DD.copy().reshape(-1, 1)
  log_prob = pred.predict(x)
  ## Error ---
  emp_log_prob = np.log(CC / np.sum(CC))
  print("error", np.std(emp_log_prob - log_prob))
  ###
  # cum_pareto = lambda a, xm, x:  1 - np.power(xm / x, a)
  # CC = np.array([CC[0]]+[CC[i] + np.sum(CC[:i]) for i in np.arange(1, CC.shape[0])])
  # CC = CC / np.sum(C)
  # fig, ax = plt.subplots(1, 1)
  # data = pd.DataFrame(
  #   {
  #     "x" : np.hstack([DD, x.ravel()]),
  #     "CCDF" : np.hstack([1 - CC, 1 - cum_pareto(a, xmin, x.ravel())]),
  #     "model" : ["data"] * CC.shape[0] + ["PARETO_MLE"] * x.shape[0] 
  #   }
  # )
  # sns.lineplot(
  #   data=data,
  #   x="x",
  #   y="CCDF",
  #   hue="model",
  #   ax=ax
  # )
  # ax.set_title(r"$\alpha$:" + f" {a:.4f}\t" + r"$x_{min}$: " + f"{xmin:.2f}")
  # ax.set_yscale("log")
  # fig.tight_layout()
  # plt.show()
  ##
  return [], x.reshape(-1), log_prob, np.zeros(log_prob.shape), pred

def fit_pareto_trunc_MLE(D, C, nodes, *args):
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy.optimize import fsolve
  sns.set_theme()
  CC = C.copy()
  CC[:nodes, :nodes] = CC[:nodes, :nodes] + CC[:nodes, :nodes].T
  # CC = CC[:nodes, :nodes]
  CC = adj2df(CC)
  CC = CC.loc[CC.source > CC.target]
  zeros = CC.weight == 0
  # DD = D[:nodes, :nodes]
  DD = D[:, :nodes]
  DD = adj2df(DD)
  DD = DD.loc[DD.source > DD.target]
  DD = DD.weight.loc[~zeros].to_numpy().ravel()
  CC = CC.weight.loc[~zeros].to_numpy().ravel()
  order = np.argsort(DD)
  DD = DD[order]
  CC = CC[order]
  # Statistical inference ----
  def func(x, *args):
    return (1 / x) + np.log(args[0]) + x * np.power(args[0] / args[1], x - 1) / (1 - np.power(args[0] / args[1], x)) - args[2]
  xmin = np.min(D[D > 0])
  xmax = np.max(D)
  a = fsolve(func, x0=0.5, args=(xmin, xmax, np.sum(CC * np.log(DD)) / np.sum(CC)))[0]
  print("\na:\t", a)
  pred = wrap_fit_pareto_trunc(a, xmin, xmax)
  print("llhood", pred.log_likelihood(DD, CC))
  x = DD.copy().reshape(-1, 1)
  log_prob = pred.predict(x)
  ## Error ----
  emp_log_prob = np.log(CC / np.sum(CC))
  print("error", np.std(emp_log_prob - log_prob))
  ###
  # cum_pareto = lambda a, xm, x:  1 - np.power(xm / x, a)
  # CC = np.array([CC[0]]+[CC[i] + np.sum(CC[:i]) for i in np.arange(1, CC.shape[0])])
  # CC = CC / np.sum(C)
  # fig, ax = plt.subplots(1, 1)
  # data = pd.DataFrame(
  #   {
  #     "x" : np.hstack([DD, x.ravel()]),
  #     "CCDF" : np.hstack([1 - CC, 1 - cum_pareto(a, xmin, x.ravel())]),
  #     "model" : ["data"] * CC.shape[0] + ["PARETO_TRUNC_MLE"] * x.shape[0] 
  #   }
  # )
  # sns.lineplot(
  #   data=data,
  #   x="x",
  #   y="CCDF",
  #   hue="model",
  #   ax=ax
  # )
  # ax.set_title(r"$\alpha$:" + f" {a:.4f}\t" + r"$x_{min}$: " + f"{xmin:.2f}\t" + r"$x_{max}$: " + f"{xmax:.2f}")
  # ax.set_yscale("log")
  # fig.tight_layout()
  # plt.show()
  ##
  return [], x.reshape(-1), log_prob, np.zeros(log_prob.shape), pred
  
def fit_exp_MLE(D, C, *args, npoints=100, **kwargs):
  import matplotlib.pyplot as plt
  import seaborn as sns

  nodes = C.shape[1]
  CC = C.copy().ravel()

  zeros = CC == 0
  DD = D[:, :nodes].ravel()

  DD = DD[~zeros]
  CC = CC[~zeros]

  ## Statistical inference ----
  from scipy.stats import expon
  N = np.sum(CC)
  N = np.ceil(N).astype(int)
  data = np.zeros(N) 

  e = 0
  for i, c in enumerate(CC):
    data[e:(e+int(c))] = DD[i]
    e += int(c)
  data = data[data > 0]

  Y, X, _ = plt.hist(data, bins=args[0], density=True)

  # print(X)
  # g = data[(data >= X[1]) & (data <= X[-2])]
  # print(np.min(g), np.max(g))
  # print(np.min(data[data>0]))

  lb = expon.fit(data[(data >= X[1]) & (data <= X[-1])])
  # lb = expon.fit(data)

  # import statsmodels.api as sm
  # Y = np.log(Y)
  # X = (X[1:] + X[:-1]) / 2
  # X = sm.add_constant(X)
  # r = sm.OLS(Y, X).fit()
  # print(r.params)
  # print(X)

  print(f"\nloc:\t", lb[0])
  print("lambda:\t", 1/lb[1])

  x = DD.copy().reshape(-1, 1)
  pred = wrap_fit_exp(1/lb[1], loc=0)
  print("llhood", pred.log_likelihood(DD, CC))
  log_prob = pred.predict(x)

  ## Error ----
  emp_log_prob = np.log(CC / np.sum(CC))
  print("error", np.std(emp_log_prob - log_prob))

  return [], x.reshape(-1), log_prob, np.zeros(log_prob.shape), pred

def fit_exp_trunc_MLE(D, C, *args, npoints=100, **kwargs):
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy.optimize import fsolve
  # def func(x, *args):
  #   return (1/x) + (args[0] * np.exp(-x * args[0])- args[1] * np.exp(-x * args[1])) / (np.exp(-x * args[0]) - np.exp(-x * args[1])) - args[2]
  
  nodes = C.shape[1]
  CC = C.copy().ravel()
  # CC[:nodes, :nodes] = CC[:nodes, :nodes] + CC[:nodes, :nodes].T
  # # CC = CC[:nodes, :nodes]
  # CC = adj2df(CC)
  # CC = CC.loc[CC.source > CC.target]
  zeros = CC == 0
  # DD = D[:nodes, :nodes]
  DD = D[:, :nodes].ravel()
  # DD = adj2df(DD)
  # DD = DD.loc[DD.source > DD.target]
  # DD = DD.weight.loc[~zeros].to_numpy().ravel()
  # CC = CC.weight.loc[~zeros].to_numpy().ravel()
  # order = np.argsort(DD)
  # DD = DD[order]
  # CC = CC[order]

  DD = DD[~zeros]
  CC = CC[~zeros]

  ## Statistical inference ----
  xmin = np.min(D[D>0])
  xmax = np.max(D)


  from scipy.stats import truncexpon

  N = np.sum(CC)
  N = np.ceil(N).astype(int)
  data = np.zeros(N) 

  e = 0
  for i, c in enumerate(CC):
    data[e:(e+int(c))] = DD[i]
    e += int(c)
  data = data[data > 0]

  _, X, _ = plt.hist(data, bins=args[0], density=True)

  if args[0] == 12:
    subdata = data[(data >= X[1]) & (data <= X[-1])]
  elif args[0] == 20:
    subdata = data[(data >= X[2]) & (data <= X[-3])]
  # subdata2 = subdata - np.min(subdata)

  loc = np.min(subdata)
  scale = np.std(subdata)
  b = np.max(subdata)

  lb = truncexpon.fit(subdata, b, loc=loc, scale=scale)

  
  print(lb)
  print("lambda:", 1/(lb[2]), "\n")
  # raise ValueError("")
 
  x = DD.copy().reshape(-1, 1)
  pred = wrap_fit_exp_trunc(1/lb[2], xmin, xmax)
  print("llhood", pred.log_likelihood(DD, CC))
  log_prob = pred.predict(x)
   ## Error ---
  emp_log_prob = np.log(CC / np.sum(CC))
  print("error", np.std(emp_log_prob - log_prob))
  ###
  # CC = np.array([CC[0]]+[CC[i] + np.sum(CC[:i]) for i in np.arange(1, CC.shape[0])])
  # CC = CC / np.sum(C)
  # cum_exp = lambda lb, x: 1 - np.exp(-lb * x)
  # fig, ax = plt.subplots(1, 1)
  # data = pd.DataFrame(
  #   {
  #     "x" : np.hstack([DD, x.ravel()]),
  #     "CCDF" : np.hstack([1 - CC, 1 - cum_exp(lb_mle, x.ravel())]),
  #     "model" : ["data"] * CC.shape[0] + ["EXP_TRUNC_MLE"] * x.shape[0] 
  #   }
  # )
  # sns.lineplot(
  #   data=data,
  #   x="x",
  #   y="CCDF",
  #   hue="model",
  #   ax=ax
  # )
  # ax.set_title(r"$\lambda$:" + f" {lb_mle:.4f}")
  # ax.set_yscale("log")
  # fig.tight_layout()
  # plt.show()
  ##
  return lb, subdata, truncexpon, np.zeros(log_prob.shape), pred

def fit_exp_STAN(D, C, nodes, bins, npoints=100, **kwargs):
  import matplotlib.pyplot as plt
  import seaborn as sns
  import stan
  CC = C.copy()
  CC[:nodes, :nodes] = CC[:nodes, :nodes] + CC[:nodes, :nodes].T
  # CC = CC[:nodes, :nodes]
  CC = adj2df(CC)
  CC = CC.loc[CC.source > CC.target]
  zeros = CC.weight == 0
  # DD = DD[:nodes, :nodes]
  DD = D[:, :nodes]
  DD = adj2df(DD)
  DD = DD.loc[DD.source > DD.target]
  DD = DD.weight.loc[~zeros].to_numpy().ravel()
  CC = CC.weight.loc[~zeros].to_numpy().ravel()
  order = np.argsort(DD)
  DD = DD[order]
  CC = CC[order]
  CC = np.array([CC[0]]+[CC[i] + np.sum(CC[:i]) for i in np.arange(1, CC.shape[0])])
  CC = CC / np.sum(CC)
  ## fit exponantial distribution ----
  # Create data ----
  data = {
    "N" : DD.shape[0],
    "x" : DD,
    "y" : CC
  }
  stan_exp_fit = """ 
  data {
    int N;
    vector[N] x;
    vector[N] y;
  }

  parameters {
    real<lower=0, upper=10> lambda;
    real<lower=0> sig;
  }

  model {
  lambda ~ inv_gamma(1, 1);
  y ~ normal(1 - exp(-lambda * x), sig);
  }

  """
  posterior = stan.build(stan_exp_fit, data=data)
  fit = posterior.sample(num_chains=4, num_warmup=1000, num_samples=1000).to_frame()
  lb = fit["lambda"].mean()
  sig = fit["lambda"].std()
  print("\nlambda:\t",lb, "\t", sig)
  x = np.linspace(np.min(D[D > 0]), np.max(D), npoints).reshape(-1, 1)
  pred = wrap_fit_exp(lb)
  prob = pred.predict(x)
  ###
  # cum_exp = lambda lb, x: 1 - np.exp(-lb * x)
  # fig, ax = plt.subplots(1, 1)
  # data = pd.DataFrame(
  #   {
  #     "x" : np.hstack([DD, x.ravel()]),
  #     "CCDF" : np.hstack([1 - CC, 1 - cum_exp(lb, x.ravel())]),
  #     "model" : ["data"] * CC.shape[0] + ["EXP_STAN"] * x.shape[0] 
  #   }
  # )
  # sns.lineplot(
  #   data=data,
  #   x="x",
  #   y="CCDF",
  #   hue="model",
  #   ax=ax
  # )
  # ax.set_title(r"$\lambda$:" + f" {lb:.4f} "+r"$\pm$ " + r"$\sigma_{\lambda}$:" + f" {sig:.5f}")
  # ax.set_yscale("log")
  # fig.tight_layout()
  # plt.show()
  ##
  return [], x.reshape(-1), prob, np.zeros(prob.shape), pred

def fit_linear_trunc(D, C, nodes, *args, npoints=100, **kwargs):
  import matplotlib.pyplot as plt
  import seaborn as sns
  import statsmodels.api as sm
  _, x, y = range_and_probs_from_DC(D, C, nodes, args[0])
  ## Statistical inference ----
  X = x.reshape(-1, 1)
  X = sm.add_constant(X)  
  lb = sm.OLS(y, X).fit()
  c = lb.params[0]
  lb = -lb.params[1]
  xmin = np.min(D[D>0])
  xmax = np.max(D)
  print("\nlambda:\t", lb, "\tb:\t", c)
  pred = wrap_fit_exp_trunc(lb, xmin, xmax)
  CC = C.copy()
  CC[:nodes, :nodes] = CC[:nodes, :nodes] + CC[:nodes, :nodes].T
  # CC = CC[:nodes, :nodes]
  CC = adj2df(CC)
  CC = CC.loc[CC.source > CC.target]
  zeros = CC.weight == 0
  # DD = D[:nodes, :nodes]
  DD = D[:, :nodes]
  DD = adj2df(DD)
  DD = DD.loc[DD.source > DD.target]
  DD = DD.weight.loc[~zeros].to_numpy().ravel()
  CC = CC.weight.loc[~zeros].to_numpy().ravel()
  order = np.argsort(DD)
  DD = DD[order]
  CC = CC[order]
  print("llhood", pred.log_likelihood(DD, CC))
  x = np.linspace(np.min(D[D>0]), np.max(D), npoints).reshape(-1, 1)
  prob = pred.predict(x)
  return [], x.reshape(-1), prob, np.zeros(prob.shape), pred

def fit_linear(D, C, nodes, *args, npoints=100, **kwargs):
  import matplotlib.pyplot as plt
  import seaborn as sns
  import statsmodels.api as sm
  # _, x, y = range_and_probs_from_DC(D, C, nodes, args[0])
  nodes = C.shape[1]
  N = np.sum(C)
  x = np.zeros(np.ceil(N).astype(int))
  e = 0
  for i in np.arange(C.shape[0]):
     for j in np.arange(C.shape[1]):
        if i == j or C[i,j] == 0: continue
        x[e:(e+int(C[i,j]))] = D[i, j]
        e += int(C[i,j])
  x = x[x > 0]
  x = np.array(x)

  Y, X, _ = plt.hist(x, bins=args[0], density=True)
  Y = np.log(Y)
  X = (X[1:] + X[:-1]) / 2
  Y = Y[3:-3]
  X = X[3:-3]

  ## Statistical inference ----
  X = x.reshape(-1, 1)
  X = sm.add_constant(X)  
  lb = sm.OLS(Y, X).fit()
  c = lb.params[0]
  lb = -lb.params[1]
  print("\nlambda:\t", lb, "\tb:\t", c)
  pred = wrap_fit_exp(lb)

  CC = C.copy()
  CC[:nodes, :nodes] = CC[:nodes, :nodes] + CC[:nodes, :nodes].T
  # CC = CC[:nodes, :nodes]
  CC = adj2df(CC)
  CC = CC.loc[CC.source > CC.target]
  zeros = CC.weight == 0
  # DD = D[:nodes, :nodes]
  DD = D[:, :nodes]
  DD = adj2df(DD)
  DD = DD.loc[DD.source > DD.target]
  DD = DD.weight.loc[~zeros].to_numpy().ravel()
  CC = CC.weight.loc[~zeros].to_numpy().ravel()
  order = np.argsort(DD)
  DD = DD[order]
  CC = CC[order]
  print("llhood", pred.log_likelihood(DD, CC))
  x = np.linspace(np.min(D[D>0]), np.max(D), npoints).reshape(-1, 1)
  prob = pred.predict(x)
  return [], x.reshape(-1), prob, np.zeros(prob.shape), pred

fitters = {
  "EXPMLE" : fit_exp_MLE,
  "EXPTRUNC" : fit_exp_trunc_MLE,
  "PARETO" : fit_pareto_MLE,
  "PARETOTRUNC" : fit_pareto_trunc_MLE,
  "LINEAR" : fit_linear,
  "LINEARTRUNC" : fit_linear_trunc
}