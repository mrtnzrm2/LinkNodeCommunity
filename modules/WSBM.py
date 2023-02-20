import matlab
import matlab.engine
from os import getcwd
from os.path import join
from pathlib import Path
import numpy as np

class WSBM:
  def __init__(self, NET, H) -> None:
    # WSBM parameters ----
    self.alpha = 0
    self.W_Distr = "Normal"
    self.E_Distr = "Bernoulli"
    self.verbosity = 1
    self.numTrials = 50
    # Copy attributes from H ----
    self.nodes = H.nodes
    self.dA = H.dA.copy().to_numpy()
    self.H = H.H
    ## Transform dA ----
    self.dA[:, :2] += 1
    # Arrange wsbm path ----
    self.python_path = getcwd()
    self.wsbm_path = "../WSBM"
    # Copy methods from NET ----
    self.K = NET.K
    self.R = NET.R
    self.common_path = NET.common_path
    self.save_class = NET.save_class
    self.read_class = NET.read_class
    # Create paths ----
    Path(
      join(
        self.wsbm_path, "PYTHON",
        self.common_path
      )
    ).mkdir(exist_ok=True, parents=True)

  def save_structure(self, R):
    from scipy.io import savemat
    A = {
      "E" : self.dAid,
      "R" : float(R),
      "W_Distr" : self.W_Distr,
      "E_Distr" : self.E_Distr,
      "alpha" : float(self.alpha),
      "numTrials" : float(self.numTrials)
    }
    savemat(
      join(
        self.wsbm_path, "PYTHON",
        self.common_path, "structure.mat"
      ),
      A
    )

  def set_K(self, K):
    from scipy.cluster.hierarchy import cut_tree
    self.dAid = self.dA.copy()
    self.dAid[:, 2] = cut_tree(
      self.H,
      n_clusters=K
    ).reshape(-1)
  
  def wsbm(self):
    # Create data ----
    from pandas import DataFrame, concat
    self.labels = DataFrame()
    self.logev = []
    # Start matlab engine ----
    eng = matlab.engine.start_matlab()
    eng.addpath(self.wsbm_path)
    for k in self.K:
      for r in self.R:
        print("Start WSBM with K: {}, R: {}".format(k, r))
        self.set_K(k)
        # Save data to matlab file ----
        self.save_structure(r)
        # Run wsbm_py ----
        labels, lev = eng.wsbm_py(
          join(
            "PYTHON",
            self.common_path, "structure.mat"
          ),
          nargout=2
        )
        self.labels = concat(
          [
            self.labels,
            DataFrame(
              {
                "K" : [k] * self.nodes,
                "R" : [r] * self.nodes,
                "labels" : np.asarray(labels).astype(int).reshape(-1) - 1
              }
            )
          ]
        )
        self.logev.append(lev)
    # Stop engine ----
    eng.quit()

  def pick_pair(self, K, R):
    return self.labels.loc[
      (self.labels["K"] == K) &
      (self.labels["R"] == R),
      "labels"
    ].to_numpy().reshape(-1).astype(int)
      
  def save(self, S, path):
    self.save_class(
      S, path, "WSBM_{}_{}".format(self.K, self.R)
    )
