import numpy as np
from pathlib import Path
from various.network_tools import *

class EDR:
  def __init__(
    self, nodes : int, lb=0.19, **kwargs
  ) -> None:
    # Directory details ----
    self.folder = "RAN"
    # Folders ----
    self.subject = "MAC"
    if "version" in kwargs.keys():
      self.version = str(kwargs["version"])
    else: self.version = str(220617)
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = "FLN"
    if "nature" in kwargs.keys():
      self.nature = kwargs["nature"]
    else: self.nature = "original"
    if "distance" in kwargs.keys():
      self.distance = kwargs["distance"]
    else: self.distance = "tracto16"
    if "model" in kwargs.keys():
      self.model = kwargs["model"]
    else: self.model = ""
    if "inj" in kwargs.keys():
      self.inj = str(kwargs["inj"])
    else: self.inj = str(49)
    if "dir" in kwargs.keys():
      self.dir = kwargs["dir"]
    else: self.dir = ""
    if "topology" in kwargs.keys():
      self.topology = kwargs["topology"]
    else: self.topology = ""
    if "b" in kwargs.keys():
      self.b = kwargs["b"]
    else: self.b = ""
    # Constants ----
    self.lb = lb #[mm^-1]
    self.Area = 10430 # [mm^2]
    self.rho =  0.59 # network density
    self.counter = int(8993482 * 100 / 49)
    # Parameters ----
    self.nodes = nodes
    ## Define mu-score parameters ----
    self.Alpha = np.array([6])
    beta1 = np.linspace(0.01, 0.2, 4)
    beta2 = np.linspace(0.2, 0.4, 2)[1:]
    self.Beta = np.hstack((beta1, beta2))

  def set_alpha(self, alpha):
    self.Alpha = alpha
  
  def set_beta(self, beta):
    self.Beta = beta

  def den(self, A):
    m = np.sum(A[:self.nodes, :self.nodes] > 0)
    return m / (self.nodes * (self.nodes - 1))
  
  def Den(self, A):
    n = A.shape[0]
    m = np.sum(A > 0)
    return m / (n * (n - 1))

  def count(self, A):
    return np.sum(A)

  def count_M(self, A):
    return np.sum(A[:, :self.nodes])

  def links_M(self, A):
    return np.sum(A[:, :self.nodes] > 0)

  def links(self, A):
    return np.sum(A[:self.nodes, :self.nodes] > 0)

  def create_directory(self, path : str):
     Path(path).mkdir(exist_ok=True, parents=True)

