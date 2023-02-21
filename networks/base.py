# Standard libs ----
import numpy as np
# Personal libs ----
from various.network_tools import *

class BASE:
  def __init__(self, linkage, **kwargs) -> None:
    # Set general attributes ----
    self.linkage = linkage
    if "version" in kwargs.keys():
      self.version = str(kwargs["version"])
    if "mode" in kwargs.keys():
      self.mode = kwargs["mode"]
    else: self.mode = "ALPHA"
    if "topology" in kwargs.keys():
      self.topology = kwargs["topology"]
    else: self.topology = ""
    ### mu parameters ----
    self.Alpha = np.array([6])
    beta1 = np.linspace(0.01, 0.2, 4)
    beta2 = np.linspace(0.2, 0.4, 2)[1:]
    self.Beta = np.hstack((beta1, beta2))
    # Save methos from net_tool ----
    self.column_normalize = column_normalize
    self.save_class = save_class
    self.read_class = read_class
    # Create paths ----
    self.common_path = self.version

  def set_alpha(self, alpha):
    self.Alpha = alpha

  def set_beta(self, beta):
    self.Beta = beta