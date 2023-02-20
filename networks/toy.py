# Standard libs ----
from pathlib import Path
from os.path import join
import numpy as np
# Personalized libs ----
from networks.base import BASE

class TOY(BASE):
  """
    Class to process the network in the format needed for the Hierarchy class to work.
    Hierarchy was created to use specifically in the macaque FLN network.
    In the future, there will be another class to run the link community algorithm
    in particular netowrks. For the moment, the network has to be directed.
    
    Parameters:
        A: An NxM directed adjacency matrix.
          The algorithm can compute similarities using
          the whole data, but only classsify links from the edge-comple graph.

        linkage: string.
          Desired linkage-method for the hierarchical agglomeration algorithm.
          Currenly, only working with the single and average linkage method.

        nlog10: bool.
          
  """
  def __init__(
    self, A, linkage, nlog10=False, lookup=False, cut=False,
    mapping="trivial", index="jacp", topolgy="MIX",
    **kwargs
  ) -> None:
    super().__init__(linkage, **kwargs)
    self.nlog10 = nlog10
    self.lookup = lookup
    self.cut = cut
    self.topology = topolgy
    self.mapping = mapping
    self.index = index
    self.A = A.copy()
    self.nodes, self.rows = A.shape
    # Set ANALYSIS NAME ----
    self.analysis = linkage.upper() + "_{}_{}".format(
      self.rows, self.nodes
    )
    if nlog10:
      self.analysis = self.analysis + "_l10"
    if lookup:
      self.analysis = self.analysis + "_lup"
    if cut:
      self.analysis = self.analysis + "_cut"
    if "EC" in kwargs.keys():
      if kwargs["EC"]:
       self.analysis = self.analysis + "_EC" 
    # Create paths ----
    if "plot_path" in kwargs.keys():
      self.plot_path = kwargs["plot_path"]
    else:
      self.plot_path = join(
        "../plots", "TOY", self.common_path,
        self.analysis, self.mode, self.topology
      )
    if "pickle_path" in kwargs.keys():
      self.pickle_path = kwargs["pickle_path"]
    else:
      self.pickle_path = join(
        "../pickle", "TOY", self.common_path,
        self.analysis, self.mode, self.topology
      )
    # Overlap ----
    self.overlap = np.array(["UNKNOWN"] * self.nodes)
    # NOCS dict ----
    self.data_nocs = {}

  def create_plot_directory(self):
    Path(
      self.plot_path
    ).mkdir(exist_ok=True, parents=True)
    
  def create_pickle_directory(self):
    Path(
      self.pickle_path
    ).mkdir(exist_ok=True, parents=True)

  def set_labels(self, labels):
    self.labels = labels
    self.struct_labels = labels