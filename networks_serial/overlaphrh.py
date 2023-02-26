# Standard libs ----
import pandas as pd
# Personal libs ----
from networks_serial.scalehrh import SCALEHRH
from various.network_tools import *

class OVERLAPHRH(SCALEHRH):
  def __init__(self, linkage) -> None:
    super().__init__(linkage)
    self.data_overlap = pd.DataFrame()

  def set_overlap_scores(self, omega, acc1, acc2, **kwargs):
    if "score" in kwargs.keys():
      subdata = pd.DataFrame(
        {
          "omega" : [omega],
          "sensitivity" : [acc1],
          "specificity" : [acc2],
          "iter" : [self.iter],
          "c" : [kwargs["score"]]
        }
      )
    else:
      subdata = pd.DataFrame(
        {
          "omega" : [omega],
          "sensitivity" : [acc1],
          "specificity" : [acc2],
          "iter" : [self.iter],
          "c" : ["node communities"]
        }
      )
    self.data_overlap = pd.concat(
      [self.data_overlap, subdata], ignore_index=True
    )
  
  def set_nmi_nc_overlap(self, l1, l2, overlap, **kwargs):
    if "score" in kwargs.keys():
      #create subdata ----
      subdata = pd.DataFrame(
        {
          "NMI" : [AD_NMI_overlap(l1, l2, overlap)],
          "c" : [kwargs["score"]],
          "iter" : [self.iter]
        }
      )
    else:
      #create subdata ----
      subdata = pd.DataFrame(
        {
          "NMI" : [AD_NMI_overlap(l1, l2, overlap)],
          "c" : ["node communities"],
          "iter" : [self.iter]
        }
      )
    # Merge with data ----
    self.data = pd.concat(
      [self.data, subdata], ignore_index=True
    )