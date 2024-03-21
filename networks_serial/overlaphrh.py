# Standard libs ----
import pandas as pd
# Personal libs ----
from networks_serial.scalehrh import SCALEHRH
from various.network_tools import *

class OVERLAPHRH(SCALEHRH):
  def __init__(self, linkage) -> None:
    super().__init__(linkage)
    self.data_overlap = pd.DataFrame()

  def set_overlap_scores(self, omega, acc1, acc2, direction="both", **kwargs):
    if "score" in kwargs.keys():
      subdata = pd.DataFrame(
        {
          "omega" : [omega],
          "sensitivity" : [acc1],
          "specificity" : [acc2],
          "iter" : [self.iter],
          "c" : [kwargs["score"]],
          "direction" : [direction]
        }
      )
    else:
      subdata = pd.DataFrame(
        {
          "omega" : [omega],
          "sensitivity" : [acc1],
          "specificity" : [acc2],
          "iter" : [self.iter],
          "c" : ["node communities"],
          "direction" : [direction]
        }
      )
    self.data_overlap = pd.concat(
      [self.data_overlap, subdata], ignore_index=True
    )
  
  def set_nmi_nc_overlap(self, l1, l2, omega, direction="both", **kwargs):
    if "score" in kwargs.keys():
      #create subdata ----
      subdata = pd.DataFrame(
        {
          "sim" : ["NMI", "OMEGA"],
          "values" : [
            AD_NMI_overlap(l1, l2), omega
          ],
          "c" : [kwargs["score"]] * 2,
          "iter" : [self.iter] * 2,
          "direction" : [direction] * 2
        }
      )
    else:
      #create subdata ----
      subdata = pd.DataFrame(
        {
           "sim" : ["NMI", "OMEGA"],
          "values" : [
            AD_NMI_overlap(l1, l2), omega,
          ],
          "iter" : [self.iter] * 2,
          "c" : ["node communities"] * 2,
          "direction" : [direction] * 2
        }
      )
    # Merge with data ----
    self.data = pd.concat(
      [self.data, subdata], ignore_index=True
    )