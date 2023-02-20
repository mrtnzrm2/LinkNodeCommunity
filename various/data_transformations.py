import numpy as np

def inverted_mapping(A, nlog10, lookup, prob, *args, **kwargs):
  R = A.copy()
  np.fill_diagonal(R, np.nan)
  # Prepare data for taking the log ----
  ## If it is a probability ----
  if prob and (np.sum(R > 1) > 0 or np.sum(R < 0) > 0):
    raise ValueError("Probabilities outside the range [0, 1]")
  ## If there are 1 entries ----
  if prob and np.sum(R == 1) > 0 and nlog10:
    print("\n   Values of 1 detected when using the\n prob=T and nlog10=T settings.")
    print("     To not set them to zero when taking the log,")
    print("     They will ve replaced by the square root of the biggest prob")
    print("     less than 1.\n")
    R[R == 1] = np.sqrt(np.nanmax(R[R < 1]))
  ## originial shift ----
  shift = 0
  # If original data is prob and a  log transformation is needed ----
  if nlog10 and prob:
    R[R != 0] = -np.log(R[R != 0])
  # If original data is not a prob and a  log transformation is needed ----
  elif nlog10 and ~prob:
    if np.sum(R[R < 0]) > 0:
      print("\n     For R: If nlog10=T, prob=F, data will be shifted to make")
      print("       the minum weight=1.01\n")
      shift = np.abs(np.nanmin(R[R != 0])) + 1.01
      R[R != 0] = R[R != 0] + shift
    elif np.sum(R[R <= 1]) > 0 and np.sum(R[R < 0] == 0):
      print("\n     For R: If nlog10=T, prob=F, data will be shifted to make")
      print("       the minum weight=1.01\n")
      shift = (1 - np.nanmin(R[R != 0])) + 0.01
      R[R != 0] = R[R != 0] + shift
    R[R > 0] = np.log(R[R > 0])
  # If non-connetions will get a weight ----
  if lookup and nlog10:
    lookup = np.nanmax(R[R != 0]) * 1.01
    # lookup = 14.127
    R[R == 0] = lookup
  elif lookup and ~nlog10:
    lookup = np.nanmin(R[R != 0])
    R[R == 0] = lookup
  else:
    lookup = 0
  return R, lookup, shift

def normal_mapping(A, nlog10, lookup, prob, *args, b=1e-5, **kwargs):
  # log transformation ----
  R = A.copy()
  np.fill_diagonal(R, np.nan)
  # Prepare data for taking the log ----
  ## If it is a probability ----
  if prob and (np.nansum(R > 1) > 0 or np.nansum(R < 0) > 0):
    raise ValueError("Probabilities outside the range [0, 1]")
  ## If there are 1 entries ----
  if prob and np.nansum(R == 1) > 0 and nlog10:
    print("\n   Values of 1 detected when using the\n prob=T and nlog10=T settings.")
    print("     To not set them to zero when taking the log,")
    print("     They will ve replaced by the square root of the biggest prob")
    print("     less than 1.\n")
    R[R == 1] = np.sqrt(np.nanmax(R[R < 1]))
  ## originial shift ----
  shift = 0
  # If original data is prob and a  log transformation is needed ----
  if nlog10 and prob:
    R[R != 0] = np.log10(R[R != 0])
    shift = np.abs(np.nanmin(R[R != 0])) + b
    R[R != 0] = R[R != 0] + shift
  # If original data is not a prob and a  log transformation is needed ----
  elif nlog10 and ~prob:
    if np.sum(R[R < 0]) > 0:
      print("\n     For R: If nlog10=T, prob=F, data will be shifted to make")
      print("       the minum weight=1.01\n")
      shift = np.abs(np.nanmin(R[R != 0])) + 1.01
      R[R != 0] = R[R != 0] + shift
    elif np.sum(R[R <= 1]) > 0 and np.sum(R[R < 0] == 0):
      print("\n     For R: If nlog10=T, prob=F, data will be shifted to make")
      print("       the minum weight=1.01\n")
      shift = (1 - np.nanmin(R[R != 0])) + 0.01
      R[R != 0] = R[R != 0] + shift
    R[R > 0] = np.log(R[R > 0])
  # If non-connetions will get a weight ----
  if lookup and nlog10:
    lookup = np.nanmin(R[R != 0]) / 1.01
    R[R == 0] = lookup
  elif lookup and ~nlog10:
    lookup = np.nanmin(R[R != 0])
    R[R == 0] = lookup
  else:
    lookup = 0
  return R, lookup, shift

def trivial_mapping(A, *args, **kwargs):
  AA = A.copy()
  np.fill_diagonal(AA, np.nan)
  return AA, 0, 0

maps = {
  "R1" : inverted_mapping,
  "R2" : normal_mapping,
  "trivial" : trivial_mapping
}
