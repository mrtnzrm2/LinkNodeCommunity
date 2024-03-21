import numpy as np

def inverted_mapping(A, nlog10, lookup, prob, *args, b=0, **kwargs):
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
  ## original shift ----
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
    lookup = np.nanmax(R[R != 0]) + b
    # lookup = 14.127
    R[R == 0] = lookup
  elif lookup and ~nlog10:
    lookup = np.nanmin(R[R != 0])
    R[R == 0] = lookup
  else:
    lookup = 0
  return R, lookup, shift

def entropy_mapping(A, nlog10, lookup, prob, *args, b=0, **kwargs):
  R = A.copy()
  np.fill_diagonal(R, np.nan)
  # Prepare data for taking the log ----
  ## If it is a probability ----
  if not prob: raise ValueError("Data has to be probabilities")
  if np.sum(R > 1) > 0 or np.sum(R < 0) > 0:
    raise ValueError("Probabilities outside the range [0, 1]")
  ## If there are 1 entries ----
  if np.sum(R > 1) > 0 and nlog10:
    raise ValueError("Probabilities more than one are not accepted")
  R[R!=0] = -R[R!=0] * np.log(R[R!=0])
  return R, 0, 0

def desperate_mapping(A, nlog10, lookup, prob, *args, b=0, **kwargs):
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
  R = R + 1
  return R, 1, shift

def normal_mapping(A, nlog10, lookup, prob, *args, b=0, **kwargs):
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

def exp_count_mapping(A, *args, **kwargs):
  AA = A.copy().astype(float)
  np.fill_diagonal(AA, np.nan)
  AA = np.log(1 + AA)
  return AA, 0, 0

def exp_count_mapping_2(A, *args, **kwargs):
  AA = A.copy().astype(float)
  np.fill_diagonal(AA, np.nan)
  AA = 1 + np.log(1 + AA)
  return AA, 0, 0

def cum_exp_mapping(A, *args, **kwargs):
  AA = A.copy().astype(float)
  np.fill_diagonal(AA, np.nan)
  AA = - np.log(1 - AA) / 0.07921125
  return AA, 0, 0

def trivial_mapping(A, *args, **kwargs):
  AA = A.copy().astype(float)
  np.fill_diagonal(AA, np.nan)
  return AA, 0, 0

def bin_mapping(A, *args, **kwargs):
  AA = A.copy().astype(float)
  AA[AA > 0] = 1.
  np.fill_diagonal(AA, np.nan)
  return AA, 0, 0

def signed_trivial_mapping(A, *args, **kwargs):
  n , m = A.shape
  AA = np.zeros((2, n, m))
  AA[0][A > 0] = A[A > 0]
  AA[1][A < 0] = -A[A < 0]
  
  BB = np.zeros((n, m, 2))
  BB[:, :, 0] = AA[0]
  BB[:, :, 1] = AA[1]
  
  np.fill_diagonal(BB[:, :, 0], np.nan)
  np.fill_diagonal(BB[:, :, 1], np.nan)
  return BB, 0, 0

def multiplexed_trivial_mapping(A, *args, **kwargs):
  n , m = args[0].shape
  
  BB = np.zeros((n, m, len(args)))
  for i in np.arange(len(args)):
    BB[:, :, i] = args[i]
    BB[:, :, i] = args[i]
    np.fill_diagonal(BB[:, :, i], np.nan)

  return BB, 0, 0

def multiplexed_colnormalized_mapping(A, *args, **kwargs):
  n , m = args[0].shape
  
  BB = np.zeros((n, m, len(args)))
  for i in np.arange(len(args)):
    BB[:, :, i] = args[i]
    BB[:, :, i] = args[i]
    np.fill_diagonal(BB[:, :, i], np.nan)
  
  for i in np.arange(BB.shape[1]):
    N = np.nansum(BB[:, i, :])
    BB[:, i, :] /= N

  return BB, 0, 0

def column_normalized_mapping(A, *args, **kwargs):
  AA = A / np.sum(A, axis=0)
  np.fill_diagonal(AA, np.nan)
  return AA, 0, 0

maps = {
  "R1" : inverted_mapping,
  "R2" : normal_mapping,
  "R3" : entropy_mapping,
  "R4" : exp_count_mapping,
  "R5" : exp_count_mapping_2,
  "trivial" : trivial_mapping,
  "bin" : bin_mapping,
  "signed_trivial" : signed_trivial_mapping,
  "multiplexed_trivial" : multiplexed_trivial_mapping,
  "multiplexed_colnormalized" : multiplexed_colnormalized_mapping,
  "cum_exp" : cum_exp_mapping,
  "desp" : desperate_mapping,
  "column_norm" : column_normalized_mapping
}
