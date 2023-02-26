import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from os.path import join, exists
from os import remove
import pickle as pk

def skim_partition(partition):
  par = partition.copy()
  from collections import Counter
  fq = Counter(par)
  for i in fq.keys():
    if fq[i] == 1: par[par == i] = -1
  new_partition = par
  ndc = np.unique(par[par != -1])
  for i, c in enumerate(ndc):
    new_partition[par == c] = i
  return new_partition

def Dc_id(dA, id):
  # Filter dataframe ----
  dAid = dA.loc[
    dA.id == id
  ]
  # Get source nodes ----
  src = dAid.source.to_numpy()
  src = set(src)
  # Get target nodes list ----
  tgt = dAid.target.to_numpy()
  tgt = set(tgt)
  # Get number of edges ----
  m = dAid.shape[0]
  # Compute Dc ----
  n = len(tgt.union(src))
  if n > 1 and m >= n: return (m - n + 1) / (n - 1) ** 2
  else: return 0

def minus_one_Dc(dA):
  ids = np.sort(
    np.unique(
      dA["id"].to_numpy()
    )
  )
  for id in ids:
    Dc = Dc_id(dA, id)
    if Dc <= 0:
      dA["id"].loc[dA["id"] == id] = -1

# def gp_fit(x, y, **kwargs):
#   from sklearn.gaussian_process import GaussianProcessRegressor
#   from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
#   kernel = RBF(2) + WhiteKernel()
#   gpr = GaussianProcessRegressor(
#     kernel=kernel, random_state=0
#   ).fit(x.reshape(-1, 1), y)
#   print("> Running gaussian regression")
#   return gpr

# def piecewise_poly_fit(x, y, th=50, **kwargs):
#   # Regression ----
#   from sklearn.pipeline import make_pipeline
#   from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#   from sklearn.linear_model import ARDRegression
#   ## Linear part ----
#   x_1 = x[x <= th]
#   y_1 = y[x <= th]
#   poly_1 = make_pipeline(
#       PolynomialFeatures(degree=1, include_bias=True),
#       StandardScaler(),
#       ARDRegression(),
#   ).fit(
#     x_1.reshape(-1, 1), y_1
#   )
#   ### Cubic part ----
#   x_2 = x[x > th]
#   y_2 = y[x > th]
#   poly_2 = make_pipeline(
#       PolynomialFeatures(degree=3, include_bias=True),
#       StandardScaler(),
#       ARDRegression(),
#   ).fit(
#     x_2.reshape(-1, 1), y_2
#   )
#   return poly_1, poly_2

# def poly_fit(x, y, deg=1, **kwargs):
#   # Regression ----
#   from sklearn.pipeline import make_pipeline
#   from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#   from sklearn.linear_model import ARDRegression
#   ard_poly = make_pipeline(
#       PolynomialFeatures(degree=deg, include_bias=True),
#       StandardScaler(),
#       ARDRegression(),
#   ).fit(
#     x.reshape(-1, 1), y
#   )
#   print("> Polynomial coefficients' means:")
#   print(ard_poly[2].coef_)
#   return ard_poly

def linear_fit(x, y):
  # Regression ----
  # x = (x - np.mean(x)) / np.std(x)
  from sklearn.linear_model import LinearRegression
  line_poly = LinearRegression(fit_intercept=True).fit(
    x.reshape(-1, 1), y
  )
  print("> Linear coefficients' means:")
  print(line_poly.coef_)
  return line_poly

def range_and_probs_from_DC(D, C, nodes, bins):
  D_ = D[:, :nodes]
  # Treat distances ----
  min_d = np.min(D_[D_ > 0])
  max_d = np.max(D_)
  d_range = np.linspace(min_d, max_d, bins)
  # Bin size - delta ----
  delta = (d_range[1] - d_range[0]) / 2
  counts = np.zeros(d_range.shape[0] - 1)
  # Sum counts ----
  for i in np.arange(C.shape[0]):
    for j in np.arange(C.shape[1]):
      d = D[i, j]
      for k in np.arange(d_range.shape[0] - 1):
        if d_range[k] <= d and d_range[k + 1] > d:
          counts[k] += C[i, j]
          break
  # Prepare x and y ----
  y = np.log(counts /(np.sum(counts) * 2 * delta))
  x = d_range[:-1] + delta
  return d_range, x, y

def predicted_D_frequency(D, C, nodes, bins, npoints=100, **kwargs):
  d_range, x, y = range_and_probs_from_DC(D, C, nodes, bins)
  x = x[y > -np.Inf]
  y = y[y > -np.Inf]
  ## Get prob ----
  pred = linear_fit(x, y)
  x = np.linspace(
    np.min(D[D > 0]),
    np.max(D),
    npoints
  ).reshape(-1, 1)
  prob = pred.predict(x)
  return d_range, x.reshape(-1), prob, np.zeros(prob.shape), pred

def get_best_kr(score, H):
  k = 1
  if score == "_maxmu":
    k = get_k_from_maxmu(
      get_H_from_BH(H)
    )
  elif score == "_D":
    k = get_k_from_D(
      get_H_from_BH(H)
    )
  elif score == "_X":
    k = get_k_from_X(
      get_H_from_BH(H), order=0
    )
  else: raise ValueError(f"Unexpected score: {score}")
  r = get_r_from_X_diag(
    k, H.H, H.Z, H.A, H.nodes
  )
  k = int(k)
  r = int(r)
  return k, r

def get_best_kr_equivalence(score, H):
  k = 1
  if score == "_maxmu":
    k = get_k_from_maxmu(
      get_H_from_BH(H)
    )
  elif score == "_D":
    k = get_k_from_D(
      get_H_from_BH(H)
    )
  elif score == "_X":
    k = get_k_from_X(
      get_H_from_BH(H), order=0
    )
  else: raise ValueError(f"Unexpected score: {score}")
  r = get_r_from_equivalence(k, H)
  k = int(k)
  return k, r

def aesthetic_ids_vector(v):
  vv = v.copy()
  ids = np.sort(np.unique(v))
  if -1 in ids:
    ids = ids[1:]
    aids = np.arange(len(ids))
  else:
    aids = np.arange(len(ids))
  for i, id in enumerate(ids):
    vv[v == id] = aids[i]
  return vv

def aesthetic_ids(dA):
    ids = np.sort(
      np.unique(
        dA["id"].to_numpy()
      )
    )
    if -1 in ids:
      ids = ids[1:]
      aids = np.arange(1, len(ids) + 1)
    else:
      aids = np.arange(1, len(ids) + 1)
    for i, id in enumerate(ids):
      dA.loc[dA["id"] == id, "id"] = aids[i].astype(str)
    dA["id"] = dA["id"].astype(int)

def combine_dics(f1, f2):
  for k in f2.keys():
    if k not in f1.keys():
      f1[k] = f2[k]
    else:
      f1[k] += f2[k]

def bar_data(dA, nodes, labels, norm=False):
  # unique ids ----
  Tids = np.unique(dA["id"])
  # Create data ----
  data = pd.DataFrame()
  # Start loop ----
  from collections import Counter
  for i in np.arange(nodes):
    ## Source ----
    ids_out = dA.id.loc[dA["source"] == i].to_numpy().ravel()
    fq = dict(Counter(ids_out))
    ## Target ----
    ids_in = dA.id.loc[dA["target"] == i].to_numpy().ravel()
    fq_in = dict(Counter(ids_in))
    # Combine dictionaries ----
    combine_dics(fq, fq_in)
    # Keys ----
    keys = list(fq.keys())
    if len(keys) == 0: continue
    # keys = np.array(keys) - 1
    keys = [
      np.where(Tids == id)[0][0] for id in keys
    ]
    # Values ----
    values = list(fq.values())
    values = np.array(values)
    # Sizes ----
    size_ids = np.zeros(len(Tids))
    size_ids[keys] = values
    if norm:
      size_ids /= np.sum(size_ids)
    data = pd.concat(
      [
        data,
        pd.DataFrame(
          {
            # 'node_index' : np.array([i] * len(Tids)), 
            "nodes" : np.array([labels[i]] * len(Tids)),
            "ids" : Tids,
            "size" : size_ids
          }
        )
      ],
      ignore_index=True
    )
  return data

def reverse_partition(Cr, labels):
  skm = skim_partition(Cr)
  s = np.unique(skm).astype(int)
  s = s[s != -1]
  k = {
    r : [] for r in s
  }
  for i, r in enumerate(skm):
    if r == -1: continue
    k[r].append(labels[i])
  return k

def nocs2parition(partition: dict, nocs: dict):
  for noc in nocs.keys():
    for cover in nocs[noc]:
      if noc not in partition[cover]: partition[cover].append(noc)

def get_H_from_BH(H):
  h = pd.DataFrame()
  for i in np.arange(len(H.BH)):
    h  = pd.concat(
      [h , H.BH[i]],
      ignore_index=True
    )
  return h

def get_H_from_BH_with_maxmu(H):
  h = pd.DataFrame()
  maxmu = []
  for i in np.arange(len(H.BH)):
    maxmu.append(H.BH[i]["mu"].to_numpy())
  maxmu = np.array(maxmu).T
  maxmu = np.nanmax(maxmu, axis=1)
  h  = pd.concat(
    [h , H.BH[0]],
    ignore_index=True
  )
  h["mu"] = maxmu
  return h

def get_k_from_ntrees(H):
  k = H["K"].loc[
    H["ntrees"] == 0
  ]
  if (len(k) > 1):
    print("warning: more than one k")
    k = np.max(k)
  return k

def get_k_from_mu(H):
  k = H["K"].loc[
    H["mu"] == np.nanmax(H["mu"])
  ]
  if (len(k) > 1):
    print("warning: more than one k")
    k = k.iloc[0]
  return k

def get_k_from_avmu(H):
  avH = H.groupby(["K"]).mean()
  k = avH.index[
    avH["mu"] == np.nanmax(avH["mu"])
  ].to_numpy().reshape(-1).astype(int)
  if len(k) > 0:
    k = k[0]
  return k

def get_k_from_maxmu(H):
  avH = H.groupby(["K", "alpha"]).max()
  avH = avH.groupby(["K"]).mean()
  k = avH.index[
    avH["mu"] == np.nanmax(avH["mu"])
  ].to_numpy().ravel().astype(int)
  if len(k) > 0:
    k = k[0]
  return k

def get_k_from_X(H, order=0):
  avH = H.groupby(["K", "alpha"]).max()
  avH = avH.groupby(["K"]).mean()
  target_maximum = np.sort(avH["X"])
  target_maximum = target_maximum[-1 - order]
  k = avH.index[
    avH["X"] == target_maximum
  ].to_numpy().reshape(-1).astype(int)
  if len(k) > 0:
    k = k[0]
  return k

def get_labels_from_Z(Z, r):
  save_Z = np.sum(Z, axis=1)
  if 0 in save_Z: return np.array([np.nan])
  from scipy.cluster.hierarchy import cut_tree
  labels = cut_tree(
    Z,
    n_clusters=r
  ).reshape(-1)
  return labels

def get_r_from_mu(H):
  r = H["NAC"].loc[
    H["mu"] == np.nanmax(H["mu"])
  ]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return r

def get_r_from_D(H):
  r = H["NAC"].loc[
    H["D"] == np.nanmax(H["D"])
  ]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return r

def get_k_from_D(H):
  r = H["K"].loc[
    H["D"] == np.nanmax(H["D"])
  ]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return r

def get_r_from_avmu(H):
  avH = H.groupby(["K"]).mean()
  r = avH["NAC"].loc[
    avH["mu"] == np.nanmax(avH["mu"])
  ].to_numpy().reshape(-1).astype(int)
  if (len(r) > 1):
    print("warning: more than one k")
    r = r[0]
  else:
    r = r[0]
  return r

def get_r_from_maxmu(H):
  avH = H.groupby(["K", "alpha"]).max()
  avH = avH.groupby(["K"]).mean()
  r = avH["NAC"].loc[
    avH["mu"] == np.nanmax(avH["mu"])
  ].to_numpy().reshape(-1).astype(int)
  if (len(r) > 1):
    print("warning: more than one k")
    r = r[0]
  else:
    r = r[0]
  return r

def get_r_from_X(H):
  avH = H.groupby(["K", "n"]).max()
  avH = avH.groupby(["K"]).mean()
  r = avH["NAC"].loc[
    avH["X"] == np.nanmax(avH["X"])
  ].to_numpy().reshape(-1).astype(int)
  if (len(r) > 1):
    print("warning: more than one k")
    r = r[0]
  else:
    r = r[0]
  return r

def get_r_from_equivalence(k, H):
  return H.equivalence[H.equivalence[:, 0] == k, 1][0]

def get_r_from_X_diag(k, H, Z, R, nodes):
  from scipy.cluster.hierarchy import cut_tree, dendrogram
  labels = cut_tree(H, k).ravel()
  dR = adj2df(R[:nodes, :])
  dR = dR.loc[(dR.weight != 0)]
  dR["id"] = labels
  minus_one_Dc(dR)
  aesthetic_ids(dR)
  ##
  den_order = np.array(
    dendrogram(Z, no_plot=True)["ivl"]
  ).astype(int)
  RR = df2adj(dR, var="id")[:, den_order][den_order, :]
  dR = adj2df(RR)
  dR["id"] = dR.weight
  dR = dR.loc[(dR.id != 0)]
  ##
  unique_labels = np.unique(dR.id)
  len_unique_labels = 0
  dR = dR.loc[dR.id != -1]
  for label in unique_labels:
    if label == -1: continue
    dr_down = dR.source.loc[(dR.id == label) & (dR.target < dR.source)]
    dr_up = dR.target.loc[(dR.id == label) & (dR.source < dR.target)]
    len_between_up_down = len(set(dr_up).intersection(set(dr_down)))
    if len_between_up_down > 0:
      len_unique_labels += 1
    else:
      dR = dR.loc[dR.id != label]
  nodes_fair_communities = set(dR.source).intersection(set(dR.target))
  return nodes - len(nodes_fair_communities) + len_unique_labels

def get_r_from_X_diag_2(k, H, Z, R, nodes):
  from scipy.cluster.hierarchy import cut_tree, dendrogram
  labels = cut_tree(H, k).ravel()
  dR = adj2df(R[:nodes, :])
  dR = dR.loc[(dR.weight != 0)]
  dR["id"] = labels
  minus_one_Dc(dR)
  aesthetic_ids(dR)
  ##
  den_order = np.array(
    dendrogram(Z)["ivl"]
  ).astype(int)
  RR = df2adj(dR, var="id")[:, den_order][den_order, :]
  dR = adj2df(RR)
  dR["id"] = dR.weight
  dR = dR.loc[(dR.id != 0)]
  ##
  unique_labels = np.unique(dR.id)
  len_unique_labels = 0
  fair_nodes = 0
  for label in unique_labels:
    if label == -1: continue
    dr_down = dR.source.loc[(dR.id == label) & (dR.target < dR.source)]
    dr_up = dR.target.loc[(dR.id == label) & (dR.source < dR.target)]
    len_between_up_down = len(set(dr_down).intersection(set(dr_up)))
    if len_between_up_down > 0:
      len_unique_labels += 1
      fair_nodes += len_between_up_down
  return nodes - fair_nodes + len_unique_labels

def stats_maxsize_linkcom(data):
    subdata = data.groupby(["nodes"])["size"].max()
    mean = subdata.mean()
    std = subdata.std()
    return mean, std

def tree_dominant_nodes(data, labels):
  catch_nodes = []
  subdata = data.groupby(["nodes"]).max()
  max_size = subdata["size"]
  nodes = subdata.index
  for i, nd in enumerate(nodes):
    ids = data["ids"].loc[data["nodes"] == nd].to_numpy()
    if -1 not in ids: continue
    x = data.loc[
      (data["nodes"] == nd) &
      (data["ids"] == -1),
      "size"
    ].iloc[0]
    if x == max_size.iloc[i]:
      y = np.where(labels == nd)[0][0]
      catch_nodes.append(y)
  return catch_nodes

def NMI_single(labels, H, on=True, **kwargs):
  if on:
    K = []
    if "K" in kwargs.keys():
      for kk in kwargs["K"]:
        BH = H.BH[0]
        k = BH["NAC"].loc[BH["K"] == kk].to_numpy()
        K.append(k[0])
    else:
      K.append(np.unique(labels).shape[0])
    from scipy.cluster.hierarchy import cut_tree
    for k in K:
      h_ids = cut_tree(
        H.Z,
        n_clusters=k
      ).reshape(-1)
      from sklearn.metrics import normalized_mutual_info_score
      nmi = normalized_mutual_info_score(labels, h_ids)
      print("NMI({}): {}".format(k, nmi))

def NMI_average(gt, K, R, WSBM, on=True):
  if on:
    from sklearn.metrics import normalized_mutual_info_score
    pred = WSBM.labels.loc[
      (WSBM.labels["K"] == K) &
      (WSBM.labels["R"] == R),
      "labels"
    ].to_numpy()
    nmi = normalized_mutual_info_score(gt, pred)
    print("K: {}\tR: {}\tNMI: {}".format(K, R, nmi))
    return nmi

def NMI_label(gt, pred, on=True):
  if on:
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(gt, pred)
    print("NMI: {}".format(nmi))
    return nmi

def AD_NMI_label(gt, pred, on=True):
  if on:
    from sklearn.metrics import adjusted_mutual_info_score
    if np.sum(np.isnan(pred)) > 0: nmi = np.nan
    elif len(np.unique(pred)) == 1: nmi = np.nan
    else:
      nmi = adjusted_mutual_info_score(gt, pred)
    print("ADNMI: {}".format(nmi))
    return nmi

def AD_NMI_overlap(gt, pred, overlap, on=True):
  if on:
    x = list(overlap.keys())
    n = gt.shape[0]
    y = np.arange(n)
    y = [i for i in y if i not in x]
    from sklearn.metrics import adjusted_mutual_info_score
    nmi = adjusted_mutual_info_score(gt[y], pred[y])
    print("ADNMI: {:.4f}".format(nmi))
    return nmi
  else:
    return -1

def save_class(
  CLASS, pickle_path, class_name="duck", **kwargs
):
  path = join(
    pickle_path, "{}.pk".format(class_name)
  )
  if exists(path): remove(path)
  if "on" in kwargs.keys():
    if kwargs["on"]:
      with open(path, "wb") as f:
        pk.dump(CLASS, f)
  else:
    with open(path, "wb") as f:
        pk.dump(CLASS, f)

def read_class(pickle_path, class_name="duck", **kwargs):
  path = join(
    pickle_path, "{}.pk".format(class_name)
  )
  with open(path, "rb") as f:
    C =  pk.load(f)
  return C

def column_normalize(A):
  if np.sum(np.isnan(A)) > 0:
    raise ValueError("\nColumn normalied does not accept nan. Use instead column_normalize_nan.\n")
  C = A.sum(axis = 0)
  C = A.copy() / C
  C[np.isnan(C)] = 0
  return C

def column_normalize_nan(A):
  C = np.nansum(A, axis=0)
  C = A.copy() / C
  C[np.isnan(C)] = 0
  return C

def match(a, b):
    b_dict = {x: i for i, x in enumerate(b)}
    return np.array([b_dict.get(x, None) for x in a])
    
def sort_by_size(ids, nodes):
  # Define location memory ---
  c = 0
  # Define new id ----
  nids = np.zeros(nodes)
  # Find membership frequency ----
  from collections import Counter
  f = Counter(ids)
  f = dict(f)
  f = sort_dict_value(f)
  for key in f:
    w = np.where(ids == key)[0]
    lw = len(w)
    nids[c:(lw + c)] = w
    c += lw
  return nids.astype(int), f

def sort_dict_value(counter : dict):
  f = {
    k: v for k, v in sorted(
      counter.items(), key=lambda item: item[1],
      reverse=True
    )
  }
  return f

def invert_dict_single(f: dict):
  return {v : k for k, v in f.items()}

def invert_dict_multiple(f : dict):
  ff = {}
  for key in f.keys():
    for val in f[key]:
      if val not in ff.keys(): ff[val] = [key]
      else: ff[val] = ff[val] + [key]
  for key in ff.keys(): ff[key] = list(np.unique(ff[key]))
  return ff

# def membership2ids(Cr, dA):
#   from collections import Counter
#   cr_skim = skim_partition(Cr)
#   skm = np.unique(cr_skim)
#   skm = skm[skm != -1]
#   nodecom_2_id = {}
#   for r in skm:
#     r_nodes = np.where(cr_skim == r)[0]
#     if len(r_nodes) == 0: continue
#     dr = dA.loc[
#       (np.isin(dA.source, r_nodes)) &
#       (np.isin(dA.target, r_nodes))
#     ]
#     id_r_counter = dict(Counter(dr.id))
#     id_r_counter = sort_dict_value(id_r_counter).keys()
#     nodecom_2_id[r] = list(id_r_counter)[0]
#   return nodecom_2_id

def membership2ids(Cr, dA):
  skimCr = skim_partition(Cr)
  # uCr = np.unique(Cr)
  uCr = np.unique(skimCr[skimCr != -1])
  nodecom_2_id = {
    ur : [] for ur in uCr
  }
  for ur in uCr:
    ur_nodes = np.where(Cr == ur)[0]
    if len(ur_nodes) == 0: continue
    dur = np.unique(dA.id.loc[np.isin(dA.source, ur_nodes)])
    nodecom_2_id[ur] = nodecom_2_id[ur] + list(dur)
    dur = np.unique(dA.id.loc[np.isin(dA.target, ur_nodes)])
    nodecom_2_id[ur] = nodecom_2_id[ur] + list(dur)
  for key in nodecom_2_id:
    nodecom_2_id[key] = list(np.unique(nodecom_2_id[key]))
  return nodecom_2_id

def condense_madtrix(A):
  n = A.shape[0] * (A.shape[0] - 1) / 2
  cma = np.zeros(int(n))
  t = 0
  for i in np.arange(A.shape[0] - 1):
    for j in np.arange(i + 1, A.shape[0]):
      cma[t] =  A[i, j]
      t += 1
  return cma

def df2adj(dA, var="weight"):
  m = np.max(dA["source"].to_numpy()) + 1
  n = np.max(dA["target"].to_numpy()) + 1
  A = np.zeros((m.astype(int), n.astype(int)))
  A[
    dA["source"].to_numpy().astype(int),
    dA["target"].to_numpy().astype(int)
  ] = dA[var].to_numpy()
  return A

def adj2df(A):
  src = np.repeat(
    np.arange(A.shape[0]),
    A.shape[1]
  )
  tgt = np.tile(
    np.arange(A.shape[1]),
    A.shape[0]
  )
  dA = pd.DataFrame(
    {
      "source" : src,
      "target" : tgt,
      "weight" : A.reshape(-1)
    }
  )
  return dA

def omega_index_format(node_partition, noc_covers : dict, node_labels):
  rev = reverse_partition(node_partition, node_labels)
  nocs2parition(rev, noc_covers)
  return rev