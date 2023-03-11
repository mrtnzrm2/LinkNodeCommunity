import numpy as np

def jacp(u, v, n : int, lup, *args):
  if n > 0:
    U = np.tile(u, (n, 1))
    U = U / u[:, None]
    U[np.isnan(U)] = np.Inf
    V = np.tile(v, (n, 1))
    V = V / v[:, None]
    V[np.isnan(V)] = np.Inf
    return np.sum(1 / np.sum(np.maximum(U, V), axis=1))
  else: return np.nan

def jacw_ferenc(u, v, n : int, lup, *args):
  if n > 0:
    A = np.zeros((2, n))
    A[0, :] = u  
    A[1, :] = v
    return np.sum(np.abs(np.nanmin(A, axis=0))) / np.sum(np.abs(np.nanmax(A, axis=0)))
  else: return np.nan

def jacw(u, v, n : int, lup, *args):
  if n > 0:
    A = np.vstack([u, v])
    return (np.nansum(np.abs(np.nanmin(A, axis=0))) - np.nansum(np.abs(np.nanmax(A, axis=0)))) / n
  else: return np.nan

def jacw2(u, v, n : int, lup, *args):
  if n > 0:
    G = set([ k for k, (i, j) in enumerate(zip(u, v)) if i != lup or j != lup])
    Guv = set([k for k, (i, j) in enumerate(zip(u, v)) if i == lup and j == lup])
    ## G
    index_G = list(G)
    A = np.vstack([u, v])
    sim_G = np.sum(np.abs(np.nanmin(A[:, index_G], axis=0)))
    sim_G -= np.sum(np.abs(np.nanmax(A[:, index_G], axis=0)))
    ## Guv
    sim_Guv = 0.5 * len(Guv)
    return (sim_G + sim_Guv) / n
  else: return np.nan

def cosine_similarity(u, v, n : int, lup, *args):
  return np.dot(u, v) / (np.dot(u, u) + np.dot(v, v) - np.dot(u, v))

def cosine_similarity_ahn(u, v, n : int, *args):
  return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))

def NT(A, i, axis=0):
  if axis == 0:
    return np.nanmax(A[i, A[i, :] != 0]) - np.nanmin(A[i, A[i, :] != 0])
  elif axis == 1:
    return np.nanmax(A[A[:, i] != 0, i]) - np.nanmin(A[A[:, i] != 0, i])
  else:
    raise ValueError("Valid only for 2D arrays")

def hgenity(u, v, n : int, lup, *args):
  if n > 0:
    A = np.array([[i, j] for i, j in zip(u, v) if i != lup and j != lup]).T
    if len(A) > 0:
      maxuv = np.max(np.abs(A[0, :] - A[1, :]))
      minuv = np.min(np.abs(A[0, :] - A[1, :]))
      return minuv - maxuv
    else: return np.nan
  else: return np.nan

def binary_similarity(u, v, n :int , lup, *args):
  if n > 0:
    U = u.copy()
    V = v.copy()
    U[U != 0] = 1
    V[V != 0] = 1
    U[np.isnan(U)] = 0
    V[np.isnan(V)] = 0
    sim = np.sum(U == V) / n
    pi = np.sum(U) / n
    pj = np.sum(V) / n
    sim -=  1 - pj - pi + 2*pi*pj
    return sim
  else: np.nan

def simetric_KL(u, v, n : int, lup, *args):
  if n > 0:
    b = -6
    U = u.copy()
    V = v.copy()
    G = set([ k for k, (i, j) in enumerate(zip(u, v)) if i != lup and j != lup])
    Gu = set([ k for k, (i, j) in enumerate(zip(u, v)) if i != lup])
    Gv = set([ k for k, (i, j) in enumerate(zip(u, v)) if j != lup])
    Guv = set([k for k, (i, j) in enumerate(zip(u, v)) if i == lup and j == lup])
    ## G
    index_G = list(G)
    index_Gu = list(Gu - Guv)
    index_Gv = list(Gv - Guv)
    bu = np.array([b] * len(Gv))
    bv = np.array([b] * len(Gu))
    ## kl
    kl = -np.sum(U[index_G] * (np.log10(V[index_G]) - np.log10(U[index_G])))
    kl -= np.sum(V[index_G] * (np.log10(U[index_G]) - np.log10(V[index_G])))
    kl -= np.sum(U[index_Gu] * (bv - np.log10(U[index_Gu])))
    kl -= np.sum(V[index_Gv] * (bu - np.log10(V[index_Gv])))
    if np.isnan(kl): return np.nan
    else: return 1 - (kl /  (2 * n))
  else: return np.nan

sims = {
  "jacp" : jacp,
  "jacw" : jacw,
  "jacw2" : jacw2,
  "cos" : cosine_similarity,
  "ahn" : cosine_similarity_ahn,
  "hgeneity" : hgenity,
  "bsim" : binary_similarity,
  "kl" : simetric_KL
}