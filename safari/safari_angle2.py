import numpy as np

def cartesian_to_spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,2] = np.arctan2(np.sqrt(xy), xyz[:,2])
    ptsnew[:,1] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

x = np.array([1, -1, 0])
xs = cartesian_to_spherical(x.reshape(1, -1)).ravel()

print(xs)

Rz = lambda phi: np.array(
  [
    [np.cos(phi), np.sin(phi), 0],
    [-np.sin(phi), np.cos(phi), 0],
    [0, 0, 1]
  ]
)

Rz2 = lambda phi: np.array(
  [
    [np.cos(phi), -np.sin(phi), 0],
    [np.sin(phi), np.cos(phi), 0],
    [0, 0, 1]
  ]
)

Ry = lambda phi: np.array(
  [
    [np.cos(phi), 0, -np.sin(phi)],
    [0,1, 0],
    [np.sin(phi), 0, np.cos(phi)]
  ]
)

print(Rz(xs[1]) @ x.reshape(-1, 1))