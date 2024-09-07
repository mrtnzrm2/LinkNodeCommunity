# Write any import statements here

def getArtisticPhotographCount(N: int, C: str, X: int, Y: int) -> int:
  # Write your code here
  if X > Y: X, Y = Y, X
  
  class carrier:
    def __init__(self) -> None:
      self.n = 0

    def insert(self, f : dict, key : str, l : int, i : int):
      if l > 0:
        if f["key"][-1] == "P" and key == "B": return
        elif f["key"][-1] == "B" and key == "P": return
      if "." in key: return
      if key in f["key"]: return
      if l == 2: self.n += 1; return
      for j in range(X, Y+1):
        if i + j > N-1: return
        f[f"tr{i}{j}"] = {"key" : f["key"]+key}
        self.insert(f[f"tr{i}{j}"], C[i+j], l+1, i+j)

  carr = carrier()

  for i in range(N - 2*X):
    f = {"key": ""}
    carr.insert(f, C[i], 0, i)
  
  return carr.n


N = 5
C = "APABA"
X = 2
Y = 3

# N = 5
# C = "APABA"
# X = 1
# Y = 2

# N = 8
# C = ".PBAAP.B"
# X = 1
# Y = 3

print(getArtisticPhotographCount(N, C, X, Y))