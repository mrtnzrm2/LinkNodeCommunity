import numpy as np

s = 10000
Pe = 1e-2
m = 4
Ui = np.random.randint(2, size=s)
U = Ui.copy()
for i in np.arange(m-1):
  U = np.vstack([U, Ui])

U = U.T.ravel()

Pr = np.array([Pe, 1-Pe])

V = np.zeros(m*s).astype(int)

G1 = np.random.choice([0, 1], m*s, p=Pr)
G0 = np.random.choice([1, 0], m*s, p=Pr)

for i in np.arange(m*s):
  if U[i] == 1:
    V[i] = G1[i]
  else:
    V[i] = G0[i]

Vf = np.zeros(s)

for i in np.arange(s):
  Vf[i] = np.mean(V[(m*i):(m*(i + 1))])

Vf = np.round(Vf).astype(int)

from sklearn.metrics import mutual_info_score


Iuv = mutual_info_score(Ui, Vf)

print(Iuv)


print((np.sum(Vf[Ui == 0] == 1) + np.sum(Vf[Ui == 1] == 0)) / s)

C = 1 + Pe * np.log(Pe) + (1-Pe) * np.log(1-Pe)
print(C)