import numpy as np
import numpy.typing as npt


def test_polynomial_model(X : npt.ArrayLike, Y : npt.ArrayLike, degree=1):
  import statsmodels.api as sm
  from sklearn.preprocessing import PolynomialFeatures, StandardScaler

  scaler = StandardScaler()
  Z = scaler.fit_transform(X.reshape(-1, 1))

  polynomial_features = PolynomialFeatures(degree=degree)
  XP = polynomial_features.fit_transform(Z)

  model = sm.OLS(Y, XP).fit()
  print(model.summary())


