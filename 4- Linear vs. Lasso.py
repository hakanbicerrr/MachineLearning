from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics._regression import r2_score
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
# Polynomial features.
poly = PolynomialFeatures(degree=12)
x_poly = poly.fit_transform(X_train)
# Linear regression model with default parameters.
model = LinearRegression()
model.fit(x_poly, y_train)
# Prediction.
pred_test = poly.transform(X_test)
result_test = model.predict(pred_test)
linreg_r2_test_score = r2_score(y_test, result_test)
print(linreg_r2_test_score)

model = Lasso(alpha=0.01, max_iter=10000)
model.fit(x_poly, y_train)
pred_test = poly.transform(X_test)
result_test = model.predict(pred_test)
lasso_r2_test_score = r2_score(y_test,result_test)
print(lasso_r2_test_score)

