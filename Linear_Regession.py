import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train, diabetes_X_test = diabetes_X[:-20], diabetes_X[-20:]

diabetes_y_train, diabetes_y_test = diabetes_y[:-20], diabetes_y[-20:]

linear_reg = linear_model.LinearRegression()
linear_reg.fit(diabetes_X_train,diabetes_y_train)
diabetes_y_pred = linear_reg.predict(diabetes_X_test)
print(linear_reg.coef_, linear_reg.intercept_)

print("mean squared error = %0.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

plt.scatter(diabetes_X_test,diabetes_y_test, c='black')
plt.plot(diabetes_X_test, diabetes_y_pred, c= 'blue', linewidth = 3)
plt.xticks()
plt.yticks()
plt.show()


