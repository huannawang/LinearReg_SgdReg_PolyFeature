# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:54:51 2023

@author: USER
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)


#====以訓練資料資料訓練線性回歸模型: 線性回歸模型的預測直線
lin_reg = LinearRegression()
lin_reg.fit(X, y)

X_new = np.array([[-3], [3]])
y_predict = lin_reg.predict(X_new)

plt.plot(X_new, y_predict, "r-", linewidth=2)


#====以訓練資料資料訓練線性回歸模型: 隨機梯度下降模型的預測直線
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel()) # ravel()轉成一維向量
y_predict = sgd_reg.predict(X_new)

         
plt.plot(X_new, y_predict, "b--", linewidth=2) #藍色虛線

#====以轉換後的訓練資料訓練線性回歸模型: 線性回歸模型的預測曲線
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X) # 產生x及x平方對應值

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) # 將x及x平方對應值帶入學習

X_new=np.linspace(-3, 3, 100).reshape(100, 1) # 產生x座標來預測
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly) # 將x及x平方對應值帶入預測y值

plt.plot(X_new, y_predict, "r-", linewidth=2, label="lin Predictions")

#====以轉換後的訓練資料訓練線性回歸模型: 隨機梯度下降模型的預測曲線
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X_poly, y.ravel()) # ravel()轉成一維向量
y_predict = sgd_reg.predict(X_new_poly)
         
plt.plot(X_new, y_predict, "b--", linewidth=2, label="sgd Predictions") #藍色虛線

#畫圖
plt.plot(X, y, "y.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.legend(loc="upper left", fontsize=14)
plt.show()





