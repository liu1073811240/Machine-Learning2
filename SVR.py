import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
import sklearn

rng = np.random.RandomState(0)

X = 5 * rng.rand(100, 1)  # 二维
y = np.sin(X).ravel()  # 一维

# 加噪声
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))  # 右边部分在-1.5,1.5范围， 每隔5个y数据就计算该范围对应的数据
# y[::5] += 3 * (0.5 - rng.rand(20, 1).ravel())
# print(y[::5])

# 当C越大，趋近无穷的时候，表示不允许分类误差的存在.
# 随着gamma的增大，存在对于测试集分类效果差而对训练分类效果好的情况，并且泛化误差容易出现过拟合。
svr = SVR(kernel='rbf', C=10, gamma=0.1)
svr.fit(X, y)

X_plot = np.linspace(0, 5, 100)
y_svr = svr.predict(X_plot[:, None])

plt.scatter(X, y)
plt.plot(X_plot, y_svr, color="red")
plt.show()






