import numpy as np
from sklearn import linear_model, svm, neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

# 加载数据
iris = datasets.load_iris()
x, y = iris.data, iris.target

# 划分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 创建模型
clf = svm.SVC(kernel='rbf')

# 模型拟合
clf.fit(x_train, y_train)

# 预测
y_pred = clf.predict(x_test)

# 评估
print(accuracy_score(y_test, y_pred))

# f1_score: F1 = 2*((P*R)/(P+R))
print(f1_score(y_test, y_pred, average='macro'))

# 分类报告
print(classification_report(y_test, y_pred))

# 混淆矩阵
print(confusion_matrix(y_test, y_pred))








