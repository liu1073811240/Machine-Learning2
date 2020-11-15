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
print(confusion_matrix(y_test, y_pred))  # 根据召回率来判断，16个全部分类正确。第二行17个分类正确，1个分类错误。第三行11个全部分类正确

"""
精确度：precision，正确预测为正的，占全部预测为正的比例，TP / (TP+FP)
召回率：recall，正确预测为正的，占全部实际为正的比例，TP / (TP+FN)
F1-score：精确率和召回率的调和平均数，2 * precision*recall / (precision+recall)
类别数量：每类数据标签的数量。

微平均值：micro average，所有数据结果的平均值
宏平均值：macro average，所有标签结果的平均值
加权平均值：weighted average，所有标签结果的加权平均值
         (P1 * support1 + P2 * support2 + P2 * support3) / (support1+support2+support3)
"""