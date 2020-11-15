import numpy as np

# [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
x1 = np.array([2, 5, 9, 4, 8, 7])
y1 = np.array([3, 4, 6, 7, 1, 2])

print(x1.var(), y1.var())

x2 = np.array([2, 5, 4])
y2 = np.array([3, 4, 7])
print(x2.var(), y2.var())

x3 = np.array([8, 9])
y3 = np.array([1, 6])
print(x3.var(), y3.var())







