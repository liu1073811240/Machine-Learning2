import numpy as np
import matplotlib.pyplot as plt


def normalization1(x):
    # 归一化（0~1）：x_ = (x-x_min) / (x_max-x_min)
    return [(float(i) -min(x)) / (max(x) - min(x)) for i in x]


def normalization2(x):
    # 均值化： x_ = (x-x_mean) / (x_max-x_min)
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


def normalization3(x):
    # 标准化（u=0, sigma=1） x=(x-u)/σ
    x_mean = np.mean(x)
    s2 = np.mean([(i - np.mean(x)) ** 2 for i in x])
    std = np.sqrt(s2)

    # return [(i - x_mean) / (s2 + 0.00001) for i in x]
    return [(i - x_mean) / (std + 0.00001) for i in x]

def normalization4(x):
    x_mean = [(float(i) / np.max(x) - 0.5) / 0.5 for i in x]
    return x_mean

l1 = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
      11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

print(np.mean(l1))

cs = []
for i in l1:
    c = l1.count(i)
    print(c)
    cs.append(c)

# print(cs)

n1 = normalization1(l1)
print(n1)
n2 = normalization2(l1)
print(n2)
n3 = normalization3(l1)
print(n3)
n4 = normalization4(l1)
print(n4)

plt.plot(l1, cs)
# plt.plot(n1, cs)
# plt.plot(n2, cs)
# plt.plot(n3, cs)
plt.plot(n4, cs)
plt.xlabel("data")
plt.ylabel("label")

plt.show()




