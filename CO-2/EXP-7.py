import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [2, 3], [3, 4], [4, 2], [5, 5],
    [6, 3], [7, 6], [1, 2], [2, 5]
], dtype=float)

y = np.array([0, 0, 0, 1, 1, 1, 0, 1])

weights = np.zeros(2)
bias = 0
lr = 0.1

for epoch in range(20):
    errors = 0
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        pred = 1 if z >= 0 else 0
        error = y[i] - pred
        weights += lr * error * X[i]
        bias += lr * error
        errors += abs(error)
    print("Epoch:", epoch + 1, "Errors:", errors)

print("Weights:", weights)
print("Bias:", bias)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
x_line = np.linspace(0, 8, 100)

if weights[1] != 0:
    y_line = -(weights[0] * x_line + bias) / weights[1]
    plt.plot(x_line, y_line, color="black")

plt.xlabel("Credit Score")
plt.ylabel("Income-to-Debt Ratio")
plt.title("Perceptron Decision Boundary")
plt.show()

XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
YOR = np.array([0, 1, 1, 0])

w = np.zeros(2)
b = 0

for epoch in range(20):
    for i in range(4):
        pred = int(np.dot(XOR[i], w) + b >= 0)
        error = YOR[i] - pred
        w += 0.1 * error * XOR[i]
        b += 0.1 * error

print("XOR Predictions:")
for x in XOR:
    print(x, int(np.dot(x, w) + b >= 0))

x = np.array([-2, -1, 0, 1, 2])
print("Step:", (x >= 0).astype(int))
print("Sigmoid:", 1 / (1 + np.exp(-x)))
print("ReLU:", np.maximum(0, x))
