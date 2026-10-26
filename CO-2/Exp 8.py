import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))
lr = 0.5
losses = []
for epoch in range(5000):
    h = 1 / (1 + np.exp(-(X @ W1 + b1)))
    out = 1 / (1 + np.exp(-(h @ W2 + b2)))
    loss = -np.mean(Y * np.log(out + 1e-8) +
                    (1 - Y) * np.log(1 - out + 1e-8))
    losses.append(loss)
    d2 = (out - Y) / len(X)
    dW2 = h.T @ d2
    db2 = np.sum(d2, axis=0, keepdims=True)
    d1 = (d2 @ W2.T) * h * (1 - h)
    dW1 = X.T @ d1
    db1 = np.sum(d1, axis=0, keepdims=True)
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
print("Predictions:")
print(np.round(out, 3))
print("Classes:", (out >= 0.5).astype(int))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Backpropagation Training Loss")
plt.grid()
plt.show()
xx, yy = np.meshgrid(
    np.linspace(-0.2, 1.2, 100),
    np.linspace(-0.2, 1.2, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]
hidden = 1 / (1 + np.exp(-(grid @ W1 + b1)))
pred = 1 / (1 + np.exp(-(hidden @ W2 + b2)))
Z = pred.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(),
            edgecolors="black", cmap="coolwarm")
plt.title("XOR Decision Boundary")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()
epsilon = 1e-5
original = W1[0, 0]
W1[0, 0] = original + epsilon
lp = -np.mean(Y * np.log(1 / (1 + np.exp(
    -(1 / (1 + np.exp(-(X @ W1 + b1))) @ W2 + b2))) + 1e-8))
W1[0, 0] = original - epsilon
lm = -np.mean(Y * np.log(1 / (1 + np.exp(
    -(1 / (1 + np.exp(-(X @ W1 + b1))) @ W2 + b2))) + 1e-8))
W1[0, 0] = original
print("Numerical Gradient:", (lp - lm) / (2 * epsilon))
