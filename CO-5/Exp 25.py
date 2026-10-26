import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

np.random.seed(42)

(X, _), _ = mnist.load_data()
X = (X[:5000].reshape(-1, 784) > 127).astype(float)

nv, nh = 784, 64
W = np.random.normal(0, 0.01, (nv, nh))
a = np.zeros(nv)
b = np.zeros(nh)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def energy(v, h):
    return -v @ a - h @ b - v @ W @ h

def free_energy(v):
    return -v @ a - np.sum(np.logaddexp(0, v @ W + b), axis=1)

errors, energies = [], []
lr, batch = 0.01, 32

for epoch in range(10):
    np.random.shuffle(X)
    err, en, count = 0, 0, 0

    for i in range(0, len(X), batch):
        v0 = X[i:i + batch]
        ph0 = sigmoid(v0 @ W + b)
        h0 = (np.random.rand(*ph0.shape) < ph0).astype(float)

        pv1 = sigmoid(h0 @ W.T + a)
        v1 = (np.random.rand(*pv1.shape) < pv1).astype(float)
        ph1 = sigmoid(v1 @ W + b)

        W += lr * (v0.T @ ph0 - v1.T @ ph1) / len(v0)
        a += lr * np.mean(v0 - v1, axis=0)
        b += lr * np.mean(ph0 - ph1, axis=0)

        err += np.sum((v0 - pv1) ** 2)
        en += np.sum(free_energy(v0))
        count += len(v0)

    errors.append(err / count / nv)
    energies.append(en / count)
    print("Epoch:", epoch + 1,
          "Error:", round(errors[-1], 4),
          "Free Energy:", round(energies[-1], 4))

plt.plot(errors)
plt.title("Reconstruction Error")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

plt.plot(energies)
plt.title("Average Free Energy")
plt.xlabel("Epoch")
plt.ylabel("Free Energy")
plt.show()

sample = X[:5]
hidden = sigmoid(sample @ W + b)
reconstructed = sigmoid(hidden @ W.T + a)

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.title("Original")

    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.title("Reconstructed")

plt.tight_layout()
plt.show()

v = sample[0]
h = (hidden[0] > 0.5).astype(float)
print("Energy:", round(energy(v, h), 4))
