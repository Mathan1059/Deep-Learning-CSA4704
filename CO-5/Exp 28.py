import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from scipy.special import expit

np.random.seed(42)

X = load_digits().data
X = (X > 8).astype(float)[:1000]

nv, nh1, nh2 = 64, 32, 16
W1 = np.random.normal(0, 0.01, (nv, nh1))
W2 = np.random.normal(0, 0.01, (nh1, nh2))
a = np.zeros(nv)
b1 = np.zeros(nh1)
b2 = np.zeros(nh2)

def sigmoid(x):
    return expit(np.clip(x, -30, 30))

def mean_field(v, steps=5):
    h1 = sigmoid(v @ W1 + b1)
    h2 = sigmoid(h1 @ W2 + b2)

    for _ in range(steps):
        h1 = sigmoid(v @ W1 + h2 @ W2.T + b1)
        h2 = sigmoid(h1 @ W2 + b2)

    return h1, h2

def reconstruct(v):
    h1, h2 = mean_field(v)
    return sigmoid(h1 @ W1.T + a)

def train_dbm(X, epochs=10, lr=0.01):
    errors = []

    for epoch in range(epochs):
        np.random.shuffle(X)
        total_error = 0

        for i in range(0, len(X), 32):
            v0 = X[i:i + 32]
            h10, h20 = mean_field(v0)

            vk = (np.random.rand(*v0.shape) < 0.5).astype(float)

            for _ in range(3):
                h1k = (np.random.rand(len(vk), nh1)
                       < sigmoid(vk @ W1 + b1)).astype(float)
                h2k = (np.random.rand(len(vk), nh2)
                       < sigmoid(h1k @ W2 + b2)).astype(float)
                vk = (np.random.rand(*vk.shape)
                      < sigmoid(h1k @ W1.T + a)).astype(float)

            h1neg, h2neg = mean_field(vk)

            W1[:] += lr * (
                v0.T @ h10 - vk.T @ h1neg
            ) / len(v0)

            W2[:] += lr * (
                h10.T @ h20 - h1neg.T @ h2neg
            ) / len(v0)

            a[:] += lr * np.mean(v0 - vk, axis=0)
            b1[:] += lr * np.mean(h10 - h1neg, axis=0)
            b2[:] += lr * np.mean(h20 - h2neg, axis=0)

            total_error += np.sum((v0 - reconstruct(v0)) ** 2)

        errors.append(total_error / len(X))
        print("Epoch:", epoch + 1,
              "Reconstruction Error:", round(errors[-1], 4))

    return errors

errors = train_dbm(X)

plt.plot(errors)
plt.title("DBM Reconstruction Error")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

sample = X[:5]
reconstructed = reconstruct(sample)

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample[i].reshape(8, 8), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed[i].reshape(8, 8), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(W1[:, i].reshape(8, 8), cmap="gray")
    plt.title("Filter " + str(i + 1))
    plt.axis("off")

plt.tight_layout()
plt.show()

h1, h2 = mean_field(sample)

plt.figure(figsize=(8, 3))
plt.imshow(h1, cmap="viridis", aspect="auto")
plt.title("Hidden Layer 1 Activations")
plt.xlabel("Hidden Units")
plt.ylabel("Samples")
plt.colorbar()
plt.show()

plt.figure(figsize=(8, 3))
plt.imshow(h2, cmap="viridis", aspect="auto")
plt.title("Hidden Layer 2 Activations")
plt.xlabel("Hidden Units")
plt.ylabel("Samples")
plt.colorbar()
plt.show()

generated = (np.random.rand(1, nv) < 0.5).astype(float)

for _ in range(100):
    h1g = (np.random.rand(1, nh1)
           < sigmoid(generated @ W1 + b1)).astype(float)
    h2g = (np.random.rand(1, nh2)
           < sigmoid(h1g @ W2 + b2)).astype(float)
    generated = (np.random.rand(1, nv)
                 < sigmoid(h1g @ W1.T + a)).astype(float)

plt.imshow(generated.reshape(8, 8), cmap="gray")
plt.title("Generated Sample")
plt.axis("off")
plt.show()

print("DBM training and visualisation completed.")
