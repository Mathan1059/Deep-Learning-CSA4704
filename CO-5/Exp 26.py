import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from time import time

np.random.seed(42)

X = load_digits().data
X = (X > 8).astype(float)
X = X[:500]

nv, nh = 64, 32
lr = 0.01
batch = 32

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

class RBM:
    def __init__(self, nv, nh):
        self.W = np.random.normal(0, 0.01, (nv, nh))
        self.a = np.zeros(nv)
        self.b = np.zeros(nh)
        self.vpersistent = None

    def sample_h(self, v):
        p = sigmoid(v @ self.W + self.b)
        return p, (np.random.rand(*p.shape) < p).astype(float)

    def sample_v(self, h):
        p = sigmoid(h @ self.W.T + self.a)
        return p, (np.random.rand(*p.shape) < p).astype(float)

    def gibbs(self, v, steps):
        for _ in range(steps):
            _, h = self.sample_h(v)
            p, v = self.sample_v(h)
        return p, v

    def free_energy(self, v):
        return -v @ self.a - np.sum(
            np.logaddexp(0, v @ self.W + self.b), axis=1
        )

    def train(self, X, k=1, pcd=False, epochs=5):
        errors, pll = [], []
        start = time()

        for epoch in range(epochs):
            np.random.shuffle(X)
            err, fe, count = 0, 0, 0

            for i in range(0, len(X), batch):
                v0 = X[i:i + batch]
                ph0, _ = self.sample_h(v0)

                if pcd:
                    if self.vpersistent is None:
                        self.vpersistent = np.random.randint(
                            0, 2, v0.shape
                        ).astype(float)
                    vk = self.vpersistent
                else:
                    vk = v0.copy()

                for _ in range(k):
                    _, hk = self.sample_h(vk)
                    pv, vk = self.sample_v(hk)

                if pcd:
                    self.vpersistent = vk.copy()

                phk, _ = self.sample_h(vk)
                n = len(v0)

                self.W += lr * (
                    v0.T @ ph0 - vk.T @ phk
                ) / n
                self.a += lr * np.mean(v0 - vk, axis=0)
                self.b += lr * np.mean(ph0 - phk, axis=0)

                err += np.sum((v0 - pv) ** 2)
                fe += np.sum(self.free_energy(v0))
                count += n

            errors.append(err / count / nv)
            pll.append(-fe / count)

        return errors, pll, time() - start

    def show_chain(self, steps):
        v = np.random.randint(0, 2, (1, nv)).astype(float)
        _, v = self.gibbs(v, steps)
        return v.reshape(8, 8)

methods = {"CD-1": (1, False), "CD-5": (5, False),
           "CD-10": (10, False), "PCD": (1, True)}

results = {}
models = {}

for name, (k, pcd) in methods.items():
    model = RBM(nv, nh)
    error, pll, duration = model.train(
        X.copy(), k=k, pcd=pcd, epochs=5
    )
    results[name] = (error, pll, duration)
    models[name] = model
    print(name, "Time:", round(duration, 2), "seconds")

plt.figure(figsize=(10, 4))
for name, (error, pll, duration) in results.items():
    plt.plot(error, label=name)
plt.title("Reconstruction Error")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
for name, (error, pll, duration) in results.items():
    plt.plot(pll, label=name)
plt.title("Negative Average Free Energy")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.show()

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for row, name in enumerate(methods):
    for col, steps in enumerate([1, 10, 100, 1000]):
        axes[row, col].imshow(
            models[name].show_chain(steps), cmap="gray"
        )
        axes[row, col].set_title(name + " " + str(steps))
        axes[row, col].axis("off")
plt.tight_layout()
plt.show()

print("\nTraining Time and Final Reconstruction Error")
for name, (error, pll, duration) in results.items():
    print(name, "Time:", round(duration, 2),
          "Error:", round(error[-1], 4))
