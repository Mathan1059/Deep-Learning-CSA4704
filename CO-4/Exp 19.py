import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X = digits.data
images = digits.images

X_mean = X.mean(axis=0)
X_centered = X - X_mean

cov = np.cov(X_centered, rowvar=False)
values, vectors = np.linalg.eigh(cov)

idx = np.argsort(values)[::-1]
values = values[idx]
vectors = vectors[:, idx]

explained = values / values.sum()
cumulative = np.cumsum(explained)

print("Components for 95% variance:",
      np.argmax(cumulative >= 0.95) + 1)

pca = PCA()
pca.fit(X)

print("Sklearn explained variance:",
      np.round(pca.explained_variance_ratio_[:5], 4))
print("Manual explained variance:",
      np.round(explained[:5], 4))

plt.plot(range(1, len(explained) + 1), explained, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.show()

plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
plt.axhline(0.95, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.show()

plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(vectors[:, i].reshape(8, 8), cmap="gray")
    plt.title(f"PC {i+1}")
    plt.axis("off")
plt.suptitle("Top Eigenvectors")
plt.show()

fig, axes = plt.subplots(3, 4, figsize=(10, 7))
sample = X[:1]

for row, n in enumerate([10, 50, 100]):
    pca = PCA(n_components=n)
    reduced = pca.fit_transform(X)
    reconstructed = pca.inverse_transform(reduced)

    for col in range(4):
        axes[row, col].imshow(
            reconstructed[col].reshape(8, 8), cmap="gray"
        )
        axes[row, col].set_title(f"{n} Components")
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
