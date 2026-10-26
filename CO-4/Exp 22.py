import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

np.random.seed(42)

n = 1000
t = np.linspace(0, 8, n)

s1 = np.sin(2 * np.pi * t)
s2 = np.sign(np.sin(2 * np.pi * 2 * t))
s3 = 2 * (t % 1) - 1

S = np.c_[s1, s2, s3]
S = (S - S.mean(axis=0)) / S.std(axis=0)

A = np.array([[1, 2, 1], [2, 1, 3], [3, 2, 2]])
X = S @ A.T

ica = FastICA(n_components=3, random_state=42)
S_ica = ica.fit_transform(X)

pca = PCA(n_components=3)
S_pca = pca.fit_transform(X)

fig, axes = plt.subplots(4, 3, figsize=(12, 9))

datasets = [S, X, S_ica, S_pca]
titles = ["Original Sources", "Mixed Signals",
          "ICA Recovered", "PCA Components"]

for row, data in enumerate(datasets):
    for col in range(3):
        axes[row, col].plot(data[:, col])
        axes[row, col].set_title(
            f"{titles[row]} - Signal {col+1}"
        )

plt.tight_layout()
plt.show()

correlation = np.abs(np.corrcoef(S.T, S_ica.T)[:3, 3:])
print("Absolute Correlation Between Original and ICA Sources:")
print(np.round(correlation, 2))

print("ICA Reconstruction Error:",
      round(np.mean((X - ica.inverse_transform(S_ica)) ** 2), 4))

print("PCA Explained Variance:")
print(np.round(pca.explained_variance_ratio_, 3))
