import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

np.random.seed(42)

n = 500
z = np.random.randn(n, 2)
W = np.array([[2, 1], [1, 3], [2, -1], [1, 2]])
X = z @ W.T + np.random.normal(0, 0.5, (n, 4))

X[::10, 0] = np.nan
X[::15, 2] = np.nan

mean = np.nanmean(X, axis=0)
missing = np.isnan(X)
X_filled = np.where(missing, mean, X)

pca = PCA(n_components=2)
pca.fit(X_filled)

components = pca.components_.T
eigenvalues = pca.explained_variance_
noise_variance = pca.noise_variance_

print("Estimated Noise Variance:", round(noise_variance, 4))
print("Explained Variance:", pca.explained_variance_ratio_)

scores = pca.transform(X_filled)
reconstructed = pca.inverse_transform(scores)

print("Reconstruction MSE:",
      round(mean_squared_error(X_filled, reconstructed), 4))

plt.figure(figsize=(6, 4))
plt.scatter(scores[:, 0], scores[:, 1], alpha=0.5)
plt.xlabel("Latent Variable 1")
plt.ylabel("Latent Variable 2")
plt.title("Probabilistic PCA Latent Space")
plt.show()

errors = []
for k in range(1, 5):
    model = PCA(n_components=k)
    reduced = model.fit_transform(X_filled)
    restored = model.inverse_transform(reduced)
    errors.append(mean_squared_error(X_filled, restored))

plt.plot(range(1, 5), errors, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Reconstruction MSE")
plt.title("Reconstruction Error")
plt.grid()
plt.show()

samples = np.random.multivariate_normal(
    pca.mean_,
    pca.get_covariance(),
    size=5
)

print("Generated Samples:")
print(np.round(samples, 2))
