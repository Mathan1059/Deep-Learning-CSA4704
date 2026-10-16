import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

data = np.random.randint(10, 100, (200, 5))
columns = ["Hemoglobin", "RBC", "Glucose",
           "Cholesterol", "BMI"]

df = pd.DataFrame(data, columns=columns)

standardized = (df - df.mean()) / df.std()

covariance = standardized.cov()
correlation = standardized.corr()

print("Covariance Matrix:\n", covariance.round(2))
print("\nCorrelation Matrix:\n", correlation.round(2))

eigenvalues, eigenvectors = np.linalg.eigh(covariance)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

variance = eigenvalues / eigenvalues.sum() * 100
print("\nVariance Percentage:\n", variance.round(2))

plt.figure(figsize=(6, 4))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(5), columns, rotation=45)
plt.yticks(range(5), columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

projected = standardized.to_numpy() @ eigenvectors[:, :2]

plt.figure(figsize=(6, 4))
plt.scatter(projected[:, 0], projected[:, 1])
plt.title("Data Projection Using Top Two Eigenvectors")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
