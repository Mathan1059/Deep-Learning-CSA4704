import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n = 300
data = {
    "Workload": np.random.normal(3, 1, n),
    "Stress": np.random.normal(3, 1, n),
    "Salary": np.random.normal(4, 1, n),
    "Promotion": np.random.normal(4, 1, n),
    "Teamwork": np.random.normal(4, 1, n),
    "Leadership": np.random.normal(3, 1, n),
    "WorkLife": np.random.normal(3, 1, n),
    "Motivation": np.random.normal(4, 1, n),
    "Recognition": np.random.normal(3, 1, n),
    "Satisfaction": np.random.normal(4, 1, n)
}

df = pd.DataFrame(data)
print(df.head())
print(df.describe())

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Employee Survey Correlation")
plt.show()

X = StandardScaler().fit_transform(df)

for k in [2, 3, 4]:
    fa = FactorAnalysis(n_components=k, random_state=42)
    fa.fit(X)
    loadings = pd.DataFrame(
        fa.components_.T,
        index=df.columns,
        columns=[f"Factor {i+1}" for i in range(k)]
    )

    print(f"\nFactor Analysis with {k} Factors")
    print(loadings.round(2))

    plt.figure(figsize=(7, 6))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Factor Loadings: {k} Factors")
    plt.show()

pca = PCA(n_components=3)
pca.fit(X)

print("PCA Explained Variance:")
print(np.round(pca.explained_variance_ratio_, 3))

print("Factor Analysis Noise Variance:")
print(np.round(fa.noise_variance_, 3))
