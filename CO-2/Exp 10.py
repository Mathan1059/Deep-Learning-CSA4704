import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

X = np.random.rand(500, 8)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = {}

models = {
    "Baseline": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=200, random_state=42
    ),
    "L2": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=0.1, max_iter=200, random_state=42
    ),
    "Dropout Alternative": MLPClassifier(
        hidden_layer_sizes=(32, 16),
        alpha=0.01, max_iter=200, random_state=42
    ),
    "Early Stopping": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        early_stopping=True, validation_fraction=0.2,
        max_iter=200, random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = [train_acc, test_acc]

    print("\n", name)
    print("Training Accuracy:", round(train_acc, 3))
    print("Test Accuracy:", round(test_acc, 3))
    print("Train-Test Gap:", round(train_acc - test_acc, 3))

plt.figure(figsize=(8, 5))
names = list(results.keys())
train = [results[n][0] for n in names]
test = [results[n][1] for n in names]

x = np.arange(len(names))
plt.bar(x - 0.2, train, 0.4, label="Training")
plt.bar(x + 0.2, test, 0.4, label="Testing")
plt.xticks(x, names, rotation=20)
plt.ylabel("Accuracy")
plt.title("Regularization Comparison")
plt.legend()
plt.tight_layout()
plt.show()
