import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix

np.random.seed(42)

X = np.random.rand(200, 4)
y = ((X[:, 0] + X[:, 1]) > 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("F1 Score:", f1_score(y_test, pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

plt.plot(model.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.grid()
plt.show()

for layers in [(8,), (16, 8), (32, 16, 8)]:
    m = MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=500,
        random_state=42
    )
    m.fit(X_train, y_train)
    print("Layers:", layers)
    print("Accuracy:", accuracy_score(y_test, m.predict(X_test)))
