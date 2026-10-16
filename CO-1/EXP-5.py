import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X = np.sort(5 * np.random.rand(150, 1), axis=0)
y = 10 + 3 * X.ravel() + 2 * X.ravel()**2 + np.random.randn(150) * 5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = [1, 3, 9, 15]
train_errors = []
test_errors = []

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data")

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

    X_plot = np.linspace(0, 5, 200).reshape(-1, 1)
    plt.plot(X_plot, model.predict(X_plot), label=f"Degree {degree}")

plt.xlabel("House Area")
plt.ylabel("House Price")
plt.title("Underfitting and Overfitting")
plt.legend()
plt.show()

plt.plot(degrees, train_errors, marker="o", label="Training Error")
plt.plot(degrees, test_errors, marker="o", label="Validation Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Analysis")
plt.legend()
plt.grid()
plt.show()

for i, degree in enumerate(degrees):
    print("Degree:", degree)
    print("Training Error:", round(train_errors[i], 2))
    print("Validation Error:", round(test_errors[i], 2))

best_degree = degrees[np.argmin(test_errors)]
print("Best Polynomial Degree:", best_degree)
