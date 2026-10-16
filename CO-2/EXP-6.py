import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

X = np.random.rand(200, 4)
y = 2 * X[:, 0] + 3 * X[:, 1] + X[:, 2] + X[:, 3]
y = y.reshape(-1, 1)

def loss(w):
    return (w[0] - 3)**2 + 2 * (w[1] - 2)**2

def gradient(w):
    return np.array([2 * (w[0] - 3), 4 * (w[1] - 2)])

optimizers = {
    "GD": 0.1,
    "SGD": 0.01,
    "Momentum": 0.1
}

plt.figure(figsize=(7, 5))

for name, lr in optimizers.items():
    w = np.array([-2.0, -2.0])
    velocity = np.zeros(2)
    path = [w.copy()]

    for epoch in range(50):
        grad = gradient(w)

        if name == "SGD":
            grad = grad + np.random.normal(0, 0.2, 2)

        if name == "Momentum":
            velocity = 0.9 * velocity + grad
            w = w - lr * velocity
        else:
            w = w - lr * grad

        path.append(w.copy())

    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], marker=".", label=name)

plt.xlabel("Weight 1")
plt.ylabel("Weight 2")
plt.title("Optimizer Convergence Paths")
plt.legend()
plt.grid()
plt.show()

results = {}

for name in ["SGD", "Momentum", "Adam"]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    if name == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    elif name == "Momentum":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.01, momentum=0.9
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer, loss="mse")

    history = model.fit(
        X, y, epochs=50, batch_size=16, verbose=0
    )

    results[name] = history.history["loss"]
    print(name, "Final Loss:", round(results[name][-1], 4))

for name, losses in results.items():
    plt.plot(losses, label=name)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Optimizer Loss Comparison")
plt.legend()
plt.grid()
plt.show()
