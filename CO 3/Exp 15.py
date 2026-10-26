import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

prices = 100 + np.cumsum(np.random.normal(0, 1, 1000))

def create_data(length):
    X, y = [], []
    for i in range(length, len(prices)):
        X.append(prices[i-length:i])
        y.append(prices[i])
    X = np.array(X).reshape(-1, length, 1)
    return X, np.array(y)

def train_model(length, activation):
    X, y = create_data(length)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(length, 1)),
        tf.keras.layers.SimpleRNN(16, activation=activation),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y[:split], epochs=5, batch_size=32, verbose=0)

    sample = tf.convert_to_tensor(X_test[:1])
    with tf.GradientTape() as tape:
        tape.watch(sample)
        output = model(sample, training=False)
    gradient = tape.gradient(output, sample).numpy()[0, :, 0]

    return gradient

for length in [10, 50, 100]:
    for activation in ["tanh", "relu"]:
        gradients = train_model(length, activation)
        plt.plot(
            range(length), np.abs(gradients),
            label=f"Length {length}, {activation}"
        )

plt.yscale("log")
plt.xlabel("Time Step")
plt.ylabel("Absolute Gradient Magnitude")
plt.title("RNN Gradient Analysis")
plt.legend()
plt.grid()
plt.show()
