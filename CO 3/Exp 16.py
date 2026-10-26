import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
tf.random.set_seed(42)

data = 100 + np.cumsum(np.random.normal(0, 1, 1200))
df = pd.DataFrame({"Passengers": data})

values = df["Passengers"].values
X, y = [], []

for i in range(30, len(values)):
    X.append(values[i-30:i])
    y.append(values[i])

X = np.array(X).reshape(-1, 30, 1)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

results = {}

for name, layer in [
    ("SimpleRNN", tf.keras.layers.SimpleRNN),
    ("LSTM", tf.keras.layers.LSTM)
]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 1)),
        layer(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    start = time.time()
    history = model.fit(
        X_train, y_train, epochs=10,
        batch_size=32, validation_split=0.2, verbose=0
    )
    train_time = time.time() - start

    pred = model.predict(X_test, verbose=0).flatten()

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / (y_test + 1e-8))) * 100

    results[name] = {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Training Time": train_time,
        "History": history.history,
        "Predictions": pred
    }

    plt.plot(history.history["loss"], label=name + " Training")
    plt.plot(history.history["val_loss"], label=name + " Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN vs LSTM Loss")
plt.legend()
plt.show()

print(pd.DataFrame([
    {k: v for k, v in result.items()
     if k in ["Model", "MAE", "RMSE", "MAPE", "Training Time"]}
    for result in results.values()
]).round(3))

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual")

for name, result in results.items():
    plt.plot(result["Predictions"], label=name + " Prediction")

plt.xlabel("Time")
plt.ylabel("Passenger Count")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()
