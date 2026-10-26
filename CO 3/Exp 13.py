import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
tf.random.set_seed(42)

data = np.sin(np.arange(8760) * 2 * np.pi / 24)
data += 0.5 * np.sin(np.arange(8760) * 2 * np.pi / 168)
data += np.random.normal(0, 0.1, 8760)

df = pd.DataFrame({"Demand": data})
df["Demand"] = df["Demand"].interpolate()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Demand"]])

X, y = [], []
for i in range(24, len(scaled)):
    X.append(scaled[i-24:i, 0])
    y.append(scaled[i, 0])

X = np.array(X).reshape(-1, 24, 1)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24, 1)),
    tf.keras.layers.SimpleRNN(32, activation="tanh"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
history = model.fit(
    X_train, y_train, epochs=10,
    batch_size=32, validation_split=0.2, verbose=1
)

pred = model.predict(X_test, verbose=0)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
predicted = scaler.inverse_transform(pred)

print("MAE:", mean_absolute_error(actual, predicted))
print("RMSE:", np.sqrt(mean_squared_error(actual, predicted)))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(actual[:100], label="Actual Demand")
plt.plot(predicted[:100], label="Predicted Demand")
plt.xlabel("Time")
plt.ylabel("Demand")
plt.legend()
plt.show()
