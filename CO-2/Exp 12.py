import tensorflow as tf
import matplotlib.pyplot as plt
import time

(X, y), (Xt, yt) = tf.keras.datasets.cifar10.load_data()
X, Xt = X[:5000] / 255.0, Xt[:1000] / 255.0
y, yt = y[:5000], yt[:1000]

results = []

for layers, filters, pooling, lr, batch in [
    (2, 32, "max", 0.001, 64),
    (3, 32, "max", 0.001, 64),
    (4, 32, "max", 0.001, 64),
    (2, 64, "average", 0.001, 64),
    (2, 32, "max", 0.0001, 32)
]:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

    for i in range(layers):
        model.add(tf.keras.layers.Conv2D(
            filters, (3, 3), padding="same",
            activation="relu"
        ))
        if i == 0:
            model.add(
                tf.keras.layers.MaxPooling2D()
                if pooling == "max"
                else tf.keras.layers.AveragePooling2D()
            )

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start = time.time()
    model.fit(X, y, epochs=2, batch_size=batch, verbose=0)
    _, acc = model.evaluate(Xt, yt, verbose=0)
    elapsed = time.time() - start

    results.append([layers, filters, pooling, lr, batch,
                    model.count_params(), round(acc, 3),
                    round(elapsed, 2)])

    print("Layers:", layers, "Filters:", filters,
          "Pooling:", pooling, "LR:", lr, "Batch:", batch)
    print("Parameters:", model.count_params())
    print("Test Accuracy:", round(acc * 100, 2), "%")
    print("Training Time:", round(elapsed, 2), "seconds\n")

print("Comparison: Layers, Filters, Pooling, LR, Batch, Parameters, Accuracy, Time")
for row in results:
    print(row)

plt.plot(range(1, len(results) + 1),
         [r[6] for r in results], marker="o")
plt.xlabel("Configuration")
plt.ylabel("Test Accuracy")
plt.title("CNN Hyperparameter Comparison")
plt.grid()
plt.show()
