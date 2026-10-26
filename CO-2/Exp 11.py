import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(X, y), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X = X.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y = y.flatten()
y_test = y_test.flatten()

X, y = X[:10000], y[:10000]
X_test, y_test = X_test[:2000], y_test[:2000]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    X, y, epochs=5, batch_size=64,
    validation_split=0.2
)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", round(accuracy * 100, 2), "%")

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

for i in range(5):
    plt.imshow(X_test[i])
    plt.title("Actual: " + str(y_test[i]) +
              " Predicted: " + str(pred[i]))
    plt.axis("off")
    plt.show()

feature_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[0].output
)
features = feature_model.predict(X_test[:1])

for i in range(min(6, features.shape[-1])):
    plt.subplot(2, 3, i + 1)
    plt.imshow(features[0, :, :, i], cmap="gray")
    plt.axis("off")

plt.suptitle("First Convolution Layer Feature Maps")
plt.tight_layout()
plt.show()
