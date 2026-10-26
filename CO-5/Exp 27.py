import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from time import time

np.random.seed(42)
tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:500], y_test[:500]

class RBM:
    def __init__(self, nv, nh):
        self.W = np.random.normal(0, 0.01, (nv, nh)).astype("float32")
        self.b = np.zeros(nh, dtype="float32")
        self.a = np.zeros(nv, dtype="float32")

    def train(self, X, epochs=2):
        for _ in range(epochs):
            for i in range(0, len(X), 32):
                v0 = X[i:i + 32]
                ph0 = tf.sigmoid(v0 @ self.W + self.b).numpy()
                h0 = (np.random.rand(*ph0.shape) < ph0).astype("float32")
                pv1 = tf.sigmoid(h0 @ self.W.T + self.a).numpy()
                v1 = (np.random.rand(*pv1.shape) < pv1).astype("float32")
                ph1 = tf.sigmoid(v1 @ self.W + self.b).numpy()

                n = len(v0)
                self.W += 0.01 * (v0.T @ ph0 - v1.T @ ph1) / n
                self.a += 0.01 * np.mean(v0 - v1, axis=0)
                self.b += 0.01 * np.mean(ph0 - ph1, axis=0)

    def transform(self, X):
        return tf.sigmoid(X @ self.W + self.b).numpy()

start = time()

rbm1 = RBM(784, 500)
rbm1.train(x_train)
h1 = rbm1.transform(x_train)

rbm2 = RBM(500, 250)
rbm2.train(h1)
h2 = rbm2.transform(h1)

print("RBM pretraining completed")
print("Training time:", round(time() - start, 2), "seconds")

def build_model(pretrained=False):
    model = Sequential()
    if pretrained:
        model.add(Dense(500, activation="sigmoid",
                        input_shape=(784,),
                        weights=[rbm1.W, rbm1.b]))
        model.add(Dense(250, activation="sigmoid",
                        weights=[rbm2.W, rbm2.b]))
    else:
        model.add(Dense(500, activation="sigmoid",
                        input_shape=(784,)))
        model.add(Dense(250, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

pretrained_model = build_model(True)
random_model = build_model(False)

history_pre = pretrained_model.fit(
    x_train, y_train, epochs=5, batch_size=32,
    validation_split=0.1, verbose=1
)

history_random = random_model.fit(
    x_train, y_train, epochs=5, batch_size=32,
    validation_split=0.1, verbose=1
)

loss1, acc1 = pretrained_model.evaluate(x_test, y_test, verbose=0)
loss2, acc2 = random_model.evaluate(x_test, y_test, verbose=0)

print("Pretrained DBN Accuracy:", round(acc1 * 100, 2), "%")
print("Random Initialization Accuracy:", round(acc2 * 100, 2), "%")

plt.plot(history_pre.history["accuracy"], label="Pretrained DBN")
plt.plot(history_random.history["accuracy"], label="Random Initialization")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend()
plt.show()

plt.plot(history_pre.history["loss"], label="Pretrained DBN")
plt.plot(history_random.history["loss"], label="Random Initialization")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend()
plt.show()

features = pretrained_model.layers[1]( 
    pretrained_model.layers[0](x_test)
).numpy()

pca = PCA(n_components=2)
points = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], c=y_test, cmap="tab10")
plt.colorbar()
plt.title("DBN Learned Representations")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

weights = rbm1.W.T[:25]
plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(weights[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.suptitle("First RBM Learned Filters")
plt.show()
