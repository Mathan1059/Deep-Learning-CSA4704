import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

text = ("deep learning improves prediction and " * 100)
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}

encoded = np.array([char_to_id[c] for c in text])
seq_len = 20

X = np.array([encoded[i:i+seq_len]
              for i in range(len(encoded)-seq_len)])
y = np.array([encoded[i+seq_len]
              for i in range(len(encoded)-seq_len)])

X = tf.one_hot(X, len(chars))
y = tf.keras.utils.to_categorical(y, len(chars))

def build_model(clipnorm=None, clipvalue=None):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.05,
        clipnorm=clipnorm,
        clipvalue=clipvalue
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, len(chars))),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(len(chars), activation="softmax")
    ])
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy")
    return model

def train_model(clipnorm=None, clipvalue=None):
    model = build_model(clipnorm, clipvalue)
    losses, norms = [], []

    for epoch in range(5):
        for start in range(0, len(X), 64):
            xb = X[start:start+64]
            yb = y[start:start+64]

            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(yb, pred)
                )

            grads = tape.gradient(loss, model.trainable_variables)
            norm = tf.linalg.global_norm(grads).numpy()
            norms.append(norm)

            model.optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )
            losses.append(loss.numpy())

    return losses, norms

loss1, norm1 = train_model()
loss2, norm2 = train_model(clipnorm=1.0)
loss3, norm3 = train_model(clipvalue=0.5)

plt.figure(figsize=(10, 4))
plt.plot(norm1, label="No Clipping")
plt.plot(norm2, label="Clip Norm = 1")
plt.plot(norm3, label="Clip Value = 0.5")
plt.ylabel("Gradient Norm")
plt.xlabel("Training Step")
plt.title("Gradient Norm Comparison")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(loss1, label="No Clipping")
plt.plot(loss2, label="Clip Norm = 1")
plt.plot(loss3, label="Clip Value = 0.5")
plt.ylabel("Loss")
plt.xlabel("Training Step")
plt.title("Training Loss Comparison")
plt.legend()
plt.show()

print("Unclipped maximum gradient norm:", np.max(norm1))
print("Clipnorm maximum gradient norm:", np.max(norm2))
print("Clipvalue maximum gradient norm:", np.max(norm3))
