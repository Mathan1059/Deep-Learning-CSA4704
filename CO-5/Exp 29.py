import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import fashion_mnist

np.random.seed(42)
tf.random.set_seed(42)

(X, _), _ = fashion_mnist.load_data()
X = X.astype("float32") / 255.0
X = X[..., np.newaxis]
X = X[:10000]

latent_dim = 2
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation="relu")(x)
mu = layers.Dense(latent_dim)(x)
log_var = layers.Dense(latent_dim)(x)

def sample(args):
    mu, log_var = args
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps

z = layers.Lambda(sample)([mu, log_var])
encoder = Model(inputs, [mu, log_var, z])

latent = layers.Input(shape=(latent_dim,))
x = layers.Dense(256, activation="relu")(latent)
x = layers.Dense(784, activation="sigmoid")(x)
outputs = layers.Reshape((28, 28, 1))(x)
decoder = Model(latent, outputs)

class VAE(Model):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(data)
            recon = self.decoder(z)
            rec_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, recon),
                    axis=(1, 2)
                )
            )
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var - tf.square(mu) - tf.exp(log_var),
                    axis=1
                )
            )
            loss = rec_loss + kl

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss}

vae = VAE()
vae.compile(optimizer="adam")
vae.fit(X, epochs=5, batch_size=128)

z_sample = np.random.normal(size=(10, latent_dim))
generated = decoder.predict(z_sample, verbose=0)

plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated[i].squeeze(), cmap="gray")
    plt.axis("off")
plt.suptitle("VAE Generated Images")
plt.show()

plt.figure(figsize=(8, 6))
mu_values, _, _ = encoder.predict(X[:2000], verbose=0)
plt.scatter(mu_values[:, 0], mu_values[:, 1], s=5)
plt.title("VAE 2D Latent Space")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()

def make_generator():
    return tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
        layers.Reshape((28, 28, 1))
    ])

def make_discriminator():
    return tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

generator = make_generator()
discriminator = make_discriminator()
discriminator.compile(
    optimizer="adam", loss="binary_crossentropy"
)

discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer="adam", loss="binary_crossentropy")

real = X[:5000]
d_losses, g_losses = [], []

for epoch in range(5):
    dl, gl = [], []

    for i in range(0, len(real), 128):
        batch_real = real[i:i + 128]
        n = len(batch_real)
        noise = np.random.normal(size=(n, 100))
        fake = generator.predict(noise, verbose=0)

        discriminator.trainable = True
        d1 = discriminator.train_on_batch(
            batch_real, np.ones((n, 1))
        )
        d2 = discriminator.train_on_batch(
            fake, np.zeros((n, 1))
        )

        discriminator.trainable = False
        noise = np.random.normal(size=(n, 100))
        g = gan.train_on_batch(noise, np.ones((n, 1)))

        dl.append((d1 + d2) / 2)
        gl.append(g)

    d_losses.append(np.mean(dl))
    g_losses.append(np.mean(gl))
    print("Epoch:", epoch + 1,
          "D Loss:", round(d_losses[-1], 4),
          "G Loss:", round(g_losses[-1], 4))

noise = np.random.normal(size=(10, 100))
generated = generator.predict(noise, verbose=0)

plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated[i].squeeze(), cmap="gray")
    plt.axis("off")
plt.suptitle("GAN Generated Images")
plt.show()

plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.show()

print("VAE and GAN training completed.")
