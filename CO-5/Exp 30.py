import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.linalg import sqrtm
import pandas as pd
import time

np.random.seed(42)
tf.random.set_seed(42)

(X, _), _ = fashion_mnist.load_data()
X = X.astype("float32") / 255.0
X = X[:1000]
X = X[..., np.newaxis]

latent_dim = 16

encoder_input = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_input)
x = layers.Dense(128, activation="relu")(x)
mu = layers.Dense(latent_dim)(x)
log_var = layers.Dense(latent_dim)(x)

def sample(args):
    mu, log_var = args
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps

z = layers.Lambda(sample)([mu, log_var])
encoder = Model(encoder_input, [mu, log_var, z])

decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(decoder_input)
x = layers.Dense(784, activation="sigmoid")(x)
decoder_output = layers.Reshape((28, 28, 1))(x)
decoder = Model(decoder_input, decoder_output)

class VAE(Model):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(data)
            reconstructed = self.decoder(z)
            rec_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        data, reconstructed
                    ), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var - tf.square(mu) - tf.exp(log_var),
                    axis=1
                )
            )
            loss = rec_loss + kl_loss

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )
        return {"loss": loss}

vae = VAE()
vae.compile(optimizer="adam")
vae.fit(X, epochs=3, batch_size=64, verbose=1)

def make_generator():
    return tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
        layers.Reshape((28, 28, 1))
    ])

def make_discriminator():
    return tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
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

d_losses, g_losses = [], []
start = time.time()

for epoch in range(3):
    dl, gl = [], []

    for i in range(0, len(X), 64):
        real = X[i:i + 64]
        n = len(real)
        noise = np.random.normal(size=(n, 100))
        fake = generator.predict(noise, verbose=0)

        discriminator.trainable = True
        d1 = discriminator.train_on_batch(real, np.ones((n, 1)))
        d2 = discriminator.train_on_batch(fake, np.zeros((n, 1)))

        discriminator.trainable = False
        noise = np.random.normal(size=(n, 100))
        g = gan.train_on_batch(noise, np.ones((n, 1)))

        dl.append((d1 + d2) / 2)
        gl.append(g)

    d_losses.append(np.mean(dl))
    g_losses.append(np.mean(gl))
    print("GAN Epoch:", epoch + 1)

print("GAN training time:", round(time.time() - start, 2), "seconds")

vae_samples = decoder.predict(
    np.random.normal(size=(1000, latent_dim)), verbose=0
)
gan_samples = generator.predict(
    np.random.normal(size=(1000, 100)), verbose=0
)

def features(images):
    return images.reshape(len(images), -1)

def inception_score(samples, classifier):
    probabilities = classifier.predict(samples, verbose=0)
    marginal = probabilities.mean(axis=0)
    kl = probabilities * (
        np.log(probabilities + 1e-10) -
        np.log(marginal + 1e-10)
    )
    return np.exp(np.mean(np.sum(kl, axis=1)))

classifier = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])
classifier.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

(X_train, y_train), _ = fashion_mnist.load_data()
X_train = X_train[:10000].astype("float32") / 255
X_train = X_train[..., np.newaxis]
classifier.fit(X_train, y_train[:10000], epochs=2, verbose=0)

is_vae = inception_score(vae_samples, classifier)
is_gan = inception_score(gan_samples, classifier)

real_features = features(X[:500])
vae_features = features(vae_samples[:500])
gan_features = features(gan_samples[:500])

def fid(real, generated):
    m1, m2 = real.mean(axis=0), generated.mean(axis=0)
    c1, c2 = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
    covmean = sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum((m1 - m2) ** 2) + np.trace(c1 + c2 - 2 * covmean)

real_pca = PCA(n_components=2).fit_transform(real_features)
vae_pca = PCA(n_components=2).fit_transform(vae_features)
gan_pca = PCA(n_components=2).fit_transform(gan_features)

results = pd.DataFrame({
    "Model": ["VAE", "GAN"],
    "Inception Score": [is_vae, is_gan],
    "FID": [
        fid(real_features, vae_features),
        fid(real_features, gan_features)
    ],
    "Unique Pixel Patterns": [
        len(np.unique(vae_samples.round(1).reshape(1000, -1), axis=0)),
        len(np.unique(gan_samples.round(1).reshape(1000, -1), axis=0))
    ]
})

print(results)

plt.figure(figsize=(10, 4))
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(vae_samples[i].squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("VAE")

    plt.subplot(2, 5, i + 6)
    plt.imshow(gan_samples[i].squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("GAN")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(real_pca[:, 0], real_pca[:, 1], label="Real", alpha=0.5)
plt.scatter(vae_pca[:, 0], vae_pca[:, 1], label="VAE", alpha=0.5)
plt.scatter(gan_pca[:, 0], gan_pca[:, 1], label="GAN", alpha=0.5)
plt.title("Real and Generated Feature Comparison")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()

print("Experiment 30 completed.")
