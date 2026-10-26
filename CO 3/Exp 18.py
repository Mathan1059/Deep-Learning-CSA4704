import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

texts = [
    "This product is excellent and amazing",
    "I love this product very much",
    "The quality is very good",
    "This is a wonderful experience",
    "The product is useful and nice",
    "This product is terrible and bad",
    "I hate this product",
    "The quality is very poor",
    "This is a disappointing experience",
    "The product is useless and worst"
] * 100

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 100)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

def build_model(cell, units, dropout, embedding):
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Embedding(1000, embedding)(inputs)

    layer = {
        "SimpleRNN": tf.keras.layers.SimpleRNN,
        "LSTM": tf.keras.layers.LSTM,
        "GRU": tf.keras.layers.GRU
    }[cell]

    x = layer(units, dropout=dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

results = []

for cell in ["SimpleRNN", "LSTM", "GRU"]:
    best = (0, None)

    for units in [16, 32]:
        for dropout in [0.2, 0.5]:
            model = build_model(cell, units, dropout, 16)
            start = time.time()

            model.fit(
                X_train, y_train,
                epochs=3, batch_size=32,
                validation_split=0.2, verbose=0
            )

            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            duration = time.time() - start

            if accuracy > best[0]:
                best = (accuracy, {
                    "Cell": cell,
                    "Units": units,
                    "Dropout": dropout,
                    "Embedding": 16,
                    "Accuracy": accuracy,
                    "Parameters": model.count_params(),
                    "Time": duration
                })

    results.append(best[1])

print(pd.DataFrame(results).round(3))

plt.bar([r["Cell"] for r in results],
        [r["Accuracy"] for r in results])
plt.xlabel("Recurrent Cell")
plt.ylabel("Test Accuracy")
plt.title("RNN vs LSTM vs GRU")
plt.show()
