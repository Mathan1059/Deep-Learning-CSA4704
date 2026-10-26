import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

tf.random.set_seed(42)
np.random.seed(42)

words = ["I", "love", "deep", "learning", "today"]
X = np.random.rand(1, 5, 4).astype("float32")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5, 4)),
    tf.keras.layers.SimpleRNN(3, return_sequences=True),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.predict(X, verbose=0)

rnn = model.layers[0]
dense = model.layers[1]

Wx, Wh, bh = rnn.get_weights()
Wy, by = dense.get_weights()

hidden = np.zeros(3)
states = []
outputs = []

for t in range(5):
    hidden = np.tanh(X[0, t] @ Wx + hidden @ Wh + bh)
    states.append(hidden.copy())
    outputs.append(hidden @ Wy + by)

states = np.array(states)
outputs = np.array(outputs).flatten()

keras_states = tf.keras.Model(
    inputs=model.input,
    outputs=rnn.output
).predict(X, verbose=0)[0]

print("Manual hidden states:")
print(np.round(states, 3))

print("Keras hidden states:")
print(np.round(keras_states, 3))

print("States match:", np.allclose(states, keras_states, atol=1e-5))

G = nx.DiGraph()

for t in range(5):
    G.add_node(f"x{t+1}", color="lightblue")
    G.add_node(f"h{t+1}", color="lightgreen")
    G.add_node(f"y{t+1}", color="salmon")

    G.add_edge(f"x{t+1}", f"h{t+1}", label="Wx")
    G.add_edge(f"h{t+1}", f"y{t+1}", label="Wy")

    if t > 0:
        G.add_edge(f"h{t}", f"h{t+1}", label="Wh")

pos = {}
for t in range(5):
    pos[f"x{t+1}"] = (t, 2)
    pos[f"h{t+1}"] = (t, 1)
    pos[f"y{t+1}"] = (t, 0)

colors = [G.nodes[n]["color"] for n in G.nodes]

plt.figure(figsize=(12, 5))
nx.draw(
    G, pos, with_labels=True, node_color=colors,
    node_size=1200, arrows=True
)
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=nx.get_edge_attributes(G, "label"),
    font_size=8
)
plt.title("Unfolded RNN Architecture")
plt.show()

plt.figure(figsize=(8, 4))
for i in range(3):
    plt.plot(range(1, 6), states[:, i], marker="o",
             label=f"Hidden Unit {i+1}")

plt.xticks(range(1, 6), words)
plt.xlabel("Input Word")
plt.ylabel("Hidden State Value")
plt.title("Hidden States Across Time")
plt.legend()
plt.grid()
plt.show()
