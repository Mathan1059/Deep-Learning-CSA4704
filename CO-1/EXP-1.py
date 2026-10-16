import numpy as np
import matplotlib.pyplot as plt

customer_genre = np.array([
    [5, 3, 0, 1, 2],
    [4, 2, 0, 1, 1],
    [1, 1, 5, 4, 0],
    [0, 1, 4, 5, 1],
    [2, 5, 1, 0, 4],
    [1, 4, 2, 0, 5]
])

genre_book = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0]
])

scores = customer_genre @ genre_book

plt.imshow(scores, cmap="YlOrRd", aspect="auto")
plt.colorbar(label="Recommendation Score")
plt.xticks(range(4), ["Book 1", "Book 2", "Book 3", "Book 4"])
plt.yticks(range(6), [
    "Customer 1", "Customer 2", "Customer 3",
    "Customer 4", "Customer 5", "Customer 6"
])

for i in range(6):
    for j in range(4):
        plt.text(j, i, str(scores[i, j]),
                 ha="center", va="center")

plt.title("Customer-Book Recommendation Scores")
plt.xlabel("Books")
plt.ylabel("Customers")
plt.tight_layout()
plt.show()
