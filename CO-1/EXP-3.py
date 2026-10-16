import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, uniform, norm

np.random.seed(42)

x = np.arange(0, 11)

plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.bar([0, 1], bernoulli.pmf([0, 1], 0.5))
plt.title("Bernoulli Distribution")

plt.subplot(2, 3, 2)
plt.bar(x, binom.pmf(x, 10, 0.5))
plt.title("Binomial Distribution")

plt.subplot(2, 3, 3)
plt.bar(x, poisson.pmf(x, 4))
plt.title("Poisson Distribution")

plt.subplot(2, 3, 4)
u = uniform.rvs(size=1000)
plt.hist(u, bins=20, density=True)
plt.title("Uniform Distribution")

plt.subplot(2, 3, 5)
g = norm.rvs(size=1000)
plt.hist(g, bins=20, density=True)
plt.title("Gaussian Distribution")

prevalence = 0.01
sensitivity = 0.95
specificity = 0.90

posterior = (sensitivity * prevalence) / (
    sensitivity * prevalence +
    (1 - specificity) * (1 - prevalence)
)

print("P(Disease | Positive):", round(posterior, 4))
print("Percentage:", round(posterior * 100, 2), "%")

n = 100000
disease = np.random.rand(n) < prevalence
positive = np.where(
    disease,
    np.random.rand(n) < sensitivity,
    np.random.rand(n) < (1 - specificity)
)

print("Simulated Probability:",
      round(positive[disease].size / max(positive.sum(), 1), 4))

prev = np.linspace(0.001, 0.20, 100)
post = (sensitivity * prev) / (
    sensitivity * prev + (1 - specificity) * (1 - prev)
)

plt.figure(figsize=(7, 4))
plt.plot(prev * 100, post * 100)
plt.xlabel("Disease Prevalence (%)")
plt.ylabel("Posterior Probability (%)")
plt.title("Bayes Theorem")
plt.grid()
plt.show()
