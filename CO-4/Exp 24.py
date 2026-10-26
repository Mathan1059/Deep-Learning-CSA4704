import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

model = DiscreteBayesianNetwork([
    ("Smoking", "Disease"),
    ("Pollution", "Disease"),
    ("Disease", "Cough")
])

model.add_cpds(
    TabularCPD("Smoking", 2, [[0.7], [0.3]]),
    TabularCPD("Pollution", 2, [[0.8], [0.2]]),
    TabularCPD("Disease", 2,
               [[0.99, 0.9, 0.9, 0.2],
                [0.01, 0.1, 0.1, 0.8]],
               evidence=["Smoking", "Pollution"],
               evidence_card=[2, 2]),
    TabularCPD("Cough", 2,
               [[0.8, 0.3], [0.2, 0.7]],
               evidence=["Disease"], evidence_card=[2])
)

print("Valid Network:", model.check_model())

G = nx.DiGraph(model.edges())
nx.draw(G, with_labels=True, node_color="lightblue",
        node_size=2000, arrows=True)
plt.title("Bayesian Network")
plt.show()

infer = VariableElimination(model)
exact = infer.query(["Disease"], evidence={"Cough": 1})
print("Exact Probability:")
print(exact)

sampler = BayesianModelSampling(model)
sizes = [100, 500, 1000, 5000]
estimates = []

for n in sizes:
    samples = sampler.forward_sample(size=n, show_progress=False)
    cough_samples = samples[samples["Cough"] == 1]
    probability = (cough_samples["Disease"] == 1).mean()
    estimates.append(probability)

results = pd.DataFrame({
    "Sample Size": sizes,
    "Estimated P(Disease=1 | Cough=1)": estimates
})

print(results.round(4))

plt.plot(sizes, estimates, marker="o", label="Sampling Estimate")
plt.axhline(float(exact.values[1]), linestyle="--",
            label="Exact Probability")
plt.xlabel("Number of Samples")
plt.ylabel("Probability")
plt.title("Exact vs Approximate Inference")
plt.legend()
plt.grid()
plt.show()
