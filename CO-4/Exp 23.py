import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ("Smoking", "LungDisease"),
    ("Pollution", "LungDisease"),
    ("LungDisease", "Cough"),
    ("LungDisease", "Xray"),
    ("LungDisease", "Fatigue")
])

cpd_smoking = TabularCPD("Smoking", 2, [[0.7], [0.3]])
cpd_pollution = TabularCPD("Pollution", 2, [[0.8], [0.2]])

cpd_disease = TabularCPD(
    "LungDisease", 2,
    [[0.99, 0.9, 0.9, 0.2],
     [0.01, 0.1, 0.1, 0.8]],
    evidence=["Smoking", "Pollution"],
    evidence_card=[2, 2]
)

cpd_cough = TabularCPD(
    "Cough", 2,
    [[0.8, 0.3], [0.2, 0.7]],
    evidence=["LungDisease"], evidence_card=[2]
)

cpd_xray = TabularCPD(
    "Xray", 2,
    [[0.9, 0.2], [0.1, 0.8]],
    evidence=["LungDisease"], evidence_card=[2]
)

cpd_fatigue = TabularCPD(
    "Fatigue", 2,
    [[0.7, 0.2], [0.3, 0.8]],
    evidence=["LungDisease"], evidence_card=[2]
)

model.add_cpds(
    cpd_smoking, cpd_pollution, cpd_disease,
    cpd_cough, cpd_xray, cpd_fatigue
)

print("Network Valid:", model.check_model())

G = nx.DiGraph(model.edges())
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G, pos, with_labels=True, node_color="lightblue",
    node_size=2500, arrows=True, font_size=9
)
plt.title("Medical Bayesian Network")
plt.show()

infer = VariableElimination(model)

print("\nProbability of Lung Disease:")
print(infer.query(["LungDisease"]))

print("\nDisease Probability given Cough:")
print(infer.query(
    ["LungDisease"], evidence={"Cough": 1}
))

print("\nIndependencies:")
print(model.get_independencies())

print("\nActive Trail from Smoking to Cough:")
print(model.active_trail_nodes("Smoking"))
