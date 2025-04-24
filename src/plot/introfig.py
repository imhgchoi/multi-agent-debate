import pickle, random, collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm



datasets = {
    'Arithmetics': np.array([
        0.81, 0.97, 0.96
    ]),
    'GSM8K': np.array([
        0.80, 0.93, 0.9133
    ]),
    'Pro.Medicine': np.array([
        0.7831, 0.8162, 0.8125
    ]),
    'HellaSwag': np.array([
        0.8067, 0.8367, 0.8333
    ]),
    'CSQA': np.array([
        0.7367, 0.7933, 0.7800 
    ]),
    'Formal Logic': np.array([
        0.4524, 0.4921, 0.5079
    ]),
    'HH-RLHF': np.array([
        0.45, 0.50,  0.4733
    ]),
}

labels = list(datasets.keys())
data = np.array(list(datasets.values()))  # shape: (num_datasets, 3)

x = np.arange(len(labels))  # label locations
width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each group
bar1 = ax.bar(x - width, data[:, 0], width, label='Single Agent', hatch='//', edgecolor='white', color='#1f3b75')
bar2 = ax.bar(x, data[:, 2], width, hatch='\\', edgecolor='white', label='Best MAD', color='#88aabb')# #69b3d0
bar3 = ax.bar(x + width, data[:, 1], width,  label='Majority Voting', color='#f5dc73')#a8b0bf

# Add labels, title, legend
ax.set_ylabel('Accuracy', fontsize=25)
ax.set_xticks(x)
ax.tick_params(axis='y', labelsize=20)
ax.set_xticklabels(labels, rotation=30, fontsize=20)
ax.legend(fontsize=20)
ax.set_ylim([0.2, 1])

plt.tight_layout()
plt.savefig("out/introfig.png")
plt.savefig('out/introfig.pdf', format='pdf', bbox_inches='tight')
