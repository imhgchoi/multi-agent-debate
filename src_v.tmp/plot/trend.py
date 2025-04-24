import numpy as np
import matplotlib.pyplot as plt

# Provided datasets
datasets = {
    'Arithmetics': np.array([
        [0.81, 0.87, 0.95, 0.99, 0.97],
        [0.71, 0.77, 0.89, 0.91, 0.97],
        [0.68, 0.68, 0.86, 0.89, 0.89],
        [0.67, 0.65, 0.81, 0.90, 0.88],
        [0.63, 0.68, 0.81, 0.86, 0.86],
        [0.69, 0.72, 0.82, 0.87, 0.83]
    ]),
    'GSM8K': np.array([
        [0.8000, 0.8733, 0.9167, 0.9333, 0.9300],
        [0.7267, 0.9000, 0.8933, 0.9167, 0.9133],
        [0.7200, 0.8733, 0.8867, 0.8900, 0.8800],
        [0.7400, 0.8533, 0.8467, 0.8967, 0.8633],
        [0.7067, 0.8433, 0.8667, 0.8900, 0.8700],
        [0.6967, 0.8333, 0.8467, 0.8900, 0.8433]
    ]),
    'Pro.Medicine': np.array([
        [0.7831, 0.7868, 0.7904, 0.7610, 0.8162],
        [0.7868, 0.7721, 0.7868, 0.8051, 0.8272],
        [0.7831, 0.7537, 0.7904, 0.8015, 0.7941],
        [0.7610, 0.7426, 0.7904, 0.7831, 0.7868],
        [0.7610, 0.7353, 0.7868, 0.7868, 0.7978],
        [0.7537, 0.7316, 0.7684, 0.7757, 0.7904]
    ]),
    # 'Formal Logic': np.array([
    #     [0.4524, 0.3889, 0.4365, 0.5079, 0.4921],
    #     [0.4524, 0.3889, 0.4841, 0.5000, 0.4762],
    #     [0.4683, 0.4127, 0.4524, 0.4921, 0.4603],
    #     [0.4603, 0.4524, 0.4603, 0.4921, 0.4603],
    #     [0.4762, 0.4286, 0.4524, 0.4524, 0.4524],
    #     [0.4524, 0.4444, 0.4762, 0.4683, 0.4683]
    # ]),
    'HellaSwag': np.array([
        [0.8067, 0.8100, 0.8533, 0.8200, 0.8367],
        [0.7967, 0.8000, 0.8267, 0.8000, 0.8200],
        [0.7967, 0.7967, 0.8167, 0.7967, 0.8333],
        [0.7833, 0.7767, 0.8100, 0.8033, 0.8167],
        [0.7733, 0.7800, 0.7867, 0.7900, 0.8267],
        [0.7733, 0.7467, 0.7933, 0.7600, 0.7967]
    ])
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors for consistency
colors = ['#1f3b75', '#69b3d0', '#7f7f7f', '#f5dc73', '#a8b0bf']

# Plot 1: Average across rows (dimension 0), i.e., per debate round
for (label, data), color in zip(datasets.items(), colors):
    data = data[:, [0, 2, 4]]
    means = data.mean(axis=0)
    stds = data.std(axis=0) / np.sqrt(6)
    axes[0].errorbar(range(1, 6, 2), means, yerr=stds, label=label, marker='o',
                     capsize=4, color=color, linewidth=2.5)

axes[0].set_xlabel('Number of Agents', fontsize=20)
axes[0].set_ylabel('Accuracy', fontsize=20)
axes[0].set_xticks([1, 3, 5])
axes[0].tick_params(axis='both', labelsize=15)
axes[0].legend(fontsize=20)

# Plot 2: Average across columns (dimension 1), i.e., per agent
for (label, data), color in zip(datasets.items(), colors):
    data = data[:, [0, 2, 4]]
    means = data.mean(axis=1)
    stds = data.std(axis=1) / np.sqrt(5)
    axes[1].errorbar(range(6), means, yerr=stds, label=label, marker='o',
                     capsize=4, color=color, linewidth=2.5)

axes[1].set_xlabel('Debate Rounds', fontsize=20)
# axes[1].set_ylabel('Normalized Accuracy', fontsize=20)
axes[1].set_xticks([0, 1, 2, 3, 4, 5])
axes[1].tick_params(axis='both', labelsize=15)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.savefig("out/trend.png")
plt.savefig('out/trend.pdf', format='pdf', bbox_inches='tight')