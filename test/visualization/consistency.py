import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Min-K%": 0.3789,
    "Min-K% + FSD": 0.4387,
    "Perplexity Score": 0.4938,
    "Perplexity Score + FSD": 0.4345,
    "SRCT": 0.2777,
    "KDE KLD": 0.2765,
    "Dot-product Kernel MSE": 0.2676,
    "Cosine Similarity Kernel MSE": 0.2750,
    "RBF Kernel MSE": 0.2762,
    "MMD": 0.3048,
    "CKA": 0.3048,
    "Kernel Divergence Score": 0.2366
}

data = {
    "Min-K%": 0.3177,
    "Min-K% + FSD": 0.3591,
    "Perplexity Score": 0.316,
    "Perplexity Score + FSD": 0.3595,
    "SRCT": 0.5299,
    "KDE KLD": 0.8814,
    "Dot-product Kernel MSE": 0.8006,
    "Cosine Similarity Kernel MSE": 0.7934,
    "RBF Kernel MSE": 0.7893,
    "MMD": 0.916,
    "CKA": 0.7507,
    "Kernel Divergence Score": 0.2957
}


# Separate data into labels and values
labels = list(data.keys())
values = list(data.values())

# Define colors (distinct color for Kernel Divergence Score)
colors = ["#89C2D9"] * 5 + ["#168AAD"] * 6 + ["#38419D"]

# Create figure and axes
plt.figure(figsize=(15, 10))

# Plot bar chart
bars = plt.bar(labels, values, color=colors, zorder=3)




# Add data values above each bar
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Center of the bar
        bar.get_height() + 0.01,  # Slightly above the bar
        str(f"{value:.4f}")[1:],  # Value text
        ha="center",  # Horizontal alignment
        va="bottom",  # Vertical alignment
        fontsize=25
    )

# Add titles and labels
plt.ylabel("Average MAPE", fontsize=30)
plt.xticks(rotation=30, ha="right", fontsize=25)
plt.ylim(0, max(values) + 0.05)

# Add grid lines
plt.grid(axis="y", linestyle="--", linewidth=1, alpha=0.9, zorder=0)

# Add legend for bar colors
plt.legend([
    plt.Rectangle((0, 0), 1, 1, color="#89C2D9"),
    plt.Rectangle((0, 0), 1, 1, color="#168AAD"),
    plt.Rectangle((0, 0), 1, 1, color="#38419D")
], [
    "Detector-based",
    "Kernel-based",
    "Ours"
], loc="best", fontsize=25)

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('out/consistency.png')