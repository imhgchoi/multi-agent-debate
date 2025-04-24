import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Fixed parameters
p0 = 0.8
n = 10
T = np.arange(0, 10, 1)

# Define more sweep values
lambda_values = np.linspace(0.3, 0.8, 6)
gamma_values = np.linspace(0.1, 0.35, 6)

# Blue to yellow gradient (cividis colormap)
colors_left = cm.cividis(np.linspace(0.2, 0.9, len(lambda_values)))
colors_right = cm.cividis(np.linspace(0.2, 0.9, len(gamma_values)))

# Prepare subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# Majority vote bound
mv_bound = 1 - np.exp(-2 * n * (p0 - 0.5)**2)

# Left plot: Vary lambda (fixed gamma)
fixed_gamma_left = 0.15
axes[0].hlines(mv_bound, xmin=0, xmax=T[-1], linestyles='dashed', colors='black',
               linewidth=3, label='Majority Vote')
for i, lam in enumerate(lambda_values):
    delta = lam - fixed_gamma_left
    num = delta * p0
    denom = lam * p0 + (delta - lam * p0) * np.exp(-delta * T)
    pt = num / denom
    group_success = 1 - np.exp(-2 * n * (pt - 0.5)**2)
    axes[0].plot(T, group_success, label=f"$\\lambda$={lam:.2f}", color=colors_left[i], linewidth=4)
axes[0].set_title(f"Varying $\\lambda$ (Fixed $\\gamma$={fixed_gamma_left})", fontsize=20)
axes[0].set_xlabel("Debate Round $t$", fontsize=18)
axes[0].set_ylabel("Group Success Probability", fontsize=18)
axes[0].tick_params(axis='both', labelsize=16)
axes[0].legend(fontsize=14)
axes[0].grid(True)

# Right plot: Vary gamma (fixed lambda)
fixed_lambda_right = 0.7
axes[1].hlines(mv_bound, xmin=0, xmax=T[-1], linestyles='dashed', colors='black',
               linewidth=3, label='Majority Vote')
for i, gamma in enumerate(gamma_values):
    delta = fixed_lambda_right - gamma
    num = delta * p0
    denom = fixed_lambda_right * p0 + (delta - fixed_lambda_right * p0) * np.exp(-delta * T)
    pt = num / denom
    group_success = 1 - np.exp(-2 * n * (pt - 0.5)**2)
    axes[1].plot(T, group_success, label=f"$\\gamma$={gamma:.2f}", color=colors_right[i], linewidth=4)
axes[1].set_title(f"Varying $\\gamma$ (Fixed $\\lambda$={fixed_lambda_right})", fontsize=20)
axes[1].set_xlabel("Debate Round $t$", fontsize=18)
axes[1].tick_params(axis='both', labelsize=16)
axes[1].legend(fontsize=14)
axes[1].grid(True)

plt.suptitle(f"Debate vs. Majority Vote (Fixed $p_0$={p0}, $N$={n})", fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save to PDF
output_path = "/mnt/data/debate_vs_majority_vote_cividis.pdf"
plt.savefig(output_path, format="pdf")
# output_path