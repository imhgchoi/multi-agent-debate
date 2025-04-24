import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function p_t(alpha, beta) with epsilon = 0
def p_t(alpha, beta, t, p0, epsilon=0):
    # Check the condition 1 - alpha + beta
    condition = 1 - alpha + beta
    
    # Initialize the output array with the same shape as alpha and beta
    pt = np.zeros_like(alpha)
    
    # Case 1: 1 - alpha + beta != 0
    mask = np.abs(condition) > 1e-6  # Avoid division by zero with a small threshold
    term1 = (p0 - (beta + epsilon) / condition) * (alpha - beta)**t
    term2 = (beta + epsilon) / condition
    pt[mask] = term1[mask] + term2[mask]
    
    # Case 2: 1 - alpha + beta = 0
    mask_special = np.abs(condition) <= 1e-6
    pt[mask_special] = p0 + t * (beta[mask_special] + epsilon)
    
    return pt

# Parameters
p0_values = [0.6, 0.7, 0.8, 0.9]  # Different p0 values
t_values = [1, 2, 3, 4, 5]         # Different t values
epsilon = 0                        # Set epsilon to 0

# Create a grid of alpha and beta values with higher resolution
alpha = np.linspace(0.01, 0.99, 200)  # Increased resolution
beta = np.linspace(0.01, 0.99, 200)   # Increased resolution
Alpha, Beta = np.meshgrid(alpha, beta)  # Alpha on x-axis, Beta on y-axis

# Create a figure with subplots (5 rows for t, 4 columns for p0)
fig = plt.figure(figsize=(20, 25))  # Adjust figure size for clarity

# Loop over t and p0 to create each subplot
for i, t in enumerate(t_values):
    for j, p0 in enumerate(p0_values):
        # Compute p_t for the current t and p0
        Z = p_t(Alpha, Beta, t, p0, epsilon)
        
        # Create a subplot (5 rows, 4 columns)
        ax = fig.add_subplot(5, 4, i * 4 + j + 1, projection='3d')
        
        # Plot the surface (Alpha on x-axis, Beta on y-axis)
        surface = ax.plot_surface(Alpha, Beta, Z, cmap='viridis', edgecolor='none', alpha=0.95)
        
        # Plot a transparent hyperplane at z = p0
        Z_plane = np.full_like(Alpha, p0)  # Constant z = p0
        ax.plot_surface(Alpha, Beta, Z_plane, color='red', alpha=0.2, label=f'$p_t = p_0$')
        
        # Compute the intersection line analytically: beta = (1 - alpha) * p0 / (1 - p0)
        # Find the range of alpha where beta is within [0.01, 0.99]
        alpha_lower = 1 - 0.99 * (1 - p0) / p0
        alpha_upper = 1 - 0.01 * (1 - p0) / p0
        
        # Ensure alpha bounds are within [0.01, 0.99]
        alpha_lower = max(0.01, alpha_lower)
        alpha_upper = min(0.99, alpha_upper)
        
        # Create alpha values within the computed bounds
        alpha_line = np.linspace(alpha_lower, alpha_upper, 200)
        beta_line = (1 - alpha_line) * p0 / (1 - p0)
        
        # The intersection with the hyperplane z = p0 occurs at z = p0
        z_line = np.full_like(alpha_line, p0)
        
        # Plot the intersection line
        ax.plot(alpha_line, beta_line, z_line, color='red', linewidth=5, label='Intersection')
        
        # Set labels and title
        ax.set_xlabel(r'$\alpha$')  # x-axis is alpha
        ax.set_ylabel(r'$\beta$')  # y-axis is beta
        ax.set_zlabel(f'$p_{{{t}}}$')
        ax.set_title(f'$t={t}$, $p_0={p0}$')
        ax.view_init(elev=30, azim=120)
        
        # Reverse the beta (y-axis) direction
        ax.invert_yaxis()
        
        # Set z-axis limits to be bounded between 0 and 1
        ax.set_zlim(0, 1)
        
        # Add legend
        ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig("./theorem2.png")