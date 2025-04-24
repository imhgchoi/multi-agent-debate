import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(n, p)
def f(n, p):
    return 1 - np.exp(-2 * n * (p - 0.5)**2)

# Create a grid of n and p values
n = np.linspace(1, 10, 10)  # n from 0 to 5
p = np.linspace(0.5, 1, 100)  # p from -2 to 2
N, P = np.meshgrid(n, p)  # Create a meshgrid for n and p

# Compute f(n, p) for each point in the grid
Z = f(N, P)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(N, P, Z, cmap='viridis', edgecolor='none')

# Add a color bar
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='f(n, p)')

# Set labels and title
ax.set_xlabel('n')
ax.set_ylabel('p')
ax.set_zlabel('f(n, p)')
ax.set_title('3D Plot of f(n, p) = 1 - exp(-2n(p + 1/2)^2)')

# Show the plot
plt.savefig("./theorem1.png")