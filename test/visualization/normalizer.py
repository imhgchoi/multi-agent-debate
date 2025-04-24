import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# Data
X = np.array([
    # 100, 
    # 150, 
    # 200, 
    # 250,
    # 300, 
    350, 400, 450, 500, 550, 600, 650, 700])
Y = np.array([
    # 8.7576, 
    # 17.4194, 
    # 25.8881, 
    # 38.4303, 
    # 48.1695, 
    54.8359, 
    63.6375, 
    99.9087,
    106.5514, 
    124.3573, 
    140.694, 
    150.3471, 
    165.3524
])
# Transform X to ensure positive quadratic coefficient
X_transformed = X - X.min()

# Perform quadratic regression on transformed data
coefficients = np.polyfit(X_transformed, Y, 2)
quadratic_model = np.poly1d(coefficients)

# Generate fitted values
X_fit = np.linspace(X_transformed.min(), X_transformed.max(), 500)
Y_fit = quadratic_model(X_fit)

# Calculate R-squared
Y_pred = quadratic_model(X_transformed)
r_squared = r2_score(Y, Y_pred)

# Plot data and quadratic fit
plt.figure(figsize=(6, 5))
plt.scatter(X, Y, color='blue', label='Observed Data', s=40)
plt.plot(X_fit + X.min(), Y_fit, color='red', linewidth=2, label=f'Quadratic Fit\n$R^2 = {r_squared:.4f}$')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(fontsize=12)
plt.grid(True)
plt.title('Quadratic Regression with Positive Leading Coefficient')
plt.savefig('out/normalizer2.png')
plt.close()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data
X = np.array([
    # 100, 
    # 150, 
    # 200, 
    # 250, 
    # 300, 
    350, 400, 450, 500, 550, 600, 650, 700]).reshape(-1, 1)
Y = np.array([
    # 8.7576, 
    # 17.4194, 
    # 25.8881, 
    # 38.4303, 
    # 48.1695, 
    54.8359, 
    63.6375, 
    99.9087,
    106.5514, 
    124.3573, 
    140.694, 
    150.3471, 
    165.3524
])


# Perform linear regression
linear_model = LinearRegression()
linear_model.fit(X, Y)

# Generate fitted values
X_fit = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
Y_fit = linear_model.predict(X_fit)

# Calculate R-squared
Y_pred = linear_model.predict(X)
r_squared = r2_score(Y, Y_pred)

# Plot data and linear fit
plt.figure(figsize=(6, 5))
plt.scatter(X, Y, color='blue', label='Observed Data', s=40)
plt.plot(X_fit, Y_fit, color='red', linewidth=2, label=f'Linear Fit\n$R^2 = {r_squared:.4f}$')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('out/normalizer.png')