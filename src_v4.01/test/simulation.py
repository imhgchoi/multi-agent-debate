

import numpy as np
import matplotlib.pyplot as plt


# Neutral
P = np.array([[0.333, 0.333, 0.333], [0.333, 0.333, 0.333], [0.333, 0.333, 0.333]])
# Stubborn
# P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# Easily Swayed
# P = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
# Confidence
# P = np.array([[0.9, 0.1, 0.0], [0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])



# Strongs
X = np.array([[0.9],[0.9],[0.9]])
# Weaks
X = np.array([[0.3],[0.3],[0.3]])
# Mixed
X = np.array([[0.9],[0.5],[0.1]])

mean_list = []
for _ in range(100):
    X = np.matmul(P, X)
    mean_list.append(np.mean(X))
    print(np.mean(X))
print(sum(np.linalg.svd(P).S > 1))
plt.plot(mean_list)
plt.savefig('out/tmp.png')
plt.close()