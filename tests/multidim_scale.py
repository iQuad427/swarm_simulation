import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Create the matrix
tmp = np.array([
    [0, 20, 20, 20, 40, 60, 60, 60, 100, 120, 120, 120],
    [20, 0, 20, 20, 60, 80, 80, 80, 120, 140, 140, 140],
    [20, 20, 0, 20, 60, 80, 80, 80, 120, 140, 140, 140],
    [20, 20, 20, 0, 60, 80, 80, 80, 120, 140, 140, 140],
    [40, 60, 60, 60, 0, 20, 20, 20, 60, 80, 80, 80],
    [60, 80, 80, 80, 20, 0, 20, 20, 40, 60, 60, 60],
    [60, 80, 80, 80, 20, 20, 0, 20, 60, 80, 80, 80],
    [60, 80, 80, 80, 20, 20, 20, 0, 60, 80, 80, 80],
    [100, 120, 120, 120, 60, 40, 60, 60, 0, 20, 20, 20],
    [120, 140, 140, 140, 80, 60, 80, 80, 20, 0, 20, 20],
    [120, 140, 140, 140, 80, 60, 80, 80, 20, 20, 0, 20],
    [120, 140, 140, 140, 80, 60, 80, 80, 20, 20, 20, 0]
])

# Convert the matrix to a distance matrix
d = distance_matrix(tmp, tmp)

# Perform classical multidimensional scaling
mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False)
mds_coors = mds.fit_transform(d)

# Plot the results
plt.scatter(mds_coors[:, 0], mds_coors[:, 1], s=50, alpha=0.5)
for i, label in enumerate(list('ABCDEFGHIJKL')):
    plt.text(mds_coors[i, 0], mds_coors[i, 1], label, fontsize=8)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('')
plt.ylabel('')
plt.show()