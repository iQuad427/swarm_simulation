from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from scipy.spatial import distance_matrix
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

# Calculate the distance matrix
d = distance_matrix(1 - tmp, 1 - tmp)

# Perform hierarchical clustering
linkage_matrix = linkage(d, method='single')

# Plot the dendrogram
dendrogram(linkage_matrix, labels=list('ABCDEFGHIJKL'), orientation='right')
plt.xlabel('')
plt.ylabel('Cluster Distance')
plt.title('Hierarchical Clustering Dendrogram')

# save the plot
plt.savefig('dendrogram.png', dpi=300)

plt.show()