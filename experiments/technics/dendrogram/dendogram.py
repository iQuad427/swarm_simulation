from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Create the matrix
tmp = np.array([
    [0, 20, 20, 40, 60, 60, 100, 120, 120],
    [20, 0, 20, 60, 80, 80, 120, 140, 140],
    [20, 20, 0, 60, 80, 80, 120, 140, 140],
    [40, 60, 60, 0, 20, 20, 60, 80, 80],
    [60, 80, 80, 20, 0, 20, 60, 80, 80],
    [60, 80, 80, 20, 20, 0, 60, 80, 80],
    [100, 120, 120, 60, 40, 60, 0, 20, 20],
    [120, 140, 140, 80, 60, 80, 20, 0, 20],
    [120, 140, 140, 80, 60, 80, 20, 20, 0],
])

# Make the matrix symmetric
tmp = tmp + tmp.T - np.diag(tmp.diagonal())

# Perform hierarchical clustering
linkage_matrix = linkage(tmp, method='single', metric='euclidean')

print(linkage_matrix)

# Plot the dendrogram
dendrogram(linkage_matrix, labels=list('ABCDEFGHI'), orientation='right')
plt.xlabel('Distance')
plt.ylabel('Agent')
plt.title('Hierarchical Clustering Dendrogram')

# save the plot
plt.savefig('dendrogram.png', dpi=300)

plt.show()

render_matrix = False
if render_matrix:
    mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, metric=True)
    mds_coors = mds.fit_transform(tmp)

    # Plot the results
    plt.scatter(mds_coors[:, 0], mds_coors[:, 1], s=50, alpha=0.5)
    plt.axis('equal')

    for i, label in enumerate(list('ABCDEFGHI')):
        plt.text(mds_coors[i, 0], mds_coors[i, 1], label, fontsize=8)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('')
    plt.ylabel('')

    # save the plot
    plt.axis('equal')
    plt.savefig('mds.png', dpi=300)

    plt.show()
