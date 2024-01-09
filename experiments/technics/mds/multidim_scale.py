import math
import random

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS

# Create the matrix
# tmp = np.array([
#     [0, 20, 20, 20, 40, 60, 60, 60, 100, 120, 120, 120],
#     [20, 0, 20, 20, 60, 80, 80, 80, 120, 140, 140, 140],
#     [20, 20, 0, 20, 60, 80, 80, 80, 120, 140, 140, 140],
#     [20, 20, 20, 0, 60, 80, 80, 80, 120, 140, 140, 140],
#     [40, 60, 60, 60, 0, 20, 20, 20, 60, 80, 80, 80],
#     [60, 80, 80, 80, 20, 0, 20, 20, 40, 60, 60, 60],
#     [60, 80, 80, 80, 20, 20, 0, 20, 60, 80, 80, 80],
#     [60, 80, 80, 80, 20, 20, 20, 0, 60, 80, 80, 80],
#     [100, 120, 120, 120, 60, 40, 60, 60, 0, 20, 20, 20],
#     [120, 140, 140, 140, 80, 60, 80, 80, 20, 0, 20, 20],
#     [120, 140, 140, 140, 80, 60, 80, 80, 20, 20, 0, 20],
#     [120, 140, 140, 140, 80, 60, 80, 80, 20, 20, 20, 0]
# ])

# Test with a smaller distance matrix representing a square
# Example of robot positions: with distance between each number equal to 20
#
# 1  0  0  0  1
# 0  0  0  0  0
# 0  0  0  0  0
# 0  0  0  0  0
# 1  0  0  0  1

tmp = np.array([
    [0, 80, 80, math.sqrt(80 ** 2 + 80 ** 2)],
    [80, 0, math.sqrt(80 ** 2 + 80 ** 2), 80],
    [80, math.sqrt(80 ** 2 + 80 ** 2), 0, 80],
    [math.sqrt(80 ** 2 + 80 ** 2), 80, 80, 0]
])
#
# tmp = np.array([
#     [0, 5, 3],
#     [5, 0, 4],
#     [3, 4, 0]
# ])

# Convert the matrix to a distance matrix
d = distance_matrix(tmp, tmp)
random.seed(0)

# Perform classical multidimensional scaling
mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, metric=True)
mds_coors = mds.fit_transform(tmp)

# Plot the results
plt.scatter(mds_coors[:, 0], mds_coors[:, 1], s=50, alpha=0.5)
plt.axis('equal')

for i, label in enumerate(list('ABCD')):
    plt.text(mds_coors[i, 0], mds_coors[i, 1], label, fontsize=8)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('')
plt.ylabel('')

# save the plot
plt.axis('equal')
plt.savefig('mds.png', dpi=300)

plt.show()

# Use matplotlib.tri to plot the triangulation
triangulation = tri.Triangulation(mds_coors[:, 0], mds_coors[:, 1])
plt.triplot(triangulation, 'bo-', lw=1.0)
plt.axis('equal')
plt.show()

# Compute area and perimeter
area = 0

for i in range(len(triangulation.triangles)):
    print(i)
    triangle = triangulation.triangles[i]
    p1 = mds_coors[triangle[0]]
    p2 = mds_coors[triangle[1]]
    p3 = mds_coors[triangle[2]]

    print(p1, p2, p3)

    # Compute area
    #  A = (1/2) |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|
    area += abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2)


print("Area:", area)
