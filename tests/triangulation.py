"""
Point triangulation
-------------------

Often, we only know the distances between wells within a well base field campaign.
To retrieve their spatial positions, we provide a routine, that triangulates
their positions from a given distance matrix.

If the solution is not unique, all possible constellations will be returned.
"""

import networkx as nx
import matplotlib.pyplot as plt

import numpy as np

from welltestpy.tools import plot_well_pos, sym, triangulate

dist_mat = np.zeros((4, 4), dtype=float)
dist_mat[0, 1] = 3  # distance between well 0 and 1
dist_mat[0, 2] = 4  # distance between well 0 and 2
dist_mat[0, 3] = 1  # distance between well 0 and 3
dist_mat[1, 2] = 2  # distance between well 1 and 2
dist_mat[1, 3] = 3  # distance between well 1 and 3
dist_mat[2, 3] = -1  # unknown distance between well 2 and 3
# dist_mat[0, 4] = 2
# dist_mat[1, 4] = 3
# dist_mat[2, 4] = -1
# dist_mat[3, 4] = -1
dist_mat = sym(dist_mat)  # make the distance matrix symmetric

print(dist_mat)

well_const = triangulate(dist_mat, prec=0.5)

print(well_const)

# well_const is a list of possible graphs
# each graph is a list of coordinates
# plot those coordinates in different subplots

fig, axs = plt.subplots(1, len(well_const), figsize=(12, 4))

for i, const in enumerate(well_const):
    for j, coord in enumerate(const):
        axs[i].set_aspect('equal')
        axs[i].set_xlim(left=-5, right=5)
        axs[i].set_ylim(bottom=-5, top=5)
        axs[i].plot(coord[0], coord[1], 'o', label=f'well {j}')


plt.show()


