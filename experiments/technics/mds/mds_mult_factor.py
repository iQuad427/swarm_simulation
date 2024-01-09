import random

import numpy as np

from experiments.utils.compute import compute_area
from experiments.utils.generate_matrix import generate_points_and_matrix
from sklearn.manifold import MDS

from experiments.utils.plot import plot_side_by_side, plot_on_top
from experiments.utils.rotation import find_rotation_matrix

if __name__ == '__main__':
    # Generate a set of 16 points
    points, matrix = generate_points_and_matrix(16, 300, 30)

    # Plot the points
    # plot_points(points, matrix)

    random.seed(0)

    # Perform classical multidimensional scaling
    mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, metric=True)
    mds_coors = mds.fit_transform(matrix)

    # Plot the results side by side
    plot_side_by_side(points, mds_coors)

    # Find the transformation matrix to put MDS points on the same scale as the original points
    # We do this by finding the best rotation and translation to minimize the sum of squared errors
    # between the original points and the MDS points

    rotation = find_rotation_matrix(points.T, mds_coors.T)

    # Apply the rotation
    mds_coors_rotated = mds_coors @ rotation

    # Plot the results side by side
    plot_side_by_side(points, mds_coors_rotated)

    # First, find the centroid of the original points
    centroid = np.mean(points, axis=0)

    # Then, find the centroid of the MDS points
    mds_centroid = np.mean(mds_coors_rotated, axis=0)

    # Find the translation vector
    translation = centroid - mds_centroid

    # Translate the MDS points
    mds_coors_translated = mds_coors_rotated + translation

    # Plot the results side by side
    plot_side_by_side(points, mds_coors_translated)

    print(compute_area(points))
    print(compute_area(mds_coors))
    print(compute_area(mds_coors_rotated))
    print(compute_area(mds_coors_translated))

    plot_on_top(points, mds_coors_translated)

