from numpy.linalg import svd
import numpy as np


def find_rotation_matrix(X, Y):
    """
    Find the rotation matrix between two point clouds using SVD.

    Parameters:
    - X: Numpy array representing the first point cloud (3xN matrix).
    - Y: Numpy array representing the second point cloud (3xN matrix).

    Returns:
    - R: 3x3 rotation matrix.
    """

    # Center the point clouds
    center_X = X.mean(axis=1).reshape(-1, 1)
    center_Y = Y.mean(axis=1).reshape(-1, 1)

    X_centered = X - center_X
    Y_centered = Y - center_Y

    # Compute the covariance matrix H
    H = X_centered @ Y_centered.T

    # Perform SVD on H
    U, _, Vt = svd(H)

    # Compute the rotation matrix R
    R = Vt.T @ U.T

    return R


def rotate_and_translate(points, mds_coors):
    rotation = find_rotation_matrix(points.T, mds_coors.T)

    # Apply the rotation
    mds_coors_rotated = mds_coors @ rotation

    # First, find the centroid of the original points
    centroid = np.mean(points, axis=0)

    # Then, find the centroid of the MDS points
    mds_centroid = np.mean(mds_coors_rotated, axis=0)

    # Find the translation vector
    translation = centroid - mds_centroid

    # Translate the MDS points
    return mds_coors_rotated + translation
