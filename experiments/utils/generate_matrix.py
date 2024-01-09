# Functions to generate a distance matrix between a given number of dots
import numpy as np
import matplotlib.pyplot as plt


def generate_points_cloud(n_dots: int, scaler: int) -> np.ndarray:
    """
    Generate a set of random points in a 2D space.

    :param n_dots: Number of points to generate
    :param scaler: The maximum value of the coordinates of the points
    """

    # Generate n_dots random points (x, y)
    return np.random.rand(n_dots, 2) * scaler


def generate_matrix_from_cloud(cloud: np.ndarray, noise=0.0):
    """
    Generate a distance matrix from a given set of points.

    :param cloud: The set of points
    :param noise: The maximum noise to add to the distance matrix
    """

    # Generate the distance matrix
    dist_mat = np.zeros((cloud.shape[0], cloud.shape[0]), dtype=float)

    for i in range(cloud.shape[0]):
        for j in range(cloud.shape[0]):
            dist_mat[i, j] = np.linalg.norm(cloud[i] - cloud[j])

    # Add noise to the distance matrix

    # Random noise between -1 and 1 times the noise factor
    noise_mat = np.random.rand(cloud.shape[0], cloud.shape[0]) * 2 - 1
    dist_mat += noise_mat * noise

    # Make the distance matrix symmetric
    dist_mat = (dist_mat + dist_mat.T) / 2

    # Remove the diagonal
    np.fill_diagonal(dist_mat, 0)

    return dist_mat


def generate_points_and_matrix(n_dots: int, scaler: int, noise=0.0):
    """
    Generate a set of random points in a 2D space and the associated distance matrix.

    :param n_dots: Number of points to generate
    :param scaler: The maximum value of the coordinates of the points
    :param noise: The maximum noise to add to the distance matrix
    """

    # Generate the points
    points = generate_points_cloud(n_dots, scaler)

    # Generate the distance matrix
    matrix = generate_matrix_from_cloud(points, noise)

    return points, matrix


def plot_points(points: np.ndarray, matrix: np.ndarray = None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the points on the first axis
    ax[0].set_title('Points')
    ax[0].axis('equal')

    ax[0].scatter(points[:, 0], points[:, 1], color='blue', marker='o', s=50)
    for i, point in enumerate(points):
        ax[0].annotate(str(i), (point[0], point[1]))

    # Plot the distance matrix on the second axis
    if matrix is not None:
        ax[1].imshow(matrix)
        ax[1].set_title('Distance Matrix')

    plt.show()
