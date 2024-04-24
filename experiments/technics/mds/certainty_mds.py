import numpy as np

from experiments.utils.rotation import rotate_and_translate


def remove_uncertain_measurement(distances, certainty, threshold=0.5):
    """
    Remove uncertain measurements from the matrix of distances.

    :param distances: Matrix of distances.
    :param certainty: Certainty of each distance.
    :param threshold: Threshold below which a measurement is considered uncertain.
    :return: Matrix of distances without uncertain measurements. (replaced with -1)
    """
    # Create a copy of the matrix of distances
    distances = distances.copy()

    # Iterate over the matrix of distances
    for i in range(distances.shape[0]):
        for j in range(i, distances.shape[1]):
            # If the certainty is below the threshold, replace the distance with -1
            if i != j and certainty[i, j] < threshold:
                distances[i, j] = -1

    return distances


def remove_uncertain_agents(distances, certainty, threshold=0.5):
    """
    Remove agents with uncertain measurements from the matrix of distances.

    :param distances: Matrix of distances.
    :param certainty: Certainty of each distance.
    :param threshold: Threshold below which a measurement is considered uncertain.
    :return: Matrix of distances without uncertain agents.
    """
    # Create a copy of the matrix of distances
    distances_without_uncertainty = remove_uncertain_measurement(distances, certainty, threshold)

    to_remove = []

    # Iterate over the matrix of distances
    for i in range(distances_without_uncertainty.shape[0]):
        # If any distances for an agent are uncertain, replace the distances with -1
        if np.any(distances_without_uncertainty[:, i] == -1):
            to_remove.append(i)

    # Remove the agents with uncertain measurements
    distances_without_uncertainty = np.delete(distances_without_uncertainty, to_remove, axis=0)
    distances_without_uncertainty = np.delete(distances_without_uncertainty, to_remove, axis=1)

    return distances_without_uncertainty, to_remove


if __name__ == '__main__':
    n = 4

    # Generate a matrix of distances from random points
    points = np.random.rand(n, 2)
    distances = np.linalg.norm(points[:, None] - points, axis=2)
    certainty = np.random.rand(n, n)

    print(distances)
    print(certainty)

    threshold = 0.2

    # Plot points
    import matplotlib.pyplot as plt

    plt.scatter(points[:, 0], points[:, 1], s=50, alpha=0.5, label='Original points')

    # Add noise to distance equivalent to the uncertainty
    distances += np.random.normal(0, certainty, distances.shape) / 10
    distances = (distances + distances.T) / 2

    # Plot MDS
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity='precomputed', metric=True)
    mds_coors = mds.fit_transform(distances)

    # Rotate and translate the MDS points
    mds_coors = rotate_and_translate(points, mds_coors)

    plt.scatter(mds_coors[:, 0], mds_coors[:, 1], s=50, alpha=0.5, label='With uncertain agents')

    mds = MDS(n_components=2, dissimilarity='precomputed', metric=True)
    new_distances, removed = remove_uncertain_agents(distances, certainty, threshold=threshold)
    mds_coors = mds.fit_transform(new_distances)
    mds_coors = rotate_and_translate(points[[i for i in range(n) if i not in removed]], mds_coors)
    plt.scatter(mds_coors[:, 0], mds_coors[:, 1], s=50, alpha=0.5, label='Without uncertain agents')

    plt.axis('equal')
    plt.legend()
    plt.show()
