
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from experiments.utils.rotation import find_rotation_matrix


def generate_points_and_matrix(n, max_distance, min_distance):
    """
    Generate a set of points and the distance matrix between the points

    :param n: number of points
    :param max_distance: maximum distance between points
    :param min_distance: minimum distance between points
    :return: the set of points and the distance matrix
    """
    # Generate the points
    points = np.random.rand(n, 2) * 100

    # Generate the distance matrix
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Compute the distance between the points
            distance = np.linalg.norm(points[i] - points[j])

            # Add some noise to the distance
            distance += np.random.rand() * 10

            # Add the distance to the matrix
            matrix[i, j] = distance
            matrix[j, i] = distance

    return points, matrix


if __name__ == '__main__':
    # Generate a set of 16 points
    points, matrix = generate_points_and_matrix(16, 300, 30)

    fig, ax = plt.subplots()

    # Chose three points and plot the distance between them
    chosen_points = [0, 1, 2]
    for i in chosen_points:
        for j in chosen_points:
            if j > i:
                plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'g-', alpha=0.5)

    # Generate a new set of points by moving those points of a given distance in random directions
    movement = 2
    new_points = points + np.random.rand(16, 2) * movement * np.random.choice([-1, 1], (16, 2))

    # Generate the new distance matrix
    new_matrix = np.zeros((16, 16))

    for i in range(16):
        for j in range(i + 1, 16):
            # Compute the distance between the points
            new_distance = np.linalg.norm(new_points[i] - new_points[j])

            # Add the distance to the matrix
            new_matrix[i, j] = new_distance
            new_matrix[j, i] = new_distance

    # Add noise to the new matrix
    noise = 5
    new_matrix += np.random.rand(16, 16) * noise * np.random.choice([-1, 1], (16, 16))
    # reduce noice (firsdt pint has no noise, last point has the most noise)
    for i in range(16):
        for j in range(i + 1, 16):
            new_matrix[i, j] = new_matrix[i, j] * (1 - i / 16) + i / 16 * 50
            new_matrix[j, i] = new_matrix[j, i] * (1 - i / 16) + i / 16 * 50

    # Decide on uncertainty allowed
    uncertainty = movement + noise

    # Plot uncertainty as circles around the points
    for i in range(16):
        ax.add_artist(plt.Circle((points[i, 0], points[i, 1]), uncertainty, fill=False, alpha=0.5, color='r'))

    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', metric=True, normalized_stress=False)
    mds_coors = mds.fit_transform((new_matrix + new_matrix.T)/2)

    mds_2 = MDS(n_components=2, dissimilarity='precomputed', metric=True, normalized_stress=False)
    sub_matrix = new_matrix[chosen_points][:, chosen_points]
    mds_coors_2 = mds_2.fit_transform((sub_matrix + sub_matrix.T)/2)

    # Transform the MDS points to the same scale as the original points

    # Rotate the MDS points
    rotation_matrix = find_rotation_matrix(points.T, mds_coors.T)
    mds_coors = mds_coors @ rotation_matrix

    rotation_matrix_2 = find_rotation_matrix(new_points[chosen_points].T, mds_coors_2.T)
    mds_coors_2 = mds_coors_2 @ rotation_matrix_2

    # Translate the MDS points
    centroid = np.mean(points, axis=0)
    mds_centroid = np.mean(mds_coors, axis=0)
    translation = centroid - mds_centroid
    mds_coors = mds_coors + translation

    centroid = np.mean(points[chosen_points], axis=0)
    mds_centroid = np.mean(mds_coors_2, axis=0)
    translation = centroid - mds_centroid
    mds_coors_2 = mds_coors_2 + translation

    # Randomly chose a point not in the chosen points
    not_chosen = [i for i in range(16) if i not in chosen_points]
    random_point = not_chosen[0]

    # Plot the new distances from chosen points to the random point
    for i in chosen_points:
        ax.add_artist(plt.Circle((mds_coors[i, 0], mds_coors[i, 1]), new_matrix[i, random_point], fill=False))

    # Plot a bounding box around the random point (using uncertainty)
    ax.add_artist(plt.Rectangle((points[random_point, 0] - uncertainty, points[random_point, 1] - uncertainty), 2 * uncertainty, 2 * uncertainty, fill=False))

    # Store estimated positions of the new points
    estimated_positions = np.zeros((16, 2))

    estimated_positions[chosen_points] = mds_coors_2

    # Loop on shuffled remaining points
    position = np.array([0, 0])
    for i in not_chosen:
        min_error = np.inf
        for x in np.linspace(points[i, 0] - uncertainty, points[i, 0] + uncertainty, 40):
            for y in np.linspace(points[i, 1] - uncertainty, points[i, 1] + uncertainty, 40):

                new_point = np.array([x, y])
                norm_summed = 0
                for j in chosen_points:
                    norm_summed += np.linalg.norm(new_point - points[j])
                new_distances = np.linalg.norm(new_point - estimated_positions[chosen_points], axis=1)
                error = np.sum(np.abs(new_distances - new_matrix[chosen_points, i])) / 50
                # plot color depending on error (progressive from red to green)
                if random_point == i and False:
                    plt.scatter(x, y, s=20, alpha=0.5, color=(error, 0, 1 - error))

                if error < min_error:
                    position = new_point
                    min_error = error
        plt.scatter(position[0], position[1], color='tab:red', alpha=1)
        estimated_positions[i] = position
        chosen_points.append(i)


    # Rotate estimated positions to match the points
    rotation_matrix = find_rotation_matrix(points.T, estimated_positions.T)
    estimated_positions = estimated_positions @ rotation_matrix

    # Translate estimated positions to match the points
    centroid = np.mean(points, axis=0)
    mds_centroid = np.mean(estimated_positions, axis=0)
    translation = centroid - mds_centroid
    estimated_positions = estimated_positions + translation

    # Plot the points
    plt.scatter(points[:, 0], points[:, 1], color='tab:blue', label='Original points')
    plt.scatter(mds_coors[:, 0], mds_coors[:, 1], color='tab:green', label='MDS points')
    plt.scatter(mds_coors_2[:, 0], mds_coors_2[:, 1], color='tab:red', label='Partial MDS points')
    plt.scatter(new_points[:, 0], new_points[:, 1], color='tab:orange', label='New points')

    plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], color='tab:purple', label='Estimated positions')

    # Equal aspect ratio
    plt.axis('equal')
    plt.legend()
    plt.show()
