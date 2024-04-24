import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from tqdm import tqdm

from experiments.utils.rotation import find_rotation_matrix, rotate_and_translate


def circle_intersection(x1, y1, r1, x2, y2, r2):
    """Compute intersection points of two circles in 2D"""
    # Calculate distance between the centers of the circles
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)

    # Check if circles are coincident or one circle is inside the other
    if dist > r1 + r2 or dist < abs(r1 - r2):
        # No intersection
        return []

    # Calculate intersection points
    a = (r1 * r1 - r2 * r2 + dist * dist) / (2 * dist)
    h = math.sqrt(r1 * r1 - a * a)
    x3 = x1 + a * (x2 - x1) / dist
    y3 = y1 + a * (y2 - y1) / dist
    x4 = x3 + h * (y2 - y1) / dist
    y4 = y3 - h * (x2 - x1) / dist

    # If circles are touching at one point
    if dist == r1 + r2 or dist == abs(r1 - r2):
        return [(x4, y4)]
    else:
        # Two intersection points
        x5 = x3 - h * (y2 - y1) / dist
        y5 = y3 + h * (x2 - x1) / dist
        return [(x4, y4), (x5, y5)]


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


def algo(points, matrix, certainty, uncertainty, visual=False, ax=None):
    n = points.shape[0]

    # Find the two points with the biggest certainty in the certainty matrix
    chosen_points = list(np.argmax(certainty))

    # Generate the first triangle with a heuristic (the first two points are the base)
    base_positions = np.zeros((2, 2))
    base_positions[0] = np.array([0, 0])
    base_positions[1] = np.array([matrix[chosen_points[0], chosen_points[1]], 0])

    # Rotate the base positions to the points
    rotation_matrix = find_rotation_matrix(points[chosen_points].T, base_positions.T)
    base_positions = base_positions @ rotation_matrix

    # Translate the base positions to the points
    centroid = np.mean(points[chosen_points], axis=0)
    base_centroid = np.mean(base_positions, axis=0)

    translation = centroid - base_centroid
    base_positions = base_positions + translation

    # Translate the first point of the base to the first point of the points
    # translation = points[chosen_points[0]] - base_positions[0]
    # base_positions = base_positions + translation

    # Randomly chose a point not in the chosen points
    not_chosen = [i for i in range(n) if i not in chosen_points]

    # Store estimated positions of the new points
    estimated_positions = np.zeros((n, 2))

    # estimated_positions[chosen_points] = mds_coors_2
    estimated_positions[chosen_points] = base_positions

    # Loop on shuffled remaining points
    position = np.array([0, 0])
    precision = 10
    errors = {}
    for i in not_chosen:
        min_error = np.inf
        for x in np.linspace(points[i, 0] - uncertainty[i], points[i, 0] + uncertainty[i], precision):
            for y in np.linspace(points[i, 1] - uncertainty[i], points[i, 1] + uncertainty[i], precision):
                new_point = np.array([x, y])

                # Trusted points are the chosen points with bounded uncertainty
                # TODO: Find a better way to chose trusted points (associate to measurements uncertainty)
                trusted_points = [j for j in chosen_points if certainty[i, j] >= 0]

                new_distances = np.linalg.norm(new_point - estimated_positions[trusted_points], axis=1)

                err = np.sum(np.abs(new_distances - matrix[trusted_points, i]))

                errors[(x, y)] = err

                if err < min_error:
                    position = new_point
                    min_error = err

        estimated_positions[i] = position
        chosen_points.append(i)

    return estimated_positions


def algorithm(points, matrix, uncertainty, visual=False, ax=None):
    n = points.shape[0]

    # Chose three points and plot the distance between them
    chosen_points = [0, 1]

    if visual and ax is not None:
        for i in chosen_points:
            for j in chosen_points:
                if j > i:
                    plt.plot([_points[i, 0], _points[j, 0]], [_points[i, 1], _points[j, 1]], 'g-', alpha=0.5)

    # Generate the first triangle with a heuristic (the first two points are the base)
    base_positions = np.zeros((2, 2))
    base_positions[0] = np.array([0, 0])
    base_positions[1] = np.array([matrix[chosen_points[0], chosen_points[1]], 0])

    # Rotate the base positions to the points
    rotation_matrix = find_rotation_matrix(points[:2].T, base_positions.T)
    base_positions = base_positions @ rotation_matrix

    # Translate the base positions to the points
    centroid = np.mean(points[chosen_points], axis=0)
    base_centroid = np.mean(base_positions, axis=0)

    translation = centroid - base_centroid
    base_positions = base_positions + translation

    # Translate the first point of the base to the first point of the points
    translation = points[chosen_points[0]] - base_positions[0]
    base_positions = base_positions + translation

    # Randomly chose a point not in the chosen points
    not_chosen = [i for i in range(n) if i not in chosen_points]
    random_point = not_chosen[0]

    # Store estimated positions of the new points
    estimated_positions = np.zeros((n, 2))

    # estimated_positions[chosen_points] = mds_coors_2
    estimated_positions[chosen_points] = base_positions

    if visual and ax is not None:
        # Plot the new distances from chosen points to the random point
        for i in chosen_points:
            ax.add_artist(
                plt.Circle((estimated_positions[i, 0], estimated_positions[i, 1]), matrix[i, random_point], fill=False))

        # Plot a bounding box around the random point (using uncertainty)
        ax.add_artist(
            plt.Rectangle(
                (points[random_point, 0] - uncertainty[random_point],
                 points[random_point, 1] - uncertainty[random_point]),
                2 * uncertainty[random_point], 2 * uncertainty[random_point], fill=False
            )
        )

    # Loop on shuffled remaining points
    position = np.array([0, 0])
    precision = 10
    errors = {}
    for i in not_chosen:
        min_error = np.inf
        for x in np.linspace(points[i, 0] - uncertainty[i], points[i, 0] + uncertainty[i], precision):
            for y in np.linspace(points[i, 1] - uncertainty[i], points[i, 1] + uncertainty[i], precision):
                new_point = np.array([x, y])

                # Trusted points are the chosen points with bounded uncertainty
                # TODO: Find a better way to chose trusted points (associate to measurements uncertainty)
                trusted_points = [j for j in chosen_points if uncertainty[j] < 100]
                # trusted_points = [j for j in chosen_points if certainty[i, j] > 50]

                new_distances = np.linalg.norm(new_point - estimated_positions[trusted_points], axis=1)

                err = np.sum(np.abs(new_distances - matrix[trusted_points, i]))

                errors[(x, y)] = err

                if err < min_error:
                    position = new_point
                    min_error = err

        # Showcase the error gradient for the first point
        if random_point == i and visual:
            # Normalize the errors between 0 and 1
            max_err = max(errors.values())

            for (x, y), err in errors.items():
                err = err / max_err
                plt.scatter(x, y, s=20, alpha=0.5, color=(err, 0, 1 - err))

        estimated_positions[i] = position

        chosen_points.append(i)

    return estimated_positions


if __name__ == '__main__':
    # Parameters
    movement = 10
    noise = 5

    n_agents = 4

    visualization = True
    iterations = 1000

    fig, _ax = plt.subplots(1, 2)

    # Run the algorithm
    mds_y = []
    estimation_y = []
    for it in tqdm(range(iterations)):
        _visualization = visualization and it == 0

        # Generate a set of n_agents points
        _points, _matrix = generate_points_and_matrix(n_agents, 300, 30)

        # Generate a new set of points by moving those points of a given distance in random directions
        new_points = _points + np.random.rand(n_agents, 2) * movement * np.random.choice([-1, 1], (n_agents, 2))

        # Generate the new distance matrix
        new_matrix = np.zeros((n_agents, n_agents))

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Compute the distance between the points
                new_distance = np.linalg.norm(new_points[i] - new_points[j])

                # Add the distance to the matrix
                new_matrix[i, j] = new_distance
                new_matrix[j, i] = new_distance

        # Add noise to the new matrix
        new_matrix += np.random.rand(n_agents, n_agents) * noise

        # Increase noise (first point has no noise, last point has the most noise)
        error = np.zeros(n_agents)
        for j in range(n_agents):
            error[j] = movement * ((j + 1) / n_agents)

        # new_matrix += error

        # Apply MDS
        mds_model = MDS(n_components=2, dissimilarity='precomputed', metric=True, normalized_stress=False)
        mds = mds_model.fit_transform((new_matrix + new_matrix.T) / 2)

        # Transform the MDS points to the same scale as the original points
        mds = rotate_and_translate(_points, mds)

        # Decide on uncertainty allowed
        uncertainty = error + movement + noise

        # Plot uncertainty as circles around the points
        if _visualization:
            for i in range(n_agents):
                _ax[1].add_artist(
                    plt.Circle((_points[i, 0], _points[i, 1]), uncertainty[i], fill=False, alpha=0.5, color='r'))

        estimation = algorithm(_points, new_matrix, uncertainty, visual=_visualization, ax=_ax[1])

        if _visualization:
            plt.scatter(_points[:, 0], _points[:, 1], color='tab:blue', label='Original points')
            plt.scatter(new_points[:, 0], new_points[:, 1], color='tab:orange', label='New points')

            plt.scatter(mds[:, 0], mds[:, 1], color='tab:green', label='MDS points')

            # plt.scatter(mds_coors_2[:, 0], mds_coors_2[:, 1], color='tab:red', label='Partial MDS points')
            plt.scatter(estimation[:, 0], estimation[:, 1], color='tab:purple', label='Estimated positions')

            letters = [chr(65 + i) for i in range(n_agents)]
            for i in range(n_agents):
                plt.text(estimation[i, 0] + 1, estimation[i, 1] + 1, letters[i], fontsize=8)

        # Compute error between new points and estimated positions
        estimation_err = np.linalg.norm(new_points - rotate_and_translate(new_points, estimation), axis=1)
        # Mean square error between new points and estimated positions
        mse = np.mean(estimation_err ** 2)

        estimation_y.append(mse)

        # Compute error between new points and MDS points
        mds_err = np.linalg.norm(new_points - rotate_and_translate(new_points, mds), axis=1)
        # Mean square error between new points and MDS points
        mse = np.mean(mds_err ** 2)

        mds_y.append(mse)

    _ax[0].plot(estimation_y, color='tab:purple', label='Heuristic')
    _ax[0].plot(mds_y, color='tab:green', label='MDS')
    _ax[0].legend()

    print(f'Heuristic: {np.mean(estimation_y)}')
    print(f'MDS: {np.mean(mds_y)}')

    plt.axis('equal')
    plt.legend()
    plt.show()
