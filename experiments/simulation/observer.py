import numpy as np


def reconstruct_distance_matrix(measures):
    # Reconstruct the distance matrix from the measures
    distance_matrix = [[(1, 0)] * len(measures) for _ in range(len(measures))]
    for i in range(len(measures)):
        for j in range(len(measures)):
            if i == j:
                continue

            # Only populate top-right triangle of the matrix
            if i > j:  # Swap i and j
                x = j
                y = i
            else:  # i <= j
                x = i
                y = j

            if measures[i][j][0] < distance_matrix[x][y][0]:
                continue

            distance_matrix[x][y] = measures[i][j]

    # Construct a numpy matrix from the distance matrix using the data[1] values
    dist_matrix = np.zeros((len(distance_matrix), len(distance_matrix)))
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            dist_matrix[i, j] = distance_matrix[i][j][1]

    # make the matrix symmetric by replication of the upper triangle
    dist_matrix = dist_matrix + dist_matrix.T - np.diag(dist_matrix.diagonal())

    return distance_matrix, dist_matrix
