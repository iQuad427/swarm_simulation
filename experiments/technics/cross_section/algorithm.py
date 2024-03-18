import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans


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


def error_computation(certainty, speed):
    # Error is small when certainty is at 100, but it increases fast when certainty decreases
    distance = (100 - certainty) * speed / 10
    return 5 + distance


def algorithm(distance, certainty):
    backup_distance = np.copy(distance + distance.T)
    backup_certainty = np.copy(certainty + certainty.T)

    distance = distance + distance.T
    points_set = set()
    placements = [[] for _ in range(distance.shape[0])]

    placements[0].append(np.array([0, 0]))
    points_set.add(0)

    # Find the second point
    max_index = np.unravel_index(
        np.nanargmax(
            certainty[list(points_set), :]
        ),
        certainty[list(points_set), :].shape
    )

    center_index = list(points_set)[max_index[0]]
    outer_index = max_index[1]

    # Plot uncertainty circle as a donut at +-5 of the radius (measured distance)
    radius = distance[center_index, outer_index]
    error = error_computation(backup_certainty[center_index, outer_index], 10)

    # Plot the point
    ax.scatter(distance[center_index, outer_index], 0, c='orange', alpha=0.5)

    # Estimated position
    placements[outer_index].append(np.array([distance[center_index, outer_index], 0]))

    # Error circle
    placements[outer_index].append((radius + error, 0))
    placements[outer_index].append((radius - error, 0))

    points_set.add(outer_index)  # Add the second point to the set

    # Remove the measurement from the matrix, shouldn't be selected anymore
    certainty[center_index, outer_index] = 0

    # while not np.all(certainty == 0):
    for _ in range(1):
        print(certainty)
        print(points_set)

        # Take the next maximum certainty point (using only the indexes of set points)
        max_index = np.unravel_index(
            np.nanargmax(
                certainty[list(points_set), :]
            ),
            certainty[list(points_set), :].shape
        )

        center_index = list(points_set)[max_index[0]]
        outer_index = max_index[1]

        print(max_index, distance[center_index, outer_index], certainty[center_index, outer_index])

        estimation_circles = []
        certainty_circles = []

        for point in points_set:
            print(point, outer_index, distance[point, outer_index], certainty[point, outer_index])

            # Plot uncertainty circle around the center point toward the outer point
            radius = distance[point, outer_index]
            error = error_computation(backup_certainty[point, outer_index], 10)

            print("error", certainty[point, outer_index], error)

            ax.add_patch(plt.Circle(placements[point][0], radius, color='b', fill=False, alpha=0.5))

            # Where we estimate the point to be (distance measurement)
            estimation_circles.append((placements[point][0][0], placements[point][0][1], radius))

            certainty[point, outer_index] = 0

            # Where the point can be (uncertainty)
            for placement in placements[point]:
                ax.add_patch(plt.Circle(placement, radius + error, color='g', fill=False, alpha=0.5))
                ax.add_patch(plt.Circle(placement, radius - error, color='g', fill=False, alpha=0.5))

                # Add the circles to the list for each 10 units of error
                certainty_circles.extend(
                    [
                        (placement[0], placement[1], radius + (error - 10 * i)) for i in range(int(error / 10) + 1)
                    ]
                )

                certainty_circles.extend(
                    [
                        (placement[0], placement[1], radius - (error + 10 * i)) for i in range(int(error / 10) + 1)
                    ]
                )

            plt.pause(1)

        print(estimation_circles)

        # Find intersection with estimation circles
        point_estimation = []
        for i in range(len(estimation_circles)):
            for j in range(i + 1, len(estimation_circles)):
                circle1 = estimation_circles[i]
                circle2 = estimation_circles[j]
                point_estimation.extend(circle_intersection(*circle1, *circle2))

        print("points", point_estimation)

        # Cluster those points in two subsets and compute centroid of each cluster
        n_clusters = len(points_set) - len(estimation_circles) if len(point_estimation) - len(estimation_circles) > 2 else 2
        print("n_clusters", n_clusters)
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10, random_state=0
            ).fit(point_estimation)
            centroids = kmeans.cluster_centers_
        except ValueError as e:
            print("No intersection found", e)
            break

        # Recover each cluster
        clusters = []
        for i in range(len(estimation_circles)):
            clusters.append(np.array(point_estimation)[kmeans.labels_ == i])

        # Compute weighed average of each cluster using np.average
        for i, cluster in enumerate(clusters):
            x, y = np.average(
                cluster,
                # TODO: add weights depending on the certainty of the measurements
                axis=0
            )
            ax.scatter(x, y, color='r', alpha=0.5)

        # Find all intersection points
        intersection_points = []
        for i in range(len(certainty_circles)):
            for j in range(i + 1, len(certainty_circles)):
                circle1 = certainty_circles[i]
                circle2 = certainty_circles[j]
                intersection_points.extend(circle_intersection(*circle1, *circle2))

        # Chose the ref points as the centroid that decreases the most the error to all measurements
        candidates = []
        for i, centroid in enumerate(centroids):
            ref_x, ref_y = centroid
            error = 0
            for point in points_set:
                coords = placements[point][0]
                factor = 1  #/ error_computation(backup_certainty[point, outer_index], 10)
                error += factor * abs(
                    math.sqrt((coords[0] - ref_x) ** 2 + (coords[1] - ref_y) ** 2) - distance[point, outer_index]
                )

            candidates.append((ref_x, ref_y, error))

        ref_x, ref_y, _ = min(candidates, key=lambda x: x[2])

        # Keep only the points inside the uncertainty circle
        intersection_points = [
            point for point in intersection_points if math.sqrt((point[0] - ref_x) ** 2 + (point[1] - ref_y) ** 2) < 20
        ]

        # Create a Triangulation object
        triangulation = Triangulation(
            [ref_x] + [point[0] for point in intersection_points],
            [ref_y] + [point[1] for point in intersection_points],
        )

        # Plot the triangulation
        ax.triplot(triangulation, c='red', alpha=0.5, label='Triangulation')

        # Plot neighbours of the first point
        neighbors = []
        for i, j in triangulation.edges:
            if i == 0:
                ax.scatter(*(intersection_points[j - 1]), c='green', alpha=0.5)
                neighbors.append(intersection_points[j - 1])
            elif j == 0:
                ax.scatter(*(intersection_points[i - 1]), c='green', alpha=0.5)
                neighbors.append(intersection_points[i - 1])

        neighbors = np.array(neighbors)

        # Define the new point as the centroid of the neighbors
        new_point = np.mean(neighbors, axis=0)

        # Plot the convex hull
        hull = ConvexHull(neighbors)

        for simplex in hull.simplices:
            ax.plot(neighbors[simplex, 0], neighbors[simplex, 1], 'c')
        ax.plot(
            neighbors[hull.vertices, 0], neighbors[hull.vertices, 1], 'o',
            mec='r', color='none', lw=1, markersize=10
        )

        # Add the points of the convex hull as the new positions
        placements[outer_index].append(new_point)
        placements[outer_index].extend(neighbors[hull.vertices])

        ax.scatter(*new_point, c='orange', alpha=0.5)

        points_set.add(outer_index)  # Add the third point to the set

        certainty[:, outer_index] = 0  # Remove the measurements from the matrix, shouldn't be selected anymore

        plt.pause(3)

    print(placements)

    return [agent[0] for agent in placements if len(agent) > 0]


if __name__ == '__main__':
    # Matrices
    distance_matrix = np.array([
        [0, 280, 69, 283, 257],
        [0, 0, 177, 168, 117],
        [0, 0, 0, 226, 204],
        [0, 0, 0, 0, 223],
        [0, 0, 0, 0, 0]
    ])

    certainty_matrix = np.array([
        [0, 100, 96.059601, 96.059601, 88.63848717],
        [0, 0, 74.71720943, 98, 77.73387535],
        [0, 0, 0, 89.52469025, 92.21721696],
        [0, 0, 0, 0, 83.32017794],
        [0, 0, 0, 0, 0]
    ])

    # Put axis on the plot, with alpha=0.5
    plt.axhline(0, color='black', lw=1, alpha=0.5)
    plt.axvline(0, color='black', lw=1, alpha=0.5)

    # Plot grid
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    ax = plt.gca()
    ax.cla()  # clear things for fresh plot

    # Axis aspect ratio should be equal
    ax.set_aspect('equal', adjustable='box')

    # Plot error function
    # x = np.linspace(0, 100, 100)
    # y = error_computation(x, 10)
    # plt.plot(x, y, label='Error function')

    results = algorithm(
        distance_matrix,
        certainty_matrix
    )

    print(results)

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    offset = 5
    for _, result in enumerate(results):
        ax.scatter(result[0], result[1], color='b', alpha=0.5)
        ax.text(result[0] + offset, result[1] + offset, letters[_])

    # Plot MDS
    # mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, metric=True)
    # mds_coors = mds.fit_transform(distance_matrix + distance_matrix.T)
    #
    # plt.scatter(mds_coors[:, 0], mds_coors[:, 1], s=50)
    # plt.scatter(mds_coors[0, 0], mds_coors[0, 1], s=50, c='r')
    # plt.axis('equal')

    plt.show()
