import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from alpha_shapes import Alpha_Shaper, plot_alpha_shape

"""
Example 1

distance matrix 
[[  0. 280.  69. 283. 257.]
 [  0.   0. 177. 168. 117.]
 [  0.   0.   0. 226. 204.]
 [  0.   0.   0.   0. 223.]
 [  0.   0.   0.   0.   0.]]
certainty matrix 
[[  0.         100.          96.059601    96.059601    88.63848717]
 [  0.           6.59073033  74.71720943  98.          77.73387535]
 [  0.           0.          42.34610142  89.52469025  92.21721696]
 [  0.           0.           0.          16.67454773  83.32017794]
 [  0.           0.           0.           0.           7.9747837 ]]

"""

def circle_intersection(x1, y1, r1, x2, y2, r2):
    """Compute intersection points of two circles in 2D"""
    # Calculate distance between the centers of the circles
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)

    # Check if circles are coincident or one circle is inside the other
    if dist > r1 + r2 or dist < abs(r1 - r2):
        # No intersection
        return []

    # Calculate intersection points
    a = (r1*r1 - r2*r2 + dist*dist) / (2 * dist)
    h = math.sqrt(r1*r1 - a*a)
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


if __name__ == '__main__':
    # Matrices
    distance_matrix = np.array([
        [0, 280, 69, 283, 257],
        [0, 0, 177, 168, 117],
        [0, 0, 0, 226, 204],
        [0, 0, 0, 0, 223],
        [0, 0, 0, 0, 0]
    ])

    distance_matrix = distance_matrix + distance_matrix.T

    certainty_matrix = np.array([
        [0, 100, 96.059601, 96.059601, 88.63848717],
        [0, 0, 74.71720943, 98, 77.73387535],
        [0, 0, 0, 89.52469025, 92.21721696],
        [0, 0, 0, 0, 83.32017794],
        [0, 0, 0, 0, 0]
    ])

    certainty = np.copy(certainty_matrix + certainty_matrix.T)

    # Put axis on the plot, with alpha=0.5
    plt.axhline(0, color='black', lw=1, alpha=0.5)
    plt.axvline(0, color='black', lw=1, alpha=0.5)

    # Plot grid
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    ax = plt.gca()
    ax.cla()  # clear things for fresh plot

    # Axis aspect ratio should be equal
    ax.set_aspect('equal', adjustable='box')

    # Plot the first agent (index 0) (on top of axis)
    ax.scatter(0, 0, alpha=0.5)
    ax.text(0, 0, 'A')

    set_points = set()
    set_points.add(0)  # Add the first point to the set

    # Possible positions of each drawn point
    placements = []
    for i in range(5):
        placements.append([])

    placements[0].append((0, 0))

    max_index = np.unravel_index(np.nanargmax(certainty_matrix[0, :]), certainty_matrix.shape[0])[0]
    print(distance_matrix[0, max_index], certainty_matrix[0, max_index])

    # Plot the point
    ax.scatter(distance_matrix[0, max_index], 0, c='orange', alpha=0.5)
    placements[max_index].append((distance_matrix[0, max_index], 0))

    print(placements)

    # Plot uncertainty circle as a donut at +-5 of the radius (measured distance)
    radius = distance_matrix[0, max_index]
    error = error_computation(certainty[0, max_index], 10)

    # Also use the external error points (points from which the error circle should also be plotted)
    # Comes from the fact that there is a surface in which the point can be, not only one point
    # The first one is the point on the x-axis, which makes two external error points,
    # then it will depend on the found polygon (Can probably keep four closest points)
    placements[max_index].append((distance_matrix[0, max_index] + error, 0))
    placements[max_index].append((distance_matrix[0, max_index] - error, 0))

    set_points.add(max_index)  # Add the second point to the set

    # Plot the uncertainty circles with radius +- error (fill the interior with green color)
    # ax.add_patch(plt.Circle((0, 0), radius, color='b', fill=False, alpha=0.5))
    ax.add_patch(plt.Circle((0, 0), radius + error, color='g', fill=False, alpha=0.5))
    ax.add_patch(plt.Circle((0, 0), radius - error, color='g', fill=False, alpha=0.5))

    # remove the point from the matrix
    certainty_matrix[0, max_index] = 0

    # Take the next maximum certainty point (using only the indexes in the set)
    print(certainty_matrix)
    print(certainty_matrix[list(set_points), :], certainty_matrix[list(set_points), :].shape)

    for _ in range(2):
        max_index = np.unravel_index(np.nanargmax(certainty_matrix[list(set_points), :]), certainty_matrix[list(set_points), :].shape)

        center_index = list(set_points)[max_index[0]]
        outer_index = max_index[1]
        print(max_index, center_index, outer_index)

        estimation_circles = []
        certainty_circles = []

        for point in set_points:
            print(distance_matrix[point, outer_index], certainty_matrix[point, outer_index])

            # Plot uncertainty circle around the center point toward the outer point
            radius = distance_matrix[point, outer_index]
            error = error_computation(certainty[point, outer_index], 10)

            ax.add_patch(plt.Circle(placements[point][0], radius, color='b', fill=False, alpha=0.5))

            # Where we estimate the point to be (distance measurement)
            estimation_circles.append((placements[point][0][0], placements[point][0][1], radius))

            certainty_matrix[point, outer_index] = 0

            # Where the point can be (uncertainty)
            for placement in placements[point]:
                ax.add_patch(plt.Circle(placement, radius + error, color='g', fill=False, alpha=0.5))
                ax.add_patch(plt.Circle(placement, radius - error, color='g', fill=False, alpha=0.5))

                certainty_circles.extend(
                    [
                        (placement[0], placement[1], radius + error),
                        (placement[0], placement[1], radius - error)
                    ]
                )

        # Find intersection with estimation circles
        point_estimation = []
        for i in range(len(estimation_circles)):
            for j in range(i+1, len(estimation_circles)):
                circle1 = estimation_circles[i]
                circle2 = estimation_circles[j]
                point_estimation.extend(circle_intersection(*circle1, *circle2))

        print(point_estimation)

        # Cluster those points in two subsets and compute centroid of each cluster
        kmeans = KMeans(n_clusters=len(estimation_circles), n_init=10, random_state=0).fit(point_estimation)
        centroids = kmeans.cluster_centers_

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
            for j in range(i+1, len(certainty_circles)):
                circle1 = certainty_circles[i]
                circle2 = certainty_circles[j]
                intersection_points.extend(circle_intersection(*circle1, *circle2))

        # Plot intersection points
        # for point in intersection_points[:]:
        #     ax.scatter(point[0], point[1], color='b', alpha=0.5)

        ref_x, ref_y = centroids[0]
        # Keep only the points inside the uncertainty circle
        intersection_points = [point for point in intersection_points if math.sqrt((point[0] - ref_x)**2 + (point[1] - ref_y)**2) < 20]

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
        ax.scatter(*new_point, c='orange', alpha=0.5)

        # Plot the convex hull
        hull = ConvexHull(neighbors)

        for simplex in hull.simplices:
            ax.plot(neighbors[simplex, 0], neighbors[simplex, 1], 'c')
        ax.plot(neighbors[hull.vertices, 0], neighbors[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)

        # Add the points of the convex hull as the new positions
        placements[outer_index].append(new_point)
        placements[outer_index].extend(neighbors[hull.vertices])

        set_points.add(outer_index)  # Add the third point to the set

        # Find the next maximum certainty point
        certainty_matrix[outer_index, :] = 0

    for agent in set_points:
        ax.scatter(*(placements[agent][0]), s=100, c='k')

    print(certainty_matrix)

    # Axis to show all plotted points
    plt.axis('equal')

    # Show the plot
    plt.show()

