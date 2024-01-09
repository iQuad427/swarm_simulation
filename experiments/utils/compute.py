import matplotlib.tri as tri


def compute_area(point_coords):
    # Use matplotlib.tri to plot the triangulation
    triangulation = tri.Triangulation(point_coords[:, 0], point_coords[:, 1])

    # Compute area and perimeter
    area = 0

    for i in range(len(triangulation.triangles)):
        triangle = triangulation.triangles[i]
        p1 = point_coords[triangle[0]]
        p2 = point_coords[triangle[1]]
        p3 = point_coords[triangle[2]]

        # Compute area
        #  A = (1/2) |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|
        area += abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2)

    return area


def mean_squared_error(point_coords, mds_coors):
    # Compute the mean squared error
    error = 0

    for i in range(len(point_coords)):
        error += (point_coords[i][0] - mds_coors[i][0]) ** 2 + (point_coords[i][1] - mds_coors[i][1]) ** 2

    return error / len(point_coords)