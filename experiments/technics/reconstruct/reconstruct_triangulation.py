import numpy as np
from matplotlib import pyplot as plt
from skimage import transform

# Example:
#
#     ###           ###           ###
#     #A#           #B#           #C#
#     ###           ###           ###
#
#
#     ###           ###           ###
#     #D#           #E#           #F#
#     ###           ###           ###
#
#
#     ###           ###           ###
#     #G#           #H#           #I#
#     ###           ###           ###
#
triangulation_data_9 = {
    "A": {
        "A": [0, 0],
        "B": [0, 10],
        "D": [10, 0],
    },
    "B": {
        "B": [0, 0],
        "A": [0, -10],
        "E": [10, 0],
        "C": [0, 10],
    },
    "C": {
        "C": [0, 0],
        "B": [0, -10],
        "F": [10, 0],
    },
    "D": {
        "D": [0, 0],
        "A": [-10, 0],
        "E": [0, 10],
        "G": [10, 0],
    },
    "E": {
        "E": [0, 0],
        "B": [-10, 0],
        "F": [0, 10],
        "D": [0, -10],
        "H": [10, 0],
    },
    "F": {
        "F": [0, 0],
        "C": [-10, 0],
        "E": [0, -10],
        "I": [10, 0],
    },
    "G": {
        "G": [0, 0],
        "D": [-10, 0],
        "H": [0, 10],
    },
    "H": {
        "H": [0, 0],
        "E": [-10, 0],
        "G": [0, -10],
        "I": [0, 10],
    },
    "I": {
        "I": [0, 0],
        "F": [-10, 0],
        "H": [0, -10],
    },
}

# Example:
#
#     ###           ###           ###           ###
#     #A#           #B#           #C#           #D#
#     ###           ###           ###           ###
#
#
#     ###           ###           ###           ###
#     #E#           #F#           #G#           #H#
#     ###           ###           ###           ###
#
#
#     ###           ###           ###           ###
#     #I#           #J#           #K#           #L#
#     ###           ###           ###           ###
#
#
#     ###           ###           ###           ###
#     #M#           #N#           #O#           #P#
#     ###           ###           ###           ###
#
triangulation_data_16 = {
    "A": {
        "A": [0, 0],
        "B": [0, 10],
        "E": [10, 0],
    },
    "B": {
        "B": [0, 0],
        "A": [0, -10],
        "C": [0, 10],
        "F": [10, 0],
    },
    "C": {
        "C": [0, 0],
        "B": [0, -10],
        "D": [0, 10],
        "G": [10, 0],
    },
    "D": {
        "D": [0, 0],
        "C": [0, -10],
        "H": [10, 0],
    },
    "E": {
        "E": [0, 0],
        "A": [-10, 0],
        "F": [0, 10],
        "I": [10, 0],
    },
    "F": {
        "F": [0, 0],
        "B": [-10, 0],
        "E": [0, -10],
        "G": [0, 10],
        "J": [10, 0],
    },
    "G": {
        "G": [0, 0],
        "C": [-10, 0],
        "F": [0, -10],
        "H": [0, 10],
        "K": [10, 0],
    },
    "H": {
        "H": [0, 0],
        "D": [-10, 0],
        "G": [0, -10],
        "L": [10, 0],
    },
    "I": {
        "I": [0, 0],
        "E": [-10, 0],
        "J": [0, 10],
        "M": [10, 0],
    },
    "J": {
        "J": [0, 0],
        "F": [-10, 0],
        "I": [0, -10],
        "K": [0, 10],
        "N": [10, 0],
    },
    "K": {
        "K": [0, 0],
        "G": [-10, 0],
        "J": [0, -10],
        "L": [0, 10],
        "O": [10, 0],
    },
    "L": {
        "L": [0, 0],
        "H": [-10, 0],
        "K": [0, -10],
        "P": [10, 0],
    },
    "M": {
        "M": [0, 0],
        "I": [-10, 0],
        "N": [0, 10],
    },
    "N": {
        "N": [0, 0],
        "J": [-10, 0],
        "M": [0, -10],
        "O": [0, 10],
    },
    "O": {
        "O": [0, 0],
        "K": [-10, 0],
        "N": [0, -10],
        "P": [0, 10],
    },
    "P": {
        "P": [0, 0],
        "L": [-10, 0],
        "O": [0, -10],
    },
}

# Set of randomly placed 16 points
np.random.seed(0)
points = np.random.randint(0, 100, (16, 2))

# scatter plot of the 16 points in blue, but in larger shape
plt.axis('equal')
plt.scatter(points[:, 0], points[:, 1], c=(0, 0, 1), marker='x', s=100)

# The triangulation data associated to the 16 points
triangulation_data_random = {
    "A": {
        "A": points[0],
        "B": points[1],
        "E": points[2],
    },
    "B": {
        "B": points[1],
        "A": points[0],
        "C": points[3],
        "F": points[4],
    },
    "C": {
        "C": points[3],
        "B": points[1],
        "D": points[5],
        "G": points[6],
    },
    "D": {
        "D": points[5],
        "C": points[3],
        "H": points[7],
    },
    "E": {
        "E": points[2],
        "A": points[0],
        "F": points[4],
        "I": points[8],
    },
    "F": {
        "F": points[4],
        "B": points[1],
        "E": points[2],
        "G": points[6],
        "J": points[9],
    },
    "G": {
        "G": points[6],
        "C": points[3],
        "F": points[4],
        "H": points[7],
        "K": points[10],
    },
    "H": {
        "H": points[7],
        "D": points[5],
        "G": points[6],
        "L": points[11],
    },
    "I": {
        "I": points[8],
        "E": points[2],
        "J": points[9],
        "M": points[12],
    },
    "J": {
        "J": points[9],
        "F": points[4],
        "I": points[8],
        "K": points[10],
        "N": points[13],
    },
    "K": {
        "K": points[10],
        "G": points[6],
        "J": points[9],
        "L": points[11],
        "O": points[14],
    },
    "L": {
        "L": points[11],
        "H": points[7],
        "K": points[10],
        "P": points[15],
    },
    "M": {
        "M": points[12],
        "I": points[8],
        "N": points[13],
    },
    "N": {
        "N": points[13],
        "J": points[9],
        "M": points[12],
        "O": points[14],
    },
    "O": {
        "O": points[14],
        "K": points[10],
        "N": points[13],
        "P": points[15],
    },
    "P": {
        "P": points[15],
        "L": points[11],
        "O": points[14],
    },
}


triangulation_data = triangulation_data_random

robot_target_knowledge = triangulation_data["A"]

print("TARGET KNOWLEDGE:", robot_target_knowledge)

transformed_robot_knowledge = robot_target_knowledge
for i, source in enumerate(triangulation_data):
    if source == "A":
        continue

    # Robot knowledge
    robot_source_knowledge = triangulation_data[source]

    print("SOURCE KNOWLEDGE:", robot_source_knowledge)

    # Extract common points
    common_points = set(robot_source_knowledge.keys()) & set(robot_target_knowledge.keys())

    print("COMMON POINTS:", common_points)

    if len(common_points) < 2:
        continue

    # Convert common points to numpy arrays
    source_points = np.array([robot_source_knowledge[point] for point in common_points])
    target_points = np.array([robot_target_knowledge[point] for point in common_points])

    # Estimate the transformation matrix
    matrix = transform.estimate_transform('euclidean', source_points, target_points)

    # print("MATRIX:", matrix)

    print("SOURCE POINTS:", source_points)
    print("TARGET POINTS:", target_points)

    # Apply the transformation to robot_1_knowledge
    transformed_robot_knowledge = {
        point: transform.matrix_transform(
            np.array(coord), matrix
        )[0] for point, coord in robot_source_knowledge.items()
    }

    print("TRANSFORMED KNOWLEDGE:", transformed_robot_knowledge)

    for point in robot_target_knowledge:
        transformed_robot_knowledge[point] = robot_target_knowledge[point]
        # if point in transformed_robot_knowledge:
        #     a = list(transformed_robot_knowledge[point])
        #     b = robot_target_knowledge[point]
        #
        #     print("A:", a)
        #     print("B:", b)
        #
        #     transformed_robot_knowledge[point] = [
        #         (a[0] + b[0]) / 2,
        #         (a[1] + b[1]) / 2
        #     ]
        # else:
        #     print("POINT NOT IN TRANSFORMED KNOWLEDGE:", point)
        #     transformed_robot_knowledge[point] = robot_target_knowledge[point]

    robot_target_knowledge = transformed_robot_knowledge
    print("FINAL KNOWLEDGE:", transformed_robot_knowledge)

    x = []
    y = []
    for point in robot_target_knowledge.values():
        x.append(point[0])
        y.append(point[1])

    # Scatter iteratively, with color starting from red and ending to blue
    plt.axis('equal')
    plt.scatter(x, y, c=(1, 0, 0))

plt.show()
