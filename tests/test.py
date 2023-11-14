import numpy as np
from skimage import transform
import matplotlib.pyplot as plt

# Robot knowledge
robot_source_knowledge = {
    "A": [2, 5],
    "B": [5, 2],
    "C": [5, 5],
    # "D": [2, 2],
    "E": [5, 6]
}

robot_target_knowledge = {
    "A": [3, 3],
    "B": [0, 0],
    "C": [0, 3],
    # "D": [3, 0],
    "F": [6, 6]
}

# Extract common points
common_points = set(robot_source_knowledge.keys()) & set(robot_target_knowledge.keys())

all_source_points = np.array([robot_source_knowledge[point] for point in robot_source_knowledge])
all_target_points = np.array([robot_target_knowledge[point] for point in robot_target_knowledge])

# Convert common points to numpy arrays
source_points = np.array([robot_source_knowledge[point] for point in common_points])
target_points = np.array([robot_target_knowledge[point] for point in common_points])

# Estimate the transformation matrix
matrix = transform.estimate_transform('affine', source_points, target_points)

# Apply the transformation to robot_1_knowledge
transformed_robot_1_knowledge = {point: transform.matrix_transform(np.array(coord), matrix) for point, coord in robot_source_knowledge.items()}

all_points_transformed = transform.matrix_transform(all_source_points, matrix)
# print(len(all_points_transformed))

print(transformed_robot_1_knowledge)
print(all_points_transformed)


# plt.axis('equal')

# Plot original, common and transformed points
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

# Plot original positions
ax1.scatter(all_source_points[:, 0], all_source_points[:, 1], label='Robot 1', c='blue')
ax1.scatter(all_target_points[:, 0], all_target_points[:, 1], label='Robot 2', c='red')
ax1.set_title('Original Positions')
ax1.axis('equal')
ax1.legend()

# Plot common positions
ax2.scatter(source_points[:, 0], source_points[:, 1], label='Robot 1', c='blue')
ax2.scatter(target_points[:, 0], target_points[:, 1], label='Robot 2', c='red')
ax2.set_title('Common Points')
ax2.axis('equal')
ax2.legend()

# Plot transformed positions
ax3.scatter(all_target_points[:, 0], all_target_points[:, 1], label='Robot 2', c='red')
ax3.scatter(
    # [point[:, 0] for point in transformed_robot_1_knowledge.values()],
    # [point[:, 1] for point in transformed_robot_1_knowledge.values()],
    all_points_transformed[:, 0],
    all_points_transformed[:, 1],
    label='Transformed Robot 1', c='green'
)
ax3.set_title('Transformed Positions')
ax3.axis('equal')
ax3.legend()

# Show the subplots
plt.show()
