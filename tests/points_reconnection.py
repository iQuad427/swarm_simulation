import numpy as np
from skimage import transform
import matplotlib.pyplot as plt

# Generate two sets of points
source_points = np.array([[1, 2], [3, 4], [5, 6]])
target_points = np.array([[10, 20], [30, 40], [50, 60]])

# Estimate the transformation matrix
matrix = transform.estimate_transform('affine', source_points, target_points)

# Define a new set of points
new_points = np.array([[7, 8], [9, 10], [11, 12]])

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot original and target points on the first subplot
ax1.scatter(source_points[:, 0], source_points[:, 1], label='Source Points', c='blue')
ax1.scatter(target_points[:, 0], target_points[:, 1], label='Target Points', c='red')
ax1.set_title('Original and Target Points')
ax1.legend()

# Apply the transformation to the new points
transformed_points = transform.matrix_transform(new_points, matrix)

# Plot original and transformed points on the second subplot
ax2.scatter(new_points[:, 0], new_points[:, 1], label='Original Points', c='green')
ax2.scatter(transformed_points[:, 0], transformed_points[:, 1], label='Transformed Points', c='purple')
ax2.set_title('Original and Transformed Points')
ax2.legend()

# Show the subplots
plt.show()
