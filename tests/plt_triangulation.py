import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

# Generate random points within a square
np.random.seed(42)  # For reproducibility
num_points = 20
x = np.random.rand(num_points)
y = np.random.rand(num_points)

# placement = [
#     [10, 10], [10, 20], [10, 30], [10, 40],
#     [20, 10], [20, 20], [20, 30], [20, 40],
#     [30, 10], [30, 20], [30, 30], [30, 40],
#     [40, 10], [40, 20], [40, 30], [40, 40],
# ]
#
# x = [point[0] for point in placement]
# y = [point[1] for point in placement]

# Create a Triangulation object
triangulation = Triangulation(x, y)

# Plot the triangulation
plt.triplot(triangulation, c='red', label='Triangulation')

print(triangulation.edges)
print(triangulation.triangles)
print(triangulation.neighbors)

# Plot the original points
plt.scatter(x, y, c='blue', label='Original Points')

# Set plot properties
plt.title('Triangulation of Random Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()
