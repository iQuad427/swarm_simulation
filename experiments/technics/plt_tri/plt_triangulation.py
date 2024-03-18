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

# Print neighbour of

# Plot the original points
plt.scatter(x, y, c='blue', label='Original Points')
plt.scatter(x[0], y[0], c='red')
# Plot neighbours of the first point
for i, j in triangulation.edges:
    print(i, j)
    if i == 0:
        print(x[j], y[j])
        plt.scatter(x[j], y[j], c='green')
    elif j == 0:
        print(x[i], y[i])
        plt.scatter(x[i], y[i], c='green')


# Set plot properties
plt.title('Triangulation of Random Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.savefig("triangulation.png", dpi=300)
plt.show()
