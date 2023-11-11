import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# Generate random points within a square
np.random.seed(42)  # For reproducibility
num_points = 20
x = np.random.rand(num_points)
y = np.random.rand(num_points)

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
