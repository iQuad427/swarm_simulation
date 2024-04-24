import matplotlib.pyplot as plt
import numpy as np


def generate_spiral(num_points, num_turns, branch_spacing, rotation):
    theta = np.linspace(0, num_turns * 2 * np.pi, num_points)
    r = branch_spacing * theta / (2 * np.pi)
    x = r * np.cos(theta + rotation)
    y = r * np.sin(theta + rotation)
    return x, y


num_points = 50
num_turns = 0.25
branch_spacing = 1000
rotations = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # Rotations for each spiral

plt.figure(figsize=(8, 8))
for i, rotation in enumerate(rotations):
    x, y = generate_spiral(num_points, num_turns, branch_spacing, rotation)
    plt.plot(x, y, label=f"Spiral {i+1}")

plt.axis('equal')
plt.title('Four-Branch Spirals with Rotation')
plt.legend()
plt.show()
