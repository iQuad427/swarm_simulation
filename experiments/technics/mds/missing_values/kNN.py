import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import pairwise_distances

from experiments.utils.generate_matrix import generate_points_and_matrix
from experiments.utils.plot import plot_on_top
from experiments.utils.rotation import find_rotation_matrix

# Create a sample distance matrix with missing values
# np.random.seed(42)
size = 20
points, dist_matrix = generate_points_and_matrix(size, 10, 0)

save_dist_matrix = dist_matrix.copy()

print("Original Distance Matrix:")
print(save_dist_matrix)

# Create a mask indicating missing values
missing_rate = 0.5
mask = np.random.rand(size, size) < missing_rate

# Make the mask symmetric
mask = np.triu(mask) | np.tril(mask.T)

# Set the diagonal to False
np.fill_diagonal(mask, False)

# Set the missing values to NaN
dist_matrix[mask] = np.nan

# Convert the distance matrix to a dissimilarity matrix (required for KNN imputation)
dissimilarity_matrix = 1 - dist_matrix


# Impute missing values using k-NN imputation
def impute_distance_matrix_knn(dissimilarity_matrix, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors, metric="nan_euclidean")
    imputed_dissimilarity_matrix = imputer.fit_transform(dissimilarity_matrix, y=None)
    imputed_dist_matrix = 1 - imputed_dissimilarity_matrix  # Convert back to distance matrix
    return imputed_dist_matrix


# Impute missing values in the distance matrix using k-NN
imputed_dist_matrix_knn = impute_distance_matrix_knn(dissimilarity_matrix)

# Print the original and imputed distance matrices
print("NaN Distance Matrix:")
print(dist_matrix)
print("\nImputed Distance Matrix (k-NN):")
print(imputed_dist_matrix_knn)

# imshow the error matrix
import matplotlib.pyplot as plt

print("Error Matrix:")
print(imputed_dist_matrix_knn - save_dist_matrix)
plt.imshow(imputed_dist_matrix_knn - save_dist_matrix)
plt.colorbar()

# Use MDS to visualize the original and imputed distance matrices
from sklearn.manifold import MDS

# Make the distance matrix symmetric again
imputed_dist_matrix_knn = (imputed_dist_matrix_knn + imputed_dist_matrix_knn.T) / 2

# Perform classical multidimensional scaling
mds = MDS(n_components=2, dissimilarity='precomputed', metric=True, normalized_stress=False)
mds_coors = mds.fit_transform(imputed_dist_matrix_knn)

rotation = find_rotation_matrix(points.T, mds_coors.T)

# Apply the rotation
mds_coors_rotated = mds_coors @ rotation


# First, find the centroid of the original points
centroid = np.mean(points, axis=0)

# Then, find the centroid of the MDS points
mds_centroid = np.mean(mds_coors_rotated, axis=0)

# Find the translation vector
translation = centroid - mds_centroid

# Translate the MDS points
mds_coors_translated = mds_coors_rotated + translation

# Plot the results
plot_on_top(points, mds_coors_translated)

print("Mean Squared Error:", np.mean((mds_coors_translated - points) ** 2))
