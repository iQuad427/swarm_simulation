import numpy as np
from sklearn.metrics import pairwise_distances
from funk_svd import SVD  # pip install git+https://github.com/gbolmier/funk-svd

# SVD only works for Python 3.10 and earlier

# Create a sample distance matrix with missing values
np.random.seed(42)
size = 10
dist_matrix = np.random.rand(size, size)
dist_matrix[np.tril_indices(size)] = np.nan  # Setting lower triangle to NaN to simulate missing values


# Impute missing values using matrix factorization (SVD)
def impute_distance_matrix(dist_matrix, rank=3, max_iter=1000):
    # Mask indicating missing values
    mask = np.isnan(dist_matrix)

    # Initialize SVD model
    svd = SVD(n_iters=max_iter, rank=rank)

    # Fit the model
    svd.fit(dist_matrix)

    # Impute missing values
    imputed_matrix = svd.predict_all()

    # Apply the mask to retain original non-missing values
    imputed_matrix[mask] = dist_matrix[mask]

    return imputed_matrix

# Impute missing values in the distance matrix
imputed_dist_matrix = impute_distance_matrix(dist_matrix)

# Print the original and imputed distance matrices
print("Original Distance Matrix:")
print(dist_matrix)
print("\nImputed Distance Matrix:")
print(imputed_dist_matrix)
