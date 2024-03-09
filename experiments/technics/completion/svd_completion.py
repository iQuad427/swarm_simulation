import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

from experiments.utils.plot import plot_side_by_side


# Function to complete a distance matrix using SVD
def complete_distance_matrix(data_matrix):
    # Calculate pairwise distances
    pairwise_distances = pdist(data_matrix, metric='euclidean')

    print(data_matrix)
    print(pairwise_distances)

    # Convert to a square distance matrix
    distance_matrix = squareform(pairwise_distances)

    print(distance_matrix)

    # Replace NaN values with the mean of non-NaN values
    mean_distance = np.nanmean(pairwise_distances)
    distance_matrix[np.isnan(distance_matrix)] = mean_distance

    return distance_matrix


# Function to perform MDS and plot the points in 2D
def get_mds(distance_matrix):
    # Apply classical MDS to reduce dimensionality to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
    embedded_points = mds.fit_transform(distance_matrix)

    return embedded_points


# Example usage
if __name__ == "__main__":
    # Use a real distance matrix
    data_matrix = np.array([[0, 1, 2, 3, 4],
                            [1, 0, 1, 2, 3],
                            [2, 1, 0, 1, 2],
                            [3, 2, 1, 0, 1],
                            [4, 3, 2, 1, 0]])

    data_matrix_2 = np.array([[0, 1, 2, 3, 4],
                              [1, 0, 1, 2, 3],
                              [2, 1, 0, np.NAN, 2],
                              [3, 2, 1, 0, 1],
                              [4, 3, 2, 1, 0]])

    # Complete distance matrix using SVD
    distance_matrix = complete_distance_matrix(data_matrix_2)

    print(data_matrix)
    print(distance_matrix)

    # Perform MDS and plot the points in 2D
    plot_side_by_side(get_mds(data_matrix), get_mds(distance_matrix))
