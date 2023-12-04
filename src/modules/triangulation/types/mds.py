import math
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS

from src.modules.triangulation.model import DistanceMatrixTriangulation


class MDScaleTriangulation(DistanceMatrixTriangulation):
    """Triangulation of the swarm using a distance matrix and the Delaunay triangulation algorithm"""

    def __init__(self, agent_id, precision=10.0, refresh_rate=0.05):
        super().__init__(agent_id, precision=precision, refresh_rate=refresh_rate)

    def update_triangulation(self):
        if self.dim < 3:
            return None, None, None

        try:
            # Convert the matrix to a distance matrix
            d = distance_matrix(self.distance_matrix, self.distance_matrix)

            # Perform classical multidimensional scaling
            mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, metric=True, random_state=1)
            mds_coors = mds.fit_transform(d)

            # Results
            x, y = mds_coors[:, 0], mds_coors[:, 1]
            x, y = x.tolist(), y.tolist()
        except ValueError:
            # print("ValueError: distances not usable for now")
            return None, None, None

        self.tri_x = x
        self.tri_y = y

        return self.tri_x, self.tri_y, None

