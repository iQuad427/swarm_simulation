import math
from welltestpy.tools import triangulate, sym

from src.modules.triangulation.model import DistanceMatrixTriangulation


class DelaunayTriangulation(DistanceMatrixTriangulation):
    """Triangulation of the swarm using a distance matrix and the Delaunay triangulation algorithm"""

    def __init__(self, agent_id, precision=10.0, refresh_rate=0.05):
        super().__init__(agent_id, precision=precision, refresh_rate=refresh_rate)

        self.previous_const = []

    def update_triangulation(self):
        if self.dim < 2:
            return None, None, None

        matrix = self._prune_distance_matrix()
        try:
            const = triangulate(sym(matrix), prec=self.precision)
        except ValueError:
            # print("ValueError: distances not usable for now")
            return None, None, None

        if not const:
            if not self.previous_const:
                self.tri_x = [0]
                self.tri_y = [0]
            return None, None, None

        points = []

        if not self.previous_const:
            points = const[0]
        else:
            current_best = math.inf
            for possibility in const:
                error = 0
                for i in range(
                        len(possibility) if len(possibility) <= len(self.previous_const) else len(self.previous_const)):
                    error += math.sqrt(
                        (possibility[i][0] - self.previous_const[i][0]) ** 2
                        +
                        (possibility[i][1] - self.previous_const[i][1]) ** 2
                    )

                if current_best > error:
                    current_best = error
                    points = possibility

        self.previous_const = points

        if points:
            x = []
            y = []
            for point in points:
                x.append(point[0])
                y.append(point[1])

            self.tri_x = x
            self.tri_y = y

            return self.tri_x, self.tri_y, None

        return None, None, None


class DelaunaySubTriangulation(DelaunayTriangulation):
    """Triangulation of the swarm using a distance matrix and the Delaunay triangulation algorithm"""

    def __init__(self, agent_id, precision=10.0, refresh_rate=0.05):
        super().__init__(agent_id, precision=precision, refresh_rate=refresh_rate)

        self.previous_const = []
        self.index_to_id = [agent_id]

    def update_information(self, other_agent_id, distance, information):
        previous = self.dim

        super().update_information(other_agent_id, distance, information)

        if previous < self.dim:
            self.index_to_id.append(other_agent_id)

    def update_triangulation(self):
        x, y, _ = super().update_triangulation()

        if x is None or y is None:
            return None, None, {}

        num = len(x) if len(x) == len(y) else 0

        res = {
            self.index_to_id[i]: [x[i], y[i]] for i in range(num)
        }

        res[self.agent_id] = [0, 0]

        return x, y, res
