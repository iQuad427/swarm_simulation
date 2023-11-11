import math
import numpy as np
from welltestpy.tools import triangulate, sym

from src.modules.triangulation.model import Triangulation


class DelaunayTriangulation(Triangulation):
    def update_triangulation(self):
        if self.dim < 2:
            return None, None

        matrix = self._prune_distance_matrix()
        try:
            const = triangulate(sym(matrix), prec=self.precision)
        except ValueError:
            # print("ValueError: distances not usable for now")
            return None, None

        if not const:
            if not self.previous_const:
                self.tri_x = [0]
                self.tri_y = [0]
            return None, None

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

            return self.tri_x, self.tri_y

        return None, None
