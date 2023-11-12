from abc import ABC, abstractmethod

import numpy as np


class Triangulation(ABC):  # abstract class
    def __init__(self, agent_id, precision=1.0, refresh_rate=0.05):
        self.tri_x = []
        self.tri_y = []

        self.agent_id = agent_id

        self.precision = precision
        self.refresh_rate = refresh_rate

    @abstractmethod
    def update_information(self, other_agent_id, distance, information):
        """
        Idea: Could want to use something else than distance matrix
            -> Made more general, each implementation of Triangulation should be able to handle
               the information received from the communication and update its triangulation accordingly
        """
        raise NotImplementedError("Triangulation does not implement a specific information update method")

    @abstractmethod
    def update_triangulation(self):
        """
        Update the triangulation of the swarm
        :return: x and y values of the points triangulated
        """
        raise NotImplementedError("Triangulation does not implement a specific triangulation algorithm")


class DistanceMatrixTriangulation(Triangulation, ABC):
    def __init__(self, agent_id, dim=1, precision=1.0, refresh_rate=0.05):
        super().__init__(
            agent_id=agent_id,
            precision=precision,
            refresh_rate=refresh_rate
        )

        self.distance_matrix = np.zeros((dim, dim), dtype=float)
        self.dim = dim

        self.id_to_index = dict({
            self.agent_id: 0
        })

    def _grow(self):
        self.dim += 1
        new_matrix = np.zeros((self.dim, self.dim), dtype=float)
        new_matrix[0:self.dim - 1, 0:self.dim - 1] = self.distance_matrix
        self.distance_matrix = new_matrix

    def _prune_distance_matrix(self):
        """Property: Only need three agents information to triangulate a fourth one position"""

        matrix = np.zeros((self.dim, self.dim), dtype=float)

        for i in range(self.dim):
            count = 0
            for j in range(i + 1, self.dim):
                if self.distance_matrix[i, j] > 0 and not count >= 3:
                    matrix[i, j] = self.distance_matrix[i, j]
                    count += 1
                else:
                    matrix[i, j] = -1

        return matrix

    def update_information(self, other_agent_id, distance, information):
        if other_agent_id == self.agent_id:
            return

        if other_agent_id not in self.id_to_index:
            self.id_to_index[other_agent_id] = self.dim
            self._grow()

        # Update distance matrix for triangulation
        self.distance_matrix[self.id_to_index[self.agent_id], self.id_to_index[other_agent_id]] = distance
        self.distance_matrix[self.id_to_index[other_agent_id], self.id_to_index[self.agent_id]] = distance

        # Update distance matrix with other agent information
        for agent in information:
            if agent in self.id_to_index:
                self.distance_matrix[self.id_to_index[other_agent_id], self.id_to_index[agent]] = information[agent]
                self.distance_matrix[self.id_to_index[agent], self.id_to_index[other_agent_id]] = information[agent]
            else:
                # agent doesn't know the given robot, does not listen
                pass


class FakeTriangulation(Triangulation):
    """Fake triangulation for testing purposes"""
    def __init__(self):
        super().__init__(
            agent_id=0,
            precision=1.0,
            refresh_rate=0.05
        )

    def update_information(self, other_agent_id, distance, information):
        pass

    def update_triangulation(self):
        return [0], [0]
