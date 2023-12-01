import copy
import math
from queue import PriorityQueue

import numpy as np
from skimage import transform

from src.modules.storage.model import DataTypes
from src.modules.triangulation.model import Triangulation
from src.modules.triangulation.types.delaunay import DelaunaySubTriangulation


class ReconstructTriangulation(Triangulation):
    # 1. Recover information from other robots
    # 2. Use Delaunay triangulation on smaller matrices (most recent information on 3 agents from each agent)
    # 3. Reconstruct the overall shape of the swarm

    def __init__(self, agent_id, precision=1.0, refresh_rate=0.05):
        super().__init__(
            agent_id=agent_id,
            precision=precision,
            refresh_rate=refresh_rate
        )

        self.triangulation_data = {
            self.agent_id: dict()
        }

        self.sub_triangulation = DelaunaySubTriangulation(
            agent_id=agent_id,
            precision=precision,
            refresh_rate=refresh_rate
        )

    def update_information(self, other_agent_id, distance, information):
        """Override from Triangulation, interface used by the agent to update its information for triangulation"""

        # Update sub triangulation
        self.sub_triangulation.update_information(other_agent_id, distance, information)

        if self.agent_id == 0:
            print("New Information:", information)

        if DataTypes.triangulation.value in information:
            self.triangulation_data[other_agent_id] = information[DataTypes.triangulation.value]

        x, y, other_points = self.sub_triangulation.update_triangulation()

        self.triangulation_data[self.agent_id] = other_points

        if self.agent_id == 0:
            print("Total Information:", self.triangulation_data)


    def update_triangulation(self):
        """Override from Triangulation, interface used by the agent to update its triangulation"""

        robot_target_knowledge = self.triangulation_data[self.agent_id]

        if not robot_target_knowledge:
            return None, None, {}

        sources = self.triangulation_data.keys()
        priority_queue = PriorityQueue()

        for source in sources:
            if source == self.agent_id or len(self.triangulation_data[source]) < 2:
                continue

            # Robot knowledge
            robot_source_knowledge = self.triangulation_data[source]

            # Extract common points
            common_points = set(robot_source_knowledge.keys()) & set(robot_target_knowledge.keys())

            priority_queue.put((-len(common_points), source))

        transformed_robot_knowledge = robot_target_knowledge
        while not priority_queue.empty():
            source = priority_queue.get()[1]

            if source == self.agent_id or len(self.triangulation_data[source]) < 2:
                continue

            # Robot knowledge
            robot_source_knowledge = self.triangulation_data[source]

            # Extract common points
            common_points = set(robot_source_knowledge.keys()) & set(robot_target_knowledge.keys())

            if len(common_points) < 2:
                continue

            # Convert common points to numpy arrays
            source_points = np.array([robot_source_knowledge[point] for point in common_points])
            source_points_sym = source_points * np.array([1, -1])  # same point but with a symmetry over X axis

            target_points = np.array([robot_target_knowledge[point] for point in common_points])

            # Estimate the transformation matrix
            matrix = transform.estimate_transform('euclidean', source_points, target_points)
            matrix_sym = transform.estimate_transform('euclidean', source_points_sym, target_points)

            # Apply the transformation
            transformed_robot_knowledge = {}
            transformed_robot_knowledge_sym = {}

            error = 0
            error_sym = 0

            for point, coord in robot_source_knowledge.items():
                transformed_robot_knowledge[point] = list(transform.matrix_transform(
                    np.array(coord), matrix
                )[0])

                transformed_robot_knowledge_sym[point] = list(transform.matrix_transform(
                    np.array(coord), matrix_sym
                )[0])

                # error with the target point
                if point in common_points:
                    error += math.sqrt(
                        (transformed_robot_knowledge[point][0] - robot_target_knowledge[point][0]) ** 2 +
                        (transformed_robot_knowledge[point][1] - robot_target_knowledge[point][1]) ** 2
                    )
                    error_sym += math.sqrt(
                        (transformed_robot_knowledge_sym[point][0] - robot_target_knowledge[point][0]) ** 2 +
                        (transformed_robot_knowledge_sym[point][1] - robot_target_knowledge[point][1]) ** 2
                    )

            # Verify the one that better fits
            if error_sym < error:
                transformed_robot_knowledge = transformed_robot_knowledge_sym

            for point in robot_target_knowledge:
                transformed_robot_knowledge[point] = robot_target_knowledge[point]

                # TODO: try to re-implement this correctly when symmetry check has been implemented
                # if point in transformed_robot_knowledge:
                #     a = transformed_robot_knowledge[point]
                #     b = robot_target_knowledge[point]
                #
                #     # if self.agent_id == 0:
                #     #     print("A:", a)
                #     #     print("B:", b)
                #
                #     transformed_robot_knowledge[point] = [
                #         (a[0] + b[0]) / 2,
                #         (a[1] + b[1]) / 2
                #     ]
                # else:
                #     transformed_robot_knowledge[point] = robot_target_knowledge[point]

            robot_target_knowledge = transformed_robot_knowledge

        x = []
        y = []
        for point in robot_target_knowledge.values():
            x.append(point[0])
            y.append(point[1])

        self.tri_x = x
        self.tri_y = y

        return self.tri_x, self.tri_y, robot_target_knowledge
