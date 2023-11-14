import copy

import numpy as np
from skimage import transform

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

        # if self.agent_id == 0:
        #     print("updating triangulation information".upper())

        # Update sub triangulation
        self.sub_triangulation.update_information(other_agent_id, distance, information)

        # if self.agent_id == 0:
        #     print("triangulation information: ".upper(), information)

        if "more" in information:
            self.triangulation_data[other_agent_id] = information["more"]

        x, y, other_points = self.sub_triangulation.update_triangulation()

        # if self.agent_id == 0:
        #     print("other points: ".upper(), other_points)

        self.triangulation_data[self.agent_id] = other_points

        # if self.agent_id == 0:
        #     print("triangulation data: ".upper(), self.triangulation_data)

    def update_triangulation(self):
        """Override from Triangulation, interface used by the agent to update its triangulation"""

        # if self.agent_id == 0:
        #     print("TRIANGULATION DATA:", self.triangulation_data)

        robot_target_knowledge = self.triangulation_data[self.agent_id]

        # if self.agent_id == 0:
        #     print("TARGET KNOWLEDGE:", robot_target_knowledge)

        if not robot_target_knowledge:
            return None, None, {}

        transformed_robot_knowledge = robot_target_knowledge
        for source in copy.deepcopy(self.triangulation_data):
            if source == self.agent_id or len(self.triangulation_data[source]) < 2:
                continue

            # Robot knowledge
            robot_source_knowledge = self.triangulation_data[source]

            if self.agent_id == 0:
                print("SOURCE KNOWLEDGE:", robot_source_knowledge)

            # Extract common points
            common_points = set(robot_source_knowledge.keys()) & set(robot_target_knowledge.keys())

            if self.agent_id == 0:
                print("COMMON POINTS:", common_points)

            if len(common_points) < 2:
                continue

            # Convert common points to numpy arrays
            source_points = np.array([robot_source_knowledge[point] for point in common_points])
            target_points = np.array([robot_target_knowledge[point] for point in common_points])

            if self.agent_id == 0:
                print("SOURCE POINTS:", source_points)
                print("TARGET POINTS:", target_points)

            # Estimate the transformation matrix
            matrix = transform.estimate_transform('euclidean', source_points, target_points)

            # if self.agent_id == 0:
            #     all_source_points = np.array([robot_source_knowledge[point] for point in robot_source_knowledge])
            #     all_points_transformed = transform.matrix_transform(all_source_points, matrix)
            #     print("TRANSFORMED KNOWLEDGE BIS", all_points_transformed)

            # Apply the transformation to robot_1_knowledge
            transformed_robot_knowledge = {
                point: transform.matrix_transform(
                    np.array(coord), matrix
                )[0] for point, coord in robot_source_knowledge.items()
            }

            if self.agent_id == 0:
                print("TRANSFORMED KNOWLEDGE:", transformed_robot_knowledge)

            for point in robot_target_knowledge:
                transformed_robot_knowledge[point] = robot_target_knowledge[point]

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

            # print("ROBOT FINAL KNOWLEDGE:", robot_target_knowledge)

        # if self.agent_id == 0:
        #     print("ROBOT_TARGET:", robot_target_knowledge)

        x = []
        y = []
        for point in robot_target_knowledge.values():
            x.append(point[0])
            y.append(point[1])

        self.tri_x = x
        self.tri_y = y

        return self.tri_x, self.tri_y, robot_target_knowledge
