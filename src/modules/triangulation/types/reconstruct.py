from src.modules.triangulation.model import DistanceMatrixTriangulation


class ReconstructTriangulation(DistanceMatrixTriangulation):
    # 1. Recover information from other robots
    # 2. Use Delaunay triangulation on smaller matrices (most recent information on 3 agents from each agent)
    # 3. Reconstruct the overall shape of the swarm

    def update_triangulation(self):
        """Override from Triangulation, interface used by the agent to update its triangulation"""
        return self.tri_x, self.tri_y

