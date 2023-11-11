import random

from src.models.simulation.simulation import Simulation, Agent
from src.modules.triangulation.types.delaunay import DelaunayTriangulation


class SameSpeedRandomPlacementNoTriangulationNoCommunicationExperiment(Simulation):
    def setup(self):
        self.agents = [
            Agent(
                i, *self.arena.place_agent_randomly(),
                agents_speed=self.agents_speed,
            ) for i in range(self.dim)
        ]


class SameSpeedRandomPlacementDelaunayTriangulationGlobalCommunicationExperiment(Simulation):
    def setup(self):
        self.agents = [
            Agent(
                i, *self.arena.place_agent_randomly(),
                agents_speed=self.agents_speed,
                communication_chances=self.communication_chances,
                triangulation=DelaunayTriangulation(
                    agent_id=0,
                    precision=self.triangulation_precision,
                ),
            ) for i in range(self.dim)
        ]


class AllStaticButOneRandomPlacementDelaunayTriangulationGlobalCommunicationExperiment(Simulation):
    def setup(self):
        self.agents = [
            Agent(
                0, *self.arena.place_agent_randomly(),
                agents_speed=self.agents_speed,
                communication_chances=self.communication_chances,
                triangulation=DelaunayTriangulation(
                    agent_id=0,
                    precision=self.triangulation_precision,
                ),
            ),
            *[Agent(
                i, *self.arena.place_agent_randomly(),
                agents_speed=0,
                communication_chances=self.communication_chances,
                triangulation=DelaunayTriangulation(
                    agent_id=0,
                    precision=self.triangulation_precision,
                ),
            ) for i in range(1, self.dim)]
        ]
