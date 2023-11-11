import random

from src.models.simulation.simulation import Simulation, Agent
from src.modules.communication.model import FakeCommunication
from src.modules.movement.movements import walk_forward
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
                communication=FakeCommunication(
                    refresh_rate=self.refresh_rate,
                    communication_frequency=self.communication_frequency,
                ),
                triangulation=DelaunayTriangulation(
                    agent_id=i,
                    precision=self.triangulation_precision,
                ),
                agent_movement=walk_forward,
            ) for i in range(self.dim)
        ]


class AllStaticButOneRandomPlacementDelaunayTriangulationGlobalCommunicationExperiment(Simulation):
    def setup(self):
        self.agents = [
            Agent(
                0, *self.arena.place_agent_randomly(),
                agents_speed=self.agents_speed,
                communication=FakeCommunication(
                    refresh_rate=self.refresh_rate,
                    communication_frequency=self.communication_frequency,
                ),
                triangulation=DelaunayTriangulation(
                    agent_id=0,
                    precision=self.triangulation_precision,
                ),
                agent_movement=walk_forward,
            ),
            *[Agent(
                i, *self.arena.place_agent_randomly(),
                agents_speed=0,
                communication=FakeCommunication(
                    refresh_rate=self.refresh_rate,
                    communication_frequency=self.communication_frequency,
                ),
                triangulation=DelaunayTriangulation(
                    agent_id=i,
                    precision=self.triangulation_precision,
                ),
                agent_movement=walk_forward,
            ) for i in range(1, self.dim)]
        ]
