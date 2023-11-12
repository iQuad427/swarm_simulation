from src.simulation.simulation import Simulation, Agent
from src.modules.communication.model import FakeCommunication
from src.modules.communication.types.general import GlobalCommunication
from src.modules.movement.simple import walk_forward, random_walk
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
                communication=GlobalCommunication(
                    agent_id=i,
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
                communication=GlobalCommunication(
                    agent_id=0,
                    refresh_rate=self.refresh_rate,
                    communication_frequency=self.communication_frequency,
                ),
                triangulation=DelaunayTriangulation(
                    agent_id=0,
                    precision=self.triangulation_precision,
                ),
                agent_movement=random_walk,
            ),
            *[Agent(
                i, *self.arena.place_agent_randomly(),
                agents_speed=0,
                communication=GlobalCommunication(
                    agent_id=i,
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
