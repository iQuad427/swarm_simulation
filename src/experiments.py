from src.modules.communication.types.delaunay import DelaunayNetworkCommunication
from src.modules.communication.types.distance import DistanceLimitedCommunication
from src.modules.communication.types.general import GlobalCommunication
from src.modules.movement.simple import walk_forward, random_walk
from src.modules.storage.types.distance import DistanceOnlyStorage
from src.modules.triangulation.types.delaunay import DelaunayTriangulation
from src.modules.triangulation.types.reconstruct import ReconstructTriangulation
from src.simulation.arena import RectangleArena, Arena
from src.simulation.simulation import Simulation, Agent


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


class AllStaticButOneRandomPlacementDelaunayTriangulationDistanceLimitedCommunicationExperiment(Simulation):
    def __init__(
            self, dim=5, arena: Arena = RectangleArena(xlim=50, ylim=50, width=50, height=50),
            refresh_rate=0.01,  # 10 milliseconds
            agents_speed=0.05,  # 5 centimeters per seconds
            triangulation_precision=1.0,  # 1 meter
            communication_frequency=0.5,  # 500 milliseconds
            communication_radius=10.0,  # 10 meters
    ):
        super().__init__(
            dim=dim, arena=arena,
            refresh_rate=refresh_rate,
            agents_speed=agents_speed,
            triangulation_precision=triangulation_precision,
            communication_frequency=communication_frequency,
        )

        self.communication_radius = communication_radius

    def setup(self):
        self.agents = [
            Agent(
                0, *self.arena.place_agent_randomly(),
                agents_speed=self.agents_speed,
                communication=DistanceLimitedCommunication(
                    agent_id=0,
                    refresh_rate=self.refresh_rate,
                    communication_frequency=self.communication_frequency,
                    radius=self.communication_radius,
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
                communication=DistanceLimitedCommunication(
                    agent_id=i,
                    refresh_rate=self.refresh_rate,
                    communication_frequency=self.communication_frequency,
                    radius=self.communication_radius,
                ),
                triangulation=DelaunayTriangulation(
                    agent_id=i,
                    precision=self.triangulation_precision,
                ),
                agent_movement=walk_forward,
            ) for i in range(1, self.dim)]
        ]


class AllStaticRectanglePlacementExperiment(Simulation):
    def __init__(
            self, dim=5, arena: Arena = RectangleArena(xlim=50, ylim=50, width=50, height=50),
            refresh_rate=0.01,  # 10 milliseconds
            agents_speed=0.05,  # 5 centimeters per seconds
            triangulation_precision=1.0,  # 1 meter
            communication_frequency=0.5,  # 500 milliseconds
            communication_radius=10.0,  # 10 meters
    ):
        super().__init__(
            dim=16, arena=arena,
            refresh_rate=refresh_rate,
            agents_speed=agents_speed,
            triangulation_precision=triangulation_precision,
            communication_frequency=communication_frequency,
        )

        self.arena = RectangleArena(xlim=50, ylim=50, width=50, height=50)
        self.communication_radius = communication_radius

    def setup(self):

        placement = [
            [10, 10], [10, 20], [10, 30], [10, 40],
            [20, 10], [20, 20], [20, 30], [20, 40],
            [30, 10], [30, 20], [30, 30], [30, 40],
            [40, 10], [40, 20], [40, 30], [40, 40],
        ]

        self.agents = [
            Agent(
                i, *placement[i],
                agents_speed=0,
                communication=DistanceLimitedCommunication(
                    agent_id=i,
                    refresh_rate=self.refresh_rate,
                    communication_frequency=self.communication_frequency,
                    radius=self.communication_radius,
                ),
                triangulation=ReconstructTriangulation(
                    agent_id=i,
                    precision=self.triangulation_precision,
                ),
                agent_movement=walk_forward,
            ) for i in range(self.dim)
        ]


class DelaunayCommunicationExperiment(Simulation):
    def setup(self):
        self.agents = [
            Agent(
                i, *self.arena.place_agent_randomly(),
                agents_speed=self.agents_speed,
                communication=DelaunayNetworkCommunication(
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


class TestExperiment(Simulation):
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
                data_storage=DistanceOnlyStorage(
                    agent_id=i,
                ),
                agent_movement=walk_forward,
            ) for i in range(self.dim)
        ]