# Define the Agent class
import math
import random
import time

from src.modules.communication.model import Communication, FakeCommunication
from src.modules.movement.simple import walk_forward
from src.modules.storage.model import DataStorage, FakeDataStorage
from src.modules.triangulation.model import Triangulation, FakeTriangulation


class Agent:
    def __init__(
            self, agent_id, x, y,
            agents_speed=0.01,
            triangulation: Triangulation = FakeTriangulation(),
            communication: Communication = FakeCommunication(),
            data_storage: DataStorage = FakeDataStorage(),
            agent_movement=walk_forward,
    ):
        # Define the agent's properties
        self.id = agent_id
        self.radius = 1
        self.color = [255, 255, 255]

        # Define the agent's position
        self.x = x
        self.y = y

        # Define the agent's starting movement
        self.speed = agents_speed
        self.angle = random.randint(0, 360)
        self.dx = math.cos(self.angle)
        self.dy = math.sin(self.angle)
        self.move = agent_movement

        # Define the agent's state
        self.paused = False
        self.communicate = True
        self.triangulate = True

        # Define the agent's communication and triangulation
        self.communication = communication
        self.triangulation = triangulation

        # Define the agent's memory
        self.tri_y = []
        self.tri_x = []

        # TODO: add a TTL mechanism to avoid overwriting new information with old ones
        #       also, might want to add an update interface for the data; add a data handler
        self.data = data_storage

        self.data = {
            self.id: {
                "distances": dict(),
                "more": dict()  # For additional triangulation information
            },
        }

        # TTL: Time To Live, allow for deciding on what information is outdated
        self.data_ttl = {
            self.id: {
                "distances": dict(),
                "more": dict()
            },
        }

    def __str__(self):
        return f"agent_{self.id}"

    def move(self):
        raise NotImplementedError("Agent does not implement a default movement")

    def collide(self, agents: list):
        for agent in agents:
            if (
                    agent.y - agent.radius < self.y + self.radius < agent.y + agent.radius or agent.y - agent.radius < self.y - self.radius < agent.y + agent.radius
            ) and (
                    agent.x - agent.radius < self.x + self.radius < agent.x + agent.radius or agent.x - agent.radius < self.x - self.radius < agent.x + agent.radius
            ):
                if self.x < agent.x:
                    self.dx = -1
                else:
                    self.dx = 1
                if self.y < agent.y:
                    self.dy *= -1
                else:
                    self.dy *= 1

    def triangulation_handler(self):
        while self.triangulate:
            if self.paused:
                # Avoid causing the thread to over-consume CPU
                time.sleep(self.triangulation.refresh_rate)
                continue

            x, y, more = self.triangulation.update_triangulation()

            if more is not None and isinstance(more, dict):
                self.data[self.id]["more"] = more

            if x is not None and y is not None:
                self.tri_x = x
                self.tri_y = y

            time.sleep(self.triangulation.refresh_rate)

    def communication_handler(self, agents, context):
        while self.communicate:
            if self.paused:
                # Avoid causing the thread to over-consume CPU
                time.sleep(self.communication.refresh_rate)
                continue

            # Receive information from another agent
            agent_id, distance, information = self.communication.receive_information(agents, context)

            if agent_id == -1:
                # Drop the information, did not communicate
                continue

            # TODO: modify to meet the new data storage requirements

            # THIS IS NO CODE: just a representation of the data structure
            # self.data = {
            #     agent.id: {
            #         "distances": dict(),
            #         "more": dict()
            #     } for agent in agents
            # }
            self.data[self.id]["distances"][agent_id] = distance

            # THIS IS NO CODE: just a representation of the data structure
            # information = {
            #     "distances": dict(),
            #     "more": dict()
            # }
            self.data[agent_id] = information

            self.communication.data = self.data

            # if self.id == 0:
            #     print("AGENT DATA:", self.data)

            # Update triangulation with the new information
            self.triangulation.update_information(agent_id, distance, information)

            # TODO: execute DataStorage clock tick

            time.sleep(self.communication.refresh_rate)

    def stop(self):
        self.triangulate = False
        self.communicate = False
