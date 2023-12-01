# Define the Agent class
import math
import random
import time

from src.modules.communication.model import Communication, FakeCommunication
from src.modules.movement.simple import walk_forward
from src.modules.storage.model import DataStorage, FakeDataStorage, DataTypes
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

        self.data = data_storage

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

            x, y, triangulation = self.triangulation.update_triangulation()

            if triangulation is not None and isinstance(triangulation, dict):
                self.data.set_information(
                    self.id,
                    information={
                        DataTypes.triangulation.value: triangulation
                    },
                )

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

            if self.id == 0:
                print("Starting Communication")

            # Receive information from another agent
            agent_id, distance, information = self.communication.receive_information(agents, context)

            if self.id == 0:
                print(f"Received Information from {agent_id}:", distance, information)

            if agent_id == -1:
                # Drop the information, did not communicate
                continue

            if self.id == 0:
                print("Updating Data")

            self.data.set_distance(agent_id, distance)
            self.data.set_information(agent_id, information)

            if self.id == 0:
                print("New Data:", self.data)
                print("Updating Data to Send")

            self.communication.data = self.data.get_data_to_send()

            if self.id == 0:
                print("Data to Send:", self.communication.data)

            # Update triangulation with the new information
            self.triangulation.update_information(agent_id, distance, information)

            # Execute clock tick in data storage so that it can measure relative time
            self.data.clock_tick()

            time.sleep(self.communication.refresh_rate)

    def stop(self):
        self.triangulate = False
        self.communicate = False
