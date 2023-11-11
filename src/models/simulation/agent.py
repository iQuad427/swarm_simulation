# Define the Agent class
import random
import time

import math
import numpy as np

from src.modules.communication.model import Communication, FakeCommunication
from src.modules.movement.movements import walk_forward
from src.modules.triangulation.model import Triangulation, FakeTriangulation


class Agent:
    def __init__(
            self, agent_id, x, y,
            agents_speed=0.01,
            triangulation: Triangulation = FakeTriangulation(agent_id=0),
            communication: Communication = FakeCommunication(),
            agent_movement=walk_forward,
    ):
        self.id = agent_id
        self.x = x
        self.y = y

        self.radius = 1
        self.color = [255, 255, 255]

        self.paused = True

        self.communication = communication
        self.communicate = True

        self.triangulation = triangulation
        self.triangulate = True

        self.move = agent_movement

        self.tri_y = []
        self.tri_x = []

        self.data = dict()
        self.data[self.id] = dict()

        self.speed = agents_speed
        self.angle = random.randint(0, 360)
        self.dx = math.cos(self.angle)
        self.dy = math.sin(self.angle)

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

    def receive_information(self, other_agents: list, context: np.ndarray):
        index = round(random.uniform(0, 1) * (len(other_agents) - 1))
        other_agent = other_agents[index]

        if other_agent.id == self.id:
            return

        information = other_agent.send_information()
        if self.id < other_agent.id:
            distance = context[self.id, other_agent.id]
        else:
            distance = context[other_agent.id, self.id]

        self.data[self.id][other_agent.id] = distance
        self.data[other_agent.id] = information

        # TODO: add distance computation errors, noise, etc.
        self.triangulation.update_distance_matrix(other_agent.id, information, distance)

    def send_information(self):
        return self.data[self.id]

    def triangulation_handler(self):
        while self.triangulate:
            if self.paused:
                # Avoid causing the thread to over-consume CPU
                time.sleep(self.triangulation.refresh_rate)
                continue

            x, y = self.triangulation.update_triangulation()

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

            if random.random() < self.communication.communication_chances:
                self.receive_information(agents, context)

            time.sleep(self.communication.refresh_rate)

    def stop(self):
        self.triangulate = False
        self.communicate = False
