import math
import random
import numpy as np


class Agent:
    def __init__(self, id, x, y, radius=5, color=(255, 0, 0), speed=1):
        self.id = id

        self.x = x
        self.y = y

        self.radius = radius
        self.color = color

        self.angle = 0
        self.dx = 0
        self.dy = 0
        self.speed = speed

        self.counter = 0
        self.delay = 100

    def measure_distances(self, agents, noise=0):
        # Return a list of distances to each agent
        distances = [0] * len(agents)
        for agent in agents:
            # Contains (discount_factor, distance_to_agent)
            distances[agent.id] = (1, math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2))

        # Add noise to the distances
        distances = [(discount_factor, distance + np.random.normal(0, noise)) for discount_factor, distance in distances]

        return distances

    def random_move(self, arena_width=500, arena_height=500):
        if self.counter % self.delay == 0:
            self.angle = random.randint(0, 360)
            self.delay = random.randint(100, 200)
            self.counter = 0

            self.dx = math.cos(self.angle * math.pi / 180)
            self.dy = math.sin(self.angle * math.pi / 180)

        self.x += self.dx * self.speed
        self.y += self.dy * self.speed

        # Change the angle if reached side of arena
        if self.x + self.radius > arena_width:
            self.dx = -self.dx
        if self.x - self.radius < 0:
            self.dx = -self.dx
        if self.y + self.radius > arena_height:
            self.dy = -self.dy
        if self.y - self.radius < 0:
            self.dy = -self.dy

        self.counter += 1

    def collide(self, agents):
        for agent in agents:
            if (
                    agent.y - agent.radius < self.y + self.radius < agent.y + agent.radius or agent.y - agent.radius < self.y - self.radius < agent.y + agent.radius
            ) and (
                    agent.x - agent.radius < self.x + self.radius < agent.x + agent.radius or agent.x - agent.radius < self.x - self.radius < agent.x + agent.radius
            ):
                if self.x < agent.x:
                    dx = -1
                else:
                    dx = 1
                if self.y < agent.y:
                    dy = -1
                else:
                    dy = 1

                self.x += dx
                self.y += dy
