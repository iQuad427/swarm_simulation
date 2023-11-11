import random
from dataclasses import dataclass

import math

from src.models.simulation.simulation import Agent


@dataclass()
class Arena:

    def collide(self, agent: Agent):
        # TODO: implement collision detection for agents in the different arenas
        raise NotImplementedError("Arena does not implement a collision detection")

    def place_agent_randomly(self):
        raise NotImplementedError("Arena does not implement a random agent placement")

    def area(self):
        raise NotImplementedError("Arena does not implement a specific area computation")

    def perimeter(self):
        raise NotImplementedError("Arena does not implement a specific perimeter computation")


@dataclass()
class RectangleArena(Arena):
    width: float
    height: float
    type: str = "rectangle"

    def place_agent_randomly(self):
        return random.randint(0, int(self.width)), random.randint(0, int(self.height))

    def collide(self, agent: Agent):
        if agent.x - agent.radius < 0:
            agent.dx = 1
        elif agent.x + agent.radius > self.width:
            agent.dx = -1

        if agent.y - agent.radius < 0:
            agent.dy = 1
        elif agent.y + agent.radius > self.height:
            agent.dy = -1

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)


@dataclass()
class CircleArena(Arena):
    radius: float
    type: str = "circle"

    def place_agent_randomly(self):
        angle = random.uniform(0, 2 * math.pi)
        return self.radius * math.cos(angle), self.radius * math.sin(angle)

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

    def collide(self, agent: Agent):
        raise NotImplementedError("CircleArena collision behavior not implemented")
