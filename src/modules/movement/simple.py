import random
import math


def walk_forward(self):
    # Make the robot walk forward
    self.x += self.speed * self.dx
    self.y += self.speed * self.dy


def random_walk(self):
    # Simulate random movement
    if random.random() < 0.001:
        self.angle += random.uniform(-60, 60)

        self.dx = math.cos(self.angle)
        self.dy = math.sin(self.angle)

    self.x += self.speed * self.dx
    self.y += self.speed * self.dy
