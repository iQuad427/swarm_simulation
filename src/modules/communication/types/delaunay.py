import random

import numpy as np
from matplotlib.tri import Triangulation

from src.modules.communication.model import Communication


class DelaunayNetworkCommunication(Communication):
    """Communication that focus on the neighbours of an agent, using the Delaunay triangulation."""
    def receive_information(self, other_agents: list, context: np.ndarray):
        x, y = [], []
        for i, agent in enumerate(other_agents):
            x.append(agent.x)
            y.append(agent.y)

        x = np.array(x)
        y = np.array(y)

        triangulation = Triangulation(x, y)

        neighbours = []
        for edge in triangulation.edges:
            if edge[0] == self.agent_id:
                neighbours.append(edge[1])
            elif edge[1] == self.agent_id:
                neighbours.append(edge[0])

        if len(neighbours) == 0:
            return -1, None, None

        index = round(random.uniform(0, 1) * (len(neighbours) - 1))

        if index < 0:
            return -1, None, None

        other_agent = other_agents[index]

        if other_agent.id == self.agent_id:
            return -1, None, None

        information = other_agent.communication.send_information()

        if self.agent_id < other_agent.id:
            distance = context[self.agent_id, other_agent.id]
        else:
            distance = context[other_agent.id, self.agent_id]

        return other_agent.id, distance, information

    def send_information(self):
        return self.data
