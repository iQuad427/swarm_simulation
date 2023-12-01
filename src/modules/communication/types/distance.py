import random
import numpy as np

from src.modules.communication.model import Communication


class DistanceLimitedCommunication(Communication):
    def __init__(self, agent_id, refresh_rate=0.01, communication_frequency=0.1, radius=10):
        super().__init__(
            agent_id=agent_id,
            refresh_rate=refresh_rate,
            communication_frequency=communication_frequency,
        )

        self.radius = radius

    def receive_information(self, other_agents: list, context: np.ndarray):
        in_range_agents = []
        for agent in other_agents:
            if self.agent_id < agent.id:
                if context[self.agent_id, agent.id] <= self.radius:
                    in_range_agents.append(agent)
            elif agent.id < self.agent_id:
                if context[agent.id, self.agent_id] <= self.radius:
                    in_range_agents.append(agent)

        if len(in_range_agents) == 0:
            return -1, None, None

        index = round(random.uniform(0, 1) * (len(in_range_agents) - 1))
        other_agent = in_range_agents[index]

        information = other_agent.communication.send_information()

        if self.agent_id < other_agent.id:
            distance = context[self.agent_id, other_agent.id]
        else:
            distance = context[other_agent.id, self.agent_id]

        return other_agent.id, distance, information

    def send_information(self):
        return self.data
