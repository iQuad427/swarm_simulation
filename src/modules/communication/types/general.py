import random

import numpy as np

from src.modules.communication.model import Communication


class GlobalCommunication(Communication):
    def receive_information(self, other_agents: list, context: np.ndarray):
        if random.random() > self.communication_chances:
            return -1, None, None  # Drop the information, did not communicate

        index = round(random.uniform(0, 1) * (len(other_agents) - 1))
        other_agent = other_agents[index]

        if other_agent.id == self.agent_id:
            return -1, None, None  # Drop the information, did not communicate

        information = other_agent.communication.send_information()
        if self.agent_id < other_agent.id:
            distance = context[self.agent_id, other_agent.id]
        else:
            distance = context[other_agent.id, self.agent_id]

        return other_agent.id, distance, information

    def send_information(self):
        # TODO: could want to make noisy response
        return self.data[self.agent_id]
