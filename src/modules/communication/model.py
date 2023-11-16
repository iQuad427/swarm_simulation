import numpy as np


class Communication:
    def __init__(self, agent_id, refresh_rate=0.01, communication_frequency=0.5):
        self.agent_id = agent_id
        self.refresh_rate = refresh_rate
        self.communication_chances = self.refresh_rate / communication_frequency

        self.data = dict()

    def receive_information(self, other_agents: list, context: np.ndarray):
        raise NotImplementedError("Communication does not implement a default receive_information")

    def send_information(self):
        raise NotImplementedError("Communication does not implement a default send_information")


class FakeCommunication(Communication):
    def __init__(self, agent_id=0, refresh_rate=0.01, communication_frequency=0.5):
        super().__init__(
            agent_id=agent_id,
            refresh_rate=refresh_rate,
            communication_frequency=communication_frequency
        )

    def receive_information(self, other_agents: list, context: np.ndarray):
        return -1, None, None

    def send_information(self):
        pass
