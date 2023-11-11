

class Communication:
    def __init__(self, refresh_rate=0.01, communication_frequency=0.5):
        self.refresh_rate = refresh_rate
        self.communication_chances = self.refresh_rate / communication_frequency


class FakeCommunication(Communication):
    pass


