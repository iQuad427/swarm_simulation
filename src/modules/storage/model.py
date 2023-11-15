from enum import Enum
from abc import ABC, abstractmethod


class DataTypes(Enum):
    information: str = "information"
    agent_id: str = "agent_id"
    distance: str = "distances"
    triangulation: str = "triangulation"
    more: str = "more"


class DataStorage(ABC):
    def __init__(self, agent_id, time_to_live=10, clock_frequency=0.01):
        self.agent_id = agent_id

        self.clock_frequency = clock_frequency if clock_frequency > 0 else 0.01
        self.time_to_live = time_to_live

        # TTL: Time To Live, allow for deciding on what information is outdated
        #  - Usage: when using information, compute the relative time from which
        #           the information was received, and if it is outdated, discard it
        #  - Relative time: time of the DataStorage minus the time of the information
        self.age = 0  # storage time of life (in seconds)

        self.data = {
            # DataTypes.information: {
            #     DataTypes.agent_id: self.agent_id,
            # }
        }

        self.data_age = dict()

    def clock_tick(self):
        self.age += 1 / self.clock_frequency  # allow for the time to be in seconds

    def _get_data_path(self, agent_id, data_type):
        return [f"from_{data_type}_of_{self.agent_id}", f"agent_{agent_id}"]

    def _compute_data_age(self, agent_id, data_type):
        data_age = None
        if data_arrival_time := self._get_data(agent_id, data_type, age=True) is not None:
            data_age = self.age - data_arrival_time

        return data_age

    def _is_outdated(self, agent_id, data_type):
        data_age = self._compute_data_age(agent_id, data_type)

        if data_age is None or data_age > self.time_to_live:
            return True
        else:
            return False

    @abstractmethod
    def _prepare_data_for_sending(self):
        """Should send data + their relative age from the POV of the sender, only if not outdated"""
        raise NotImplementedError("DataStorage does not implement a default data preparation")

    def get_data_to_send(self):
        return self._prepare_data_for_sending()

    def _get_data(self, agent_id, data_type, age=False):
        path = self._get_data_path(agent_id, data_type)

        data = self.data if not age else self.data_age
        for element in path:
            if element in data:
                data = data[element]
            else:
                return None

        return data

    def _set_data(self, agent_id, data_type, data):
        path = self._get_data_path(agent_id, data_type)

        if data_type not in self.data:
            self.data[data_type] = dict()
        if data_type not in self.data_age:
            self.data_age[data_type] = dict()

        data_storage = self.data if not isinstance(data, float) else self.data_age
        for element in path[:-1]:
            if element not in data_storage:
                data_storage[element] = dict()
            data_storage = data_storage[element]
        data_storage[path[-1]] = data

    def set_data(self, agent_id, data_type, data):
        self._set_data(agent_id, data_type, data)

    def get_distance_information(self, agent_id):
        return self._get_data(agent_id, DataTypes.distance)

    def set_distance_information(self, agent_id, distance):
        self.set_data(agent_id, DataTypes.distance, distance)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)


class FakeDataStorage(DataStorage):
    def _prepare_data_for_sending(self):
        return dict()
