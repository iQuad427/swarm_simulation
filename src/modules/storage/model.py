from enum import Enum
from abc import ABC, abstractmethod


class DataTypes(Enum):
    information: str = "information"
    agent_id: str = "agent_id"
    distance: str = "distances"
    triangulation: str = "triangulation"
    custom: str = "custom"


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
            DataTypes.distance.value: {
                self.agent_id: {}
            },
            DataTypes.triangulation.value: {
                self.agent_id: {}
            },
        }
        self.data_age = dict()

    def clock_tick(self):
        self.age += self.clock_frequency  # allow for the time to be in seconds

    def _get_data_path(self, agent_id, data_type):
        return [data_type, agent_id]

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

    def _set_data(self, agent_id, data_type, data):
        path = self._get_data_path(agent_id, data_type)

        if self.agent_id == 0:
            print("Path:", path)

        if data_type not in self.data:
            self.data[data_type] = dict()
        if data_type not in self.data_age:
            self.data_age[data_type] = dict()

        data_storage = self.data
        data_age = self.data_age

        for element in path[:-1]:
            if element not in data_storage:
                data_storage[element] = dict()
                data_age[element] = dict()
            data_storage = data_storage[element]
            data_age = data_age[element]

        data_storage[path[-1]] = data
        data_age[path[-1]] = self.age

        if self.agent_id == 0:
            print("Data:", data_storage[path[-1]])
            print("Data Age:", data_age[path[-1]])
            print("Data Storage:", self.data)

    def _get_data(self, agent_id, data_type, age=False):
        path = self._get_data_path(agent_id, data_type)

        data = self.data if not age else self.data_age
        for element in path:
            if element in data:
                data = data[element]
            else:
                return None
            return data

    # def _get_all_distances_not_outdated(self):
    #     for agent_id in self.data[DataTypes.distance.value]:
    #         if not self._is_outdated(agent_id, DataTypes.distance.value):
    #             yield agent_id, self._get_data(agent_id, DataTypes.distance.value)

    # def _get_all_triangulation_not_outdated(self):
    #     for agent_id in self.data[DataTypes.triangulation]:
    #         if not self._is_outdated(agent_id, DataTypes.triangulation):
    #             yield agent_id, self._get_data(agent_id, DataTypes.triangulation)

    @abstractmethod
    def get_data_to_send(self):
        """Should send data + their relative age from the POV of the sender, only if not outdated"""
        raise NotImplementedError("DataStorage does not implement a default data preparation")

    def set_distance(self, agent_id, distance):
        path = [DataTypes.distance.value, self.agent_id, agent_id]

        if self.agent_id == 0:
            print("Path:", path)

        if DataTypes.distance.value not in self.data:
            self.data[DataTypes.distance.value] = dict()
        if DataTypes.distance.value not in self.data_age:
            self.data_age[DataTypes.distance.value] = dict()

        data_storage = self.data
        data_age = self.data_age

        for element in path[:-1]:
            if element not in data_storage:
                data_storage[element] = dict()
            if element not in data_age:
                data_age[element] = dict()

            data_storage = data_storage[element]
            data_age = data_age[element]

        data_storage[path[-1]] = distance
        data_age[path[-1]] = self.age

    def set_information(self, agent_id, information):
        """
        :param agent_id: id of the agent that sent the information
        :param information: data to store
        """
        for data_type in information:
            self._set_data(agent_id, data_type, information[data_type])

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)


class FakeDataStorage(DataStorage):
    def __init__(self):
        super().__init__(0, time_to_live=10, clock_frequency=0.01)

    def get_data_to_send(self):
        return dict()
