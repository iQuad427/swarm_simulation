from src.modules.storage.model import DataStorage, DataTypes


class DistanceOnlyStorage(DataStorage):
    def __init__(self, agent_id, time_to_live=0.1, clock_frequency=10):
        super().__init__(
            agent_id=agent_id,
            time_to_live=time_to_live,
            clock_frequency=clock_frequency,
        )

    def _prepare_data_for_sending(self):
        return {
            DataTypes.distance.value: self.data[DataTypes.distance.value][self.agent_id],
        }

    def get_data_to_send(self):
        return self._prepare_data_for_sending()


class DistanceOnlyTTLStorage(DataStorage):
    # TODO: verify the good working of this class
    def _prepare_data_for_sending(self):
        return {
            # DataTypes.distance.value: dict(self._get_all_distances_not_outdated()),
        }

    def get_data_to_send(self):
        return self._prepare_data_for_sending()


class DistanceAndTriangulationStorage(DataStorage):

    def _prepare_data_for_sending(self):

        # if DataTypes.triangulation.value not in self.data:
        #     self.data[DataTypes.triangulation.value] = dict()
        # if DataTypes.triangulation.value not in self.data_age:
        #     self.data_age[DataTypes.triangulation.value] = dict()

        return {
            DataTypes.triangulation.value: self.data[DataTypes.triangulation.value][self.agent_id],
            DataTypes.distance.value: self.data[DataTypes.distance.value][self.agent_id],
        }

    def get_data_to_send(self):
        return self._prepare_data_for_sending()


class DistanceAndTriangulationTTLStorage(DataStorage):
    # TODO: verify the good working of this class
    def _prepare_data_for_sending(self):
        return {
            # DataTypes.triangulation.value: dict(self._get_all_triangulation_not_outdated()),
            # DataTypes.distance.value: dict(self._get_all_distances_not_outdated()),
        }

    def get_data_to_send(self):
        return self._prepare_data_for_sending()
