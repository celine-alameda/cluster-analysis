from abc import abstractmethod


class ClusterValueCalculator:

    @abstractmethod
    def compute_values(self, data_frame):
        pass

    @abstractmethod
    def compute_value(self, data_frame):
        pass
