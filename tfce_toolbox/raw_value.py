from abc import abstractmethod
from tfce_toolbox.cluster_value_calculator import ClusterValueCalculator


class RawValue(ClusterValueCalculator):

    def __init__(self, dv):
        self.dv = dv

    def compute_value(self, data_frame):
        value = data_frame.iloc[1, self.dv]
        return value

    @abstractmethod
    def compute_values(self, data_frame, datapoints_list, datapoint_name):
        pass


class RawValueSingleProcess(RawValue):

    def compute_values(self, data_frame, datapoints_list, datapoint_name):
        values = []
        for datapoint in datapoints_list:
            partial_data_frame = data_frame[data_frame[datapoint_name] == datapoint]
            value = self.compute_value(partial_data_frame)
            values.append(value)
        return values
