import random
from abc import abstractmethod
from tfce_toolbox.cluster_value_calculator import ClusterValueCalculator


class RawValue(ClusterValueCalculator):

    def __init__(self, dv, datapoint_name):
        self.dv = dv
        self.datapoint_name = datapoint_name

    def compute_value(self, data_frame):
        value = data_frame.loc[0, self.dv]
        return value

    @abstractmethod
    def compute_values(self, data_frame):
        pass


class RawValueSingleProcess(RawValue):

    def compute_values(self, data_frame):
        datapoints_list = data_frame.loc[:, self.datapoint_name].unique().tolist()
        values = []
        for datapoint in datapoints_list:
            partial_data_frame = data_frame[data_frame[self.datapoint_name] == datapoint]
            partial_data_frame = partial_data_frame.reset_index()
            value = self.compute_value(partial_data_frame)
            values.append(value)
        return values

    def resample_values(self, data_frame):
        datapoints_list = data_frame.loc[:, self.datapoint_name].unique().tolist()
        sample = random.choices(datapoints_list, k=len(datapoints_list))
        data_frame = data_frame.reindex(sample)
        data_frame[self.datapoint_name] = datapoints_list
        data_frame = data_frame.reset_index(drop=True)
        return data_frame
