import concurrent
import random
from abc import abstractmethod
from tfce_toolbox.cluster_value_calculator import ClusterValueCalculator


class QuickerRawValue():

    def __init__(self, dv, datapoint_name):
        self.dv = dv
        self.datapoint_name = datapoint_name

    def compute_value(self, data_frame):
        value = data_frame.loc[0, self.dv]
        return value

    @abstractmethod
    def compute_values(self, data_frame):
        pass


class QuickerRawValueSingleProcess(QuickerRawValue):

    def compute_values(self, data_frame):
        values = data_frame[self.dv].to_list()
        return values

    def resample_values(self, values):
        resampled = random.choices(values, k=len(values))
        return resampled

    def resample_and_compute_values(self, values, n_resamplings):
        resampled_values = []
        for _ in range(n_resamplings):
            resampled_values.append(self.resample_values(values))
        return resampled_values
