import concurrent
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
        sample = random.choices(range(len(data_frame)), k=len(data_frame))
        data_frame = data_frame.reindex(sample)
        data_frame[self.datapoint_name] = datapoints_list
        data_frame = data_frame.reset_index()
        return data_frame

    def resample_and_compute_values(self, data_frame, n_resamplings):
        resampled_data_frames = []
        for _ in range(n_resamplings):
            resampled_data_frames.append(self.resample_values(data_frame))
        print("Computing values for each resampling")
        rs_values = []
        for resampled_df in resampled_data_frames:
            rs_val = self.compute_values(resampled_df)
            rs_values.append(rs_val)
        return rs_values


class RawValueMultiProcess(RawValueSingleProcess):

    def __init__(self, dv, datapoint_name, n_workers):
        super().__init__(dv, datapoint_name)
        self.n_workers = n_workers

    def resample_and_compute_values(self, data_frame, n_resamplings):
        resampled_data_frames = []
        for _ in range(n_resamplings):
            resampled_data_frames.append(self.resample_values(data_frame))
        print("Computing values for each resampling")
        rs_values = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_url = {executor.submit(self.compute_values, df): df for df in
                             resampled_data_frames}
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    data = future.result()
                    rs_values.append(data)
                except Exception as exc:
                    print('Exception: {}'.format(exc))
        return rs_values
