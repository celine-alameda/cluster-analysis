import numpy as np
import pandas as pd
from scipy import stats

from tfce_computation import tfce_from_distribution


class TFCEToolbox:

    def __init__(self, rng, data_frame):
        self.rng = rng
        self.local_o_name = "local_o"
        self.is_condition = False
        self.condition_name = "condition"
        self.condition_value_pre = 0
        self.condition_value_post = 1
        self.trial_name = "trial"
        self.datapoint_name = "time"
        self.datapoint_list = []
        tmp_trial_no = 1
        while len(self.datapoint_list) == 0:
            self.datapoint_list = data_frame.loc[
                data_frame[self.trial_name] == tmp_trial_no, self.datapoint_name].unique().tolist()
            tmp_trial_no += 1
            if tmp_trial_no > 200:
                print("could not find any trial to extract datapoint from")
                exit(1)

    def get_t_values(self, data_frame: pd.DataFrame):
        if self.is_condition:
            actual_t_list = self.t_values_from_dataframe_two_samples(data_frame)
        else:
            values = data_frame[self.local_o_name]
            theoretical_mean = np.mean(values)
            actual_t_list = self.t_values_from_dataframe_one_sample(data_frame, theoretical_mean)
        return actual_t_list

    def shuffle_t_cluster_sign(self, t_values: list):
        """shuffle the sign of the t values. Useful when the t-values are computed between two conditions.
        In this case it is equivalent to shuffling the two conditions"""
        ints = self.rng.integers(low=0, high=2, size=len(t_values)) * 2 - 1
        return t_values * ints

    def shuffle_t_cluster_position(self, t_values: list):
        """shuffle the position of the t values, with replacement (as should be for bootstrapping)"""
        ints = self.rng.integers(low=0, high=len(t_values), size=len(t_values))
        return [t_values[index] for index in ints]

    def t_values_from_dataframe_one_sample(self, df1: pd.DataFrame, mu: float):
        t_values = []
        # compute t value for each datapoint to establish clusters
        for datapoint in self.datapoint_list:
            values4t = df1.loc[
                (df1[self.datapoint_name] == datapoint), self.local_o_name].to_list()
            t, p = stats.ttest_1samp(values4t, mu)
            t_values.append(t)
        return t_values

    def t_values_from_dataframe_two_samples(self, df1: pd.DataFrame):
        datapoint_list = df1.loc[df1[self.trial_name] == 1, self.datapoint_name].unique().tolist()
        t_values = []
        # compute t value for each datapoint to establish clusters
        for datapoint in datapoint_list:
            values_pre = df1.loc[
                (df1[self.condition_name] == self.condition_value_pre) & (
                        df1[self.datapoint_name] == datapoint), self.local_o_name].to_list()
            values_post = df1.loc[
                (df1[self.condition_name] == self.condition_value_post) & (
                        df1[self.datapoint_name] == datapoint), self.local_o_name].to_list()
            t, p = stats.ttest_rel(values_post, values_pre)
            t_values.append(t)
        return t_values


def compute_resampling(t_resamplings, start, number_to_do):
    min_values = []
    max_values = []
    for i in range(start, start + number_to_do):
        resampling = t_resamplings[i]
        tfce_list = tfce_from_distribution(resampling)
        min_values.append(min(tfce_list))
        max_values.append(max(tfce_list))
    return min_values, max_values
