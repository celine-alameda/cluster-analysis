import concurrent.futures
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from tfce_computation import tfce_from_distribution
import time


def shuffle_t_cluster_sign(t_values: list):
    """shuffle the sign of the t values. Useful when the t-values are computed between two conditions.
    In this case it is equivalent to shuffling the two conditions"""
    ints = rng.integers(low=0, high=2, size=len(t_values)) * 2 - 1
    return t_values * ints


def shuffle_t_cluster_position(t_values: list):
    """shuffle the position of the t values, with replacement (as should be for bootstrapping)"""
    ints = rng.integers(low=0, high=len(t_values), size=len(t_values))
    return [t_values[index] for index in ints]


def t_values_from_dataframe_one_sample(df1: pd.DataFrame):
    datapoint_list = df1.loc[data_frame[trial_name] == 1, datapoint_name].unique().tolist()
    t_values = []
    # compute t value for each datapoint to establish clusters
    for datapoint in datapoint_list:
        values = df1.loc[
            (df1[datapoint_name] == datapoint), local_o_name].to_list()
        t, p = stats.ttest_1samp(values, 0.0)
        t_values.append(t)
    return t_values


def t_values_from_dataframe_two_samples(df1: pd.DataFrame):
    datapoint_list = df1.loc[data_frame[trial_name] == 1, datapoint_name].unique().tolist()
    t_values = []
    # compute t value for each datapoint to establish clusters
    for datapoint in datapoint_list:
        values_pre = df1.loc[
            (df1[condition_name] == condition_value_pre) & (
                    df1[datapoint_name] == datapoint), local_o_name].to_list()
        values_post = df1.loc[
            (df1[condition_name] == condition_value_post) & (
                    df1[datapoint_name] == datapoint), local_o_name].to_list()
        t, p = stats.ttest_rel(values_post, values_pre)
        t_values.append(t)
    return t_values


def compute_resampling(t_resamplings):
    min_values = []
    max_values = []
    for resampling in t_resamplings:
        tfce_list = tfce_from_distribution(resampling)
        min_values.append(min(tfce_list))
        max_values.append(max(tfce_list))
    return min_values, max_values


if __name__ == "__main__":

    n_resamplings = 1000
    n_workers = 8
    # each worker will process n_resamplings // n_workers tasks
    alpha = 0.05
    # init random number generator with seed (for reproducibility)
    seed = 42
    data_file = "time_resolved_local_o_3_f_tfe.tsv"

    rng = np.random.default_rng(seed)
    local_o_name = "local_o"
    is_condition = False
    condition_name = "condition"
    condition_value_pre = 0
    condition_value_post = 1
    trial_name = "trial"
    datapoint_name = "time"

    data_frame = pd.read_csv('data/' + data_file, sep="\t")

    # for plotting in R
    if is_condition:
        actual_t_list = t_values_from_dataframe_two_samples(data_frame)
    else:
        actual_t_list = t_values_from_dataframe_one_sample(data_frame)
    t_data = pd.DataFrame(actual_t_list)
    t_data = t_data.rename(columns={0: "t"})
    t_data.to_csv("outputs/t_" + data_file.split('.')[0] + ".tsv", header=True, index=False)

    actual_tfce_list = tfce_from_distribution(actual_t_list)
    tfce_data = pd.DataFrame(actual_tfce_list)
    tfce_data = tfce_data.rename(columns={0: "tfce"})

    # now, resample and check for significance
    max_tfces = []
    min_tfces = []

    all_resamplings = []
    print("Generating {} resamplings".format(n_resamplings))
    for _ in range(n_workers):
        resamplings_for_worker = []
        for _ in range(n_resamplings // n_workers):
            if is_condition:
                resamplings_for_worker.append(shuffle_t_cluster_sign(actual_t_list))
            else:
                resamplings_for_worker.append(shuffle_t_cluster_position(actual_t_list))
        all_resamplings.append(resamplings_for_worker)
    print("Processing resamplings with {} processes".format(n_workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_resampling, resamplings): resamplings for resamplings
                   in all_resamplings}
        for future in concurrent.futures.as_completed(futures):
            min_tfce, max_tfce = future.result()
            min_tfces.append(min_tfce)
            max_tfces.append(max_tfce)

    lower = np.percentile(min_tfces, 100 * alpha / 2.0)
    upper = np.percentile(max_tfces, 100 * (1 - (alpha / 2.0)))
    significance = []
    for i in range(len(actual_tfce_list)):
        actual_tfce = actual_tfce_list[i]
        if lower <= actual_tfce <= upper:
            significance.append(0)
        else:
            significance.append(1)

    tfce_data["sig"] = significance
    tfce_data.to_csv("outputs/tfce_" + data_file.split('.')[0] + ".tsv", header=True, index=False, sep='\t')
