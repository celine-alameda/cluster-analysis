import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from tfce_computation import tfce_from_distribution
import time

n_resamplings = 1000
alpha = 0.05
# init random number generator with seed (for reproducibility)
rng = np.random.default_rng(42)


def shuffle_t_cluster(t_values: list):
    ints = rng.integers(low=0, high=2, size=len(t_values)) * 2 - 1
    return t_values * ints


def t_values_from_dataframe(df1: pd.DataFrame):
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


local_o_name = "Local.o"
condition_name = "sound_state"
condition_value_pre = 0
condition_value_post = 1
trial_name = "trial"
datapoint_name = "datapoint"

data_frame = pd.read_csv('data/data.tsv', sep="\t")

# for plotting in R
actual_t_list = t_values_from_dataframe(data_frame)
t_data = pd.DataFrame(actual_t_list)
t_data = t_data.rename(columns={0: "t"})
t_data.to_csv("outputs/t_values.tsv", header=True, index=False)

actual_tfce_list = tfce_from_distribution(actual_t_list)
tfce_data = pd.DataFrame(actual_tfce_list)
tfce_data = tfce_data.rename(columns={0: "tfce"})

# now, resample and check for significance
max_tfces = []
min_tfces = []
for _ in tqdm(range(n_resamplings)):
    shuffled_ts = shuffle_t_cluster(actual_t_list)
    tfce_list = tfce_from_distribution(shuffled_ts)
    max_tfces.append(max(tfce_list))
    min_tfces.append(min(tfce_list))

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
tfce_data.to_csv("outputs/tfce_values.tsv", header=True, index=False, sep='\t')
