import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from tfce_computation import tfce_from_distribution
import time

n_resamplings = 100
alpha = 0.05
# init random number generator with seed (for reproducibility)
rng = np.random.default_rng(42)


def resample_data_frame(df2: pd.DataFrame):
    n_samples = df2.shape[0]
    random_ints = rng.integers(low=0, high=n_samples, size=n_samples)
    resampled_df = df2.iloc[random_ints]
    resampled_df[trial_name] = df2[trial_name].values
    resampled_df[datapoint_name] = df2[datapoint_name].values
    resampled_df[condition_name] = df2[condition_name].values
    resampled_df.index = range(n_samples)
    return resampled_df


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
resampled_tfces = []
for _ in tqdm(range(n_resamplings)):
    df = resample_data_frame(data_frame)
    t_list = t_values_from_dataframe(df)
    tfce_list = tfce_from_distribution(t_list)
    resampled_tfces.append(tfce_list)

resampled_tfces = np.array(resampled_tfces)
significance = []
for i in tqdm(range(resampled_tfces.shape[1])):
    one_datapoint = np.squeeze(resampled_tfces[:, i])
    lower = np.percentile(one_datapoint, 100 * alpha / 2.0)
    upper = np.percentile(one_datapoint, (1 - (alpha / 2.0)) * 100)
    actual_tfce = actual_tfce_list[i]
    if lower <= actual_tfce <= upper:
        significance.append(0)
    else:
        significance.append(1)

tfce_data["sig"] = significance
tfce_data.to_csv("outputs/tfce_values.tsv", header=True, index=False, sep='\t')
