import pandas as pd
from scipy import stats
from tqdm import tqdm

local_o_name = "Local.o"
condition_name = "sound_state"
condition_value_pre = 0
condition_value_post = 1
trial_name = "trial"
datapoint_name = "datapoint"

data_frame = pd.read_csv('data/data.tsv', sep="\t")
datapoint_list = data_frame.loc[data_frame[trial_name] == 1, datapoint_name].to_list()

t_values = []

# compute t value for each datapoint to establish clusters
for datapoint in tqdm(datapoint_list):
    values_pre = data_frame.loc[
        (data_frame[condition_name] == condition_value_pre) & (
                data_frame[datapoint_name] == datapoint), local_o_name].to_list()
    values_post = data_frame.loc[
        (data_frame[condition_name] == condition_value_post) & (
                data_frame[datapoint_name] == datapoint), local_o_name].to_list()
    t, p = stats.ttest_rel(values_post, values_pre)
    t_values.append(t)

# TFCE enhancement
tfce_values = []

# for plotting in R
t_data = pd.DataFrame(t_values)
t_data = t_data.rename(columns={0: "t"})
t_data.to_csv("outputs/out.tsv", header=True, index=False)
