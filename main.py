
import pandas as pd
from scipy import stats
from tqdm import tqdm
from tfce_computation import tfce_from_distribution
# values as per doi:10.1016/j.neuroimage.2008.03.061


extend_weight = 0.5
height_weight = 2
dh = 0.1

local_o_name = "Local.o"
condition_name = "sound_state"
condition_value_pre = 0
condition_value_post = 1
trial_name = "trial"
datapoint_name = "datapoint"

data_frame = pd.read_csv('data/data.tsv', sep="\t")

datapoint_list = data_frame.loc[data_frame[trial_name] == 1, datapoint_name].unique().tolist()

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

# for plotting in R
t_data = pd.DataFrame(t_values)
t_data = t_data.rename(columns={0: "t"})
t_data.to_csv("outputs/t_values.tsv", header=True, index=False)

tfce_values = tfce_from_distribution(t_values)
tfce_data = pd.DataFrame(tfce_values)
tfce_data = tfce_data.rename(columns={0: "tfce"})
tfce_data.to_csv("outputs/tfce_values.tsv", header=True, index=False)
