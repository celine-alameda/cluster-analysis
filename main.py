import math
import pandas as pd
from scipy import stats
from tqdm import tqdm

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

### TODO find a way to create unique names, here 1:600 repeated twice
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

for i in tqdm(range(len(t_values))):
    # floor to 0.1
    # notations are similar to those in the paper
    if t_values[i] == 0:
        tfce_values.append(0)
        continue
    signum = t_values[i] / t_values[i]
    h_p = math.floor(abs(t_values[i]) / dh) * dh
    height = dh
    tfce = 0
    while height <= h_p:
        # extent is how many samples have values of at least h
        # reach forward
        extend = 1  # at least this sample
        index = i + 1
        while index < len(t_values):
            if abs(t_values[index] < height):
                break
            extend += 1
            index += 1
        # reach backward
        index = i - 1
        while index > 0:
            if abs(t_values[index] < height):
                break
            extend += 1
            index += 1
        tfce = tfce + math.pow(extend, extend_weight) * math.pow(height, height_weight)
        height += dh
    tfce = tfce * signum
    tfce_values.append(tfce)

# for plotting in R
t_data = pd.DataFrame(t_values)
t_data = t_data.rename(columns={0: "t"})
# t_data.to_csv("outputs/out.tsv", header=True, index=False)

tfce_data = pd.DataFrame(tfce_values)
tfce_data = tfce_data.rename(columns={0: "tfce"})
tfce_data.to_csv("outputs/tfce.tsv", header=True, index=False)
