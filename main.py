import concurrent.futures
import time

import pandas as pd
import numpy as np

from tfce_toolbox.tfce_computation import tfce_from_distribution, compute_resampling, shuffle_t_cluster_position
from tfce_toolbox.two_by_two_f import TwoByTwoFSingleProcess


def analyze(data_file, dv, seed):
    print("go " + data_file)
    t_overall = time.time()
    rng = np.random.default_rng(seed)
    data_frame = pd.read_csv('data/' + data_file, sep="\t")
    analyzer = TwoByTwoFSingleProcess(dv=dv, within1="condition_tdcs", within2="condition_time",
                                      subject="subject")
    print("Computing actual list of F values")
    t = time.time()
    datapoint_list = data_frame.loc[:, "datapoint"].unique().tolist()
    f_tdcs_values, f_time_values, f_inter_values = analyzer.compute_values(data_frame, datapoint_name="datapoint",
                                                                           datapoints_list=datapoint_list)
    print("Done in {} seconds.".format(time.time() - t))

    f_data = {"datapoint": datapoint_list, "F_tdcs": f_tdcs_values, "F_time": f_time_values, "F_inter": f_inter_values}

    print("Computing actual TFCE values")
    t = time.time()
    tfce_tdcs = tfce_from_distribution(f_tdcs_values)
    tfce_time = tfce_from_distribution(f_time_values)
    tfce_inter = tfce_from_distribution(f_inter_values)
    print("Done in {} seconds.".format(time.time() - t))
    f_data["TFCE_tdcs"] = tfce_tdcs
    f_data["TFCE_time"] = tfce_time
    f_data["TFCE_inter"] = tfce_inter

    print("Generating {} resamplings for {} datapoints".format(n_resamplings, len(datapoint_list)))
    t = time.time()
    resampled_data_frames = []
    for _ in range(n_resamplings):
        # todo make more generic
        real_pre = data_frame[data_frame["condition"] == "real_pre"]
        real_pre = real_pre[dv].to_list()
        real_post = data_frame[data_frame["condition"] == "real_post"]
        real_post = real_post[dv].to_list()
        sham_pre = data_frame[data_frame["condition"] == "sham_pre"]
        sham_pre = sham_pre[dv].to_list()
        sham_post = data_frame[data_frame["condition"] == "sham_post"]
        sham_post = sham_post[dv].to_list()
        resampled_df = data_frame.copy()
        resampled_df["new_value"] = 0.0
        shuffled_value_indexes = rng.integers(low=0, high=4, size=len(data_frame))
        for index in range(len(data_frame)):
            subject_index = index // (4 * len(datapoint_list))
            index_wo_subject = index % (4 * len(datapoint_list))
            # condition_index = index_wo_subject // len(datapoint_list)
            datapoint_index = index_wo_subject % len(datapoint_list)
            new_condition_index = shuffled_value_indexes[index]
            if new_condition_index == 0:
                new_value = real_pre[subject_index * len(datapoint_list) + datapoint_index]
            elif new_condition_index == 1:
                new_value = real_post[subject_index * len(datapoint_list) + datapoint_index]
            elif new_condition_index == 2:
                new_value = sham_pre[subject_index * len(datapoint_list) + datapoint_index]
            else:
                new_value = sham_post[subject_index * len(datapoint_list) + datapoint_index]
            resampled_df.at[index, "new_value"] = new_value
        resampled_df = resampled_df.drop(dv, axis=1)
        resampled_df = resampled_df.rename(columns={"new_value": dv})
        resampled_data_frames.append(resampled_df)


    # takes 4 secs only for 1 resample >_>
    # resampled_df = data_frame.copy()
    # shuffled_value_indexes = rng.integers(low=0, high=4, size=len(data_frame))
    # resampled_df["shuffled_indexes"] = shuffled_value_indexes
    # resampled_df["new_value"] = 0.0
    # subject_list = data_frame.loc[:, "subject"].unique().tolist()
    # for datapoint in datapoint_list:
    #     for subject in subject_list:
    #         resampled_df_by_dp = resampled_df[
    #             (resampled_df["datapoint"] == datapoint) & (resampled_df["subject"] == subject)]
    #         indexes = resampled_df_by_dp.index.to_list()  # indexes that concern this subject and dp
    #         for index in indexes:
    #             shuffled_value_index = resampled_df.loc[index, "shuffled_indexes"]
    #             shuffled_value = resampled_df_by_dp.loc[indexes[shuffled_value_index], dv]
    #             resampled_df.at[index, "new_value"] = shuffled_value
    # print(resampled_df)
    # resampled_df = resampled_df.drop(dv, axis=1)
    # resampled_df = resampled_df.rename(columns={"new_value": dv})
    print("Done in {} seconds.".format(time.time() - t))

    print("Computing F values for each resampling")
    f_data = pd.DataFrame(f_data)
    f_data.to_csv("outputs/f_" + data_file.split('.')[0] + ".tsv", header=True, index=False, sep="\t")
    exit(1)

    tfce_data = pd.DataFrame(actual_tfce_list)
    tfce_data = tfce_data.rename(columns={0: "tfce"})

    # now, resample and check for significance
    max_tfces = []
    min_tfces = []

    all_resamplings = []
    print("CHANGE RESAMPLING CODE")
    exit(1)
    print("Generating {} resamplings".format(n_resamplings))
    for _ in range(n_resamplings):
        all_resamplings.append(shuffle_t_cluster_position(rng, actual_unclustered_list))
    print("Processing resamplings with {} processes".format(n_workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_resampling,
                                   all_resamplings,
                                   i * n_resamplings // n_workers,
                                   n_resamplings // n_workers): i for i
                   in range(n_workers)}
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

    f_data = pd.DataFrame(f_data)
    f_data.to_csv("outputs/f_" + data_file.split('.')[0] + ".tsv", header=True, index=False, sep="\t")
    print("done {} in {} seconds.".format(data_file, time.time() - t_overall))


if __name__ == "__main__":
    n_resamplings = 100
    n_workers = 8
    # each worker will process n_resamplings // n_workers tasks
    alpha = 0.05
    seed = 42
    inputFiles = []
    # analyses = ["3_f", "3_u", "4_fa", "4_ua"]
    analyses = ["3_f"]
    dv = "mean_local_o"
    # this logic only creates the data files to be analyzed. It can be changed at will.
    for analysis in analyses:
        fileName = analysis + ".tsv"
        inputFiles.append(fileName)

    counter = 0
    for data_file in inputFiles:
        analyze(data_file, dv, seed + counter)
        counter += 1
