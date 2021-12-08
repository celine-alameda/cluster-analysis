import concurrent.futures
import time

import pandas as pd
import numpy as np

from tfce_toolbox.tfce_computation import tfce_from_distribution, compute_resampling, shuffle_t_cluster_position
import tfce_toolbox.two_by_two_f
import tfce_toolbox.quicker_raw_value


def analyze(data_file, dv, seed):
    print("go " + data_file)
    time_overall = time.time()
    rng = np.random.default_rng(seed)
    data_frame = pd.read_csv('data/' + data_file, sep="\t")
    # analyzer = tfce_toolbox.two_by_two_f.TwoByTwoFMultiProcess(dv=dv, within1="condition_tdcs",
    #                                                           within2="condition_time",
    #                                                           subject="subject", n_workers=8)
    analyzer = tfce_toolbox.quicker_raw_value.QuickerRawValueSingleProcess(dv=dv, datapoint_name="datapoint")
    print("Computing actual list of values")
    t = time.time()
    values = analyzer.compute_values(data_frame)
    print("Done in {} seconds.".format(time.time() - t))
    datapoints_list = data_frame.loc[:, "datapoint"].unique().tolist()
    output_data = {"datapoint": datapoints_list, "value": values}

    print("Computing actual TFCE values")
    t = time.time()
    actual_tfce = tfce_from_distribution(values)
    print("Done in {} seconds.".format(time.time() - t))
    output_data["tfce"] = actual_tfce

    print("Generating {} resamplings for {} datapoints".format(n_resamplings, len(datapoints_list)))
    t = time.time()
    rs_values = analyzer.resample_and_compute_values(values, n_resamplings)

    print("Done in {} seconds.".format(time.time() - t))

    resampled_tfce_tdcs = []
    print("Computing max tfce values for resamplings")
    t = time.time()
    max_tfce = []
    min_tfce = []
    for i in range(len(rs_values)):
        tfce = tfce_from_distribution(rs_values[i])
        max_tfce.append(max(tfce))
        min_tfce.append(min(tfce))
    print("Done in {} seconds.".format(time.time() - t))

    # now, check for significance
    upper_tfce = np.percentile(max_tfce, 100 * (1 - (alpha / 2.0)))
    lower_tfce = np.percentile(min_tfce, 100 * (alpha / 2.0))
    significance = []
    for i in range(len(datapoints_list)):
        if lower_tfce <= actual_tfce[i] <= upper_tfce:
            significance.append(0)
        else:
            significance.append(1)

    output_data["sig"] = significance

    output_data = pd.DataFrame(output_data)
    output_data.to_csv("outputs/tfce_" + data_file.split('.')[0] + ".tsv", header=True, index=False, sep="\t")
    print("done {} in {} seconds.".format(data_file, time.time() - time_overall))


if __name__ == "__main__":
    n_resamplings = 600
    n_workers = 8
    # each worker will process n_resamplings // n_workers tasks
    alpha = 0.05
    seed = 42
    inputFiles = []
    subjects = range(1, 11)
    # analyses = ["3_f"]
    dv = "local_o"
    # this logic only creates the data files to be analyzed. It can be changed at will.
    for subject in subjects:
        fileName = "subject_" + str(subject) + ".tsv"
        inputFiles.append(fileName)
    counter = 0
    for data_file in inputFiles:
        analyze(data_file, dv, seed + counter)
        counter += 1
