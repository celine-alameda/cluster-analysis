import concurrent.futures
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from tfce_computation import tfce_from_distribution
import time

from tfce_toolbox import TFCEToolbox, compute_resampling

if __name__ == "__main__":

    n_resamplings = 1000
    n_workers = 8
    # each worker will process n_resamplings // n_workers tasks
    alpha = 0.05
    # init random number generator with seed (for reproducibility)
    seed = 42
    rng = np.random.default_rng(42)
    datapoint_list = []
    bootstrap = False

    inputFiles = []
    analyses = ["4_fa", "4_ua"]
    subjects = list(range(1, 6)) + list(range(7, 16))
    conditions_tdcs = ["real", "sham"]
    conditions_time = ["pre", "post"]
    for analysis in analyses:
        for subject in subjects:
            for condition_tdcs in conditions_tdcs:
                for condition_time in conditions_time:
                    fileName = "time_resolved_local_o_" + analysis + "_" + str(
                        subject) + "_" + condition_tdcs + "_" + condition_time + ".tsv"
                    inputFiles.append(fileName)

    for data_file in inputFiles:
        print("go " + data_file)
        data_frame = pd.read_csv('data/' + data_file, sep="\t")

        tb = TFCEToolbox(rng, data_frame)
        actual_t_list = tb.get_t_values(data_frame)

        # for plotting in R
        t_data = pd.DataFrame(actual_t_list)
        t_data = t_data.rename(columns={0: "t"})
        t_data.to_csv("outputs/t_" + data_file.split('.')[0] + ".tsv", header=True, index=False)

        actual_tfce_list = tfce_from_distribution(actual_t_list)
        tfce_data = pd.DataFrame(actual_tfce_list)
        tfce_data = tfce_data.rename(columns={0: "tfce"})

        if bootstrap:
            # now, resample and check for significance
            max_tfces = []
            min_tfces = []

            all_resamplings = []
            print("Generating {} resamplings".format(n_resamplings))
            for _ in range(n_resamplings):
                if tb.is_condition:
                    all_resamplings.append(tb.shuffle_t_cluster_sign(actual_t_list))
                else:
                    all_resamplings.append(tb.shuffle_t_cluster_position(actual_t_list))
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
        print("done " + data_file)
