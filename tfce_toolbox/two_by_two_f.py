import concurrent
from abc import abstractmethod

import pingouin as pg

from tfce_toolbox.cluster_value_calculator import ClusterValueCalculator


class TwoByTwoF(ClusterValueCalculator):

    def __init__(self, dv, within1, within2, subject, datapoint_name):
        self.dv = dv
        self.within1 = within1
        self.within2 = within2
        self.subject = subject
        self.datapoint_name = datapoint_name

    def compute_value(self, data_frame):
        aov = pg.rm_anova(data=data_frame, dv=self.dv, within=[self.within1, self.within2], subject=self.subject)
        return aov["F"].to_list()

    @abstractmethod
    def compute_values(self, data_frame):
        pass


class TwoByTwoFSingleProcess(TwoByTwoF):

    def compute_values(self, data_frame):
        datapoints_list = data_frame.loc[:, self.datapoint_name].unique().tolist()
        f_within1_values = []
        f_within2_values = []
        f_inter_values = []
        for datapoint in datapoints_list:
            partial_data_frame = data_frame[data_frame[self.datapoint_name] == datapoint]
            [f_within1, f_within2, f_inter] = self.compute_value(partial_data_frame)
            f_within1_values.append(f_within1)
            f_within2_values.append(f_within2)
            f_inter_values.append(f_inter)
        return f_within1_values, f_within2_values, f_inter_values


class TwoByTwoFMultiProcess(TwoByTwoF):

    def __init__(self, dv, within1, within2, subject, n_workers):
        super().__init__(dv, within1, within2, subject)
        self.n_workers = n_workers

    def compute_values(self, data_frame):
        # spread datapoints
        datapoints_list = data_frame.loc[:, self.datapoint_name].unique().tolist()
        works = []
        datapoints_in_work = len(datapoints_list) // self.n_workers
        n_extra_datapoints = len(datapoints_list) % self.n_workers
        for i in range(self.n_workers):
            begin = i * datapoints_in_work
            end = (i + 1) * datapoints_in_work
            works.append(datapoints_list[begin:end])
        # extra
        if n_extra_datapoints != 0:
            extra_work = datapoints_list[datapoints_in_work * self.n_workers:len(datapoints_list)]
            works.append(extra_work)

        data_dict = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_url = {executor.submit(self.compute_sub_df, data_frame, work, self.datapoint_name): work for work in
                             works}
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    data = future.result()
                    data_dict.update(data)
                    # return dictionary {datapoint : [F_tdcs, F_time, F_inter]}
                except Exception as exc:
                    print('Exception: {}'.format(exc))
        f_within1_values = []
        f_within2_values = []
        f_inter_values = []
        for i in datapoints_list:
            f_within1 = data_dict[i][0]
            f_within2 = data_dict[i][1]
            f_inter = data_dict[i][2]
            f_within1_values.append(f_within1)
            f_within2_values.append(f_within2)
            f_inter_values.append(f_inter)
        return f_within1_values, f_within2_values, f_inter_values

    def compute_sub_df(self, data_frame, datapoints_list, datapoint_name):
        # return dictionary {datapoint : [F_tdcs, F_time, F_inter]}
        return_dict = {}
        for datapoint in datapoints_list:
            partial_data_frame = data_frame[data_frame[datapoint_name] == datapoint]
            [f_within1, f_within2, f_inter] = self.compute_value(partial_data_frame)
            return_dict[datapoint] = (f_within1, f_within2, f_inter)
        return return_dict

    #     code for resampling df
    # for _ in range(n_resamplings):
    #     real_pre = data_frame[data_frame["condition"] == "real_pre"]
    #     real_pre = real_pre[dv].to_list()
    #     real_post = data_frame[data_frame["condition"] == "real_post"]
    #     real_post = real_post[dv].to_list()
    #     sham_pre = data_frame[data_frame["condition"] == "sham_pre"]
    #     sham_pre = sham_pre[dv].to_list()
    #     sham_post = data_frame[data_frame["condition"] == "sham_post"]
    #     sham_post = sham_post[dv].to_list()
    #     resampled_df = data_frame.copy()
    #     resampled_df["new_value"] = 0.0
    #     shuffled_value_indexes = rng.integers(low=0, high=4, size=len(data_frame))
    #     for index in range(len(data_frame)):
    #         subject_index = index // (4 * len(datapoint_list))
    #         index_wo_subject = index % (4 * len(datapoint_list))
    #         # condition_index = index_wo_subject // len(datapoint_list)
    #         datapoint_index = index_wo_subject % len(datapoint_list)
    #         new_condition_index = shuffled_value_indexes[index]
    #         if new_condition_index == 0:
    #             new_value = real_pre[subject_index * len(datapoint_list) + datapoint_index]
    #         elif new_condition_index == 1:
    #             new_value = real_post[subject_index * len(datapoint_list) + datapoint_index]
    #         elif new_condition_index == 2:
    #             new_value = sham_pre[subject_index * len(datapoint_list) + datapoint_index]
    #         else:
    #             new_value = sham_post[subject_index * len(datapoint_list) + datapoint_index]
    #         resampled_df.at[index, "new_value"] = new_value
    #     resampled_df = resampled_df.drop(dv, axis=1)
    #     resampled_df = resampled_df.rename(columns={"new_value": dv})
    #     resampled_data_frames.append(resampled_df)
