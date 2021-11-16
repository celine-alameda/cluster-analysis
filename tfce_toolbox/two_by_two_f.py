import pingouin as pg


class TwoByTwoF:

    def __init__(self, dv, within1, within2, subject):
        self.dv = dv
        self.within1 = within1
        self.within2 = within2
        self.subject = subject

    def compute_value(self, data_frame):
        aov = pg.rm_anova(data=data_frame, dv=self.dv, within=[self.within1, self.within2], subject=self.subject)
        return aov["F"].to_list()


class TwoByTwoFSingleProcess(TwoByTwoF):

    def compute_values(self, data_frame, datapoints_list, datapoint_name):
        f_within1_values = []
        f_within2_values = []
        f_inter_values = []
        for datapoint in datapoints_list:
            partial_data_frame = data_frame[data_frame[datapoint_name] == datapoint]
            [f_within1, f_within2, f_inter] = self.compute_value(partial_data_frame)
            f_within1_values.append(f_within1)
            f_within2_values.append(f_within2)
            f_inter_values.append(f_inter)
        return f_within1_values, f_within2_values, f_inter_values
