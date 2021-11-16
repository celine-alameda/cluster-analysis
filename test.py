""" test for F statistics computation"""

import statsmodels.api as sm
import pingouin as pg
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import time

pre_real = [0.7, 2.3, 4.2, 5.8, 6.9, 2.4, 0.8, 3.4]
post_real = [3.9, 5.2, 4.6, 7.5, 1.2, 6.1, 5.5, 5.8]
pre_sham = [0.9, 3.2, 8.4, 0.5, 4.2, 7.1, 1.8, 2.3]
post_sham = [0.9, 2.8, 2.4, 1.5, 0.2, 1.1, 2.8, 4.9]

values = pre_real + post_real + pre_sham + post_sham


data = {"measure" : values}
data = pd.DataFrame(data)
data = data.assign(condition_tdcs=np.repeat(['real', 'sham'], 16))
data = data.assign(condition_time=np.tile(np.repeat(["pre", "post"], [8, 8]), 2))
data = data.assign(subject=np.tile(list(range(1,9)), 4))

print(data)
#perform two-way ANOVA
model = ols('measure ~ C(condition_tdcs) + C(condition_time) + C(condition_tdcs):C(condition_time)', data=data).fit()
print(sm.stats.anova_lm(model, typ=2))
t = time.time()
aov = pg.rm_anova(data=data, dv="measure", within=["condition_tdcs", "condition_time"], subject="subject")
print("elapsed {}".format(time.time() - t))

print(aov["F"])

print(aov["F"][1])

