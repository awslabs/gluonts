# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from datetime import datetime, timedelta

from tad.anomaly_detect_ts import anomaly_detect_ts


def hourly_datelist(start, length):
    datelist = []
    delta_hr = timedelta(hours=1)
    curr_time = start
    for k in range(length):
        datelist += [curr_time]
        curr_time += delta_hr
    return datelist


n_runs = 50
file_name = "results_auc/TwitterAD/test_simulated_ds1_msig_test.txt"
# Parameters of to generate the data
T = 4000
detection_length = 2000
n_malfunctions = 60
n_obs = 60

auc_scores = np.zeros((n_runs, 2))
np.random.seed(0)

for i in range(n_runs):
    dates = range(T)

    period = 24
    noise = np.random.normal(scale=0.1, size=T)
    mean_ts = np.sin([2 * math.pi * k / period for k in range(T)])  # + noise
    sigma_ts = np.ones(T) + noise

    malfunctions_idx = (
        T
        - 1
        - np.random.choice(
            np.arange(detection_length), n_malfunctions, replace=False
        )
    )
    # sigma_ts[malfunctions_idx] -= .5
    mean_ts[malfunctions_idx] += 1.0

    samples = np.random.normal(mean_ts, sigma_ts, size=(n_obs, T))
    samples = samples.transpose()

    start_date = datetime(2018, 1, 1, 0, 00)
    ts_time = hourly_datelist(start_date, T)
    time_array = np.array(ts_time)

    # AD on empirical mean per hour
    ts = np.mean(samples, axis=1)
    ts_mean = pd.Series(ts, index=ts_time)

    anoms = anomaly_detect_ts(ts_mean, alpha=1.0)
    anoms_times = anoms["anoms"].index

    scores_mean = np.zeros(T)
    for k in range(len(anoms_times)):
        anom_idx = int(np.argwhere(time_array == anoms_times[k]))
        scores_mean[anom_idx] = len(anoms_times) - k

    # AD on empirical std
    ts = np.std(samples, axis=1)
    ts_std = pd.Series(ts, index=ts_time)

    anoms = anomaly_detect_ts(ts_std, alpha=1.0)
    anoms_times = anoms["anoms"].index

    scores_std = np.zeros(T)
    for k in range(len(anoms_times)):
        anom_idx = int(np.argwhere(time_array == anoms_times[k]))
        scores_std[anom_idx] = len(anoms_times) - k

    true_anom = np.zeros(T)
    true_anom[malfunctions_idx] = 1.0
    auc_scores[i, 0] = roc_auc_score(
        true_anom[-detection_length:], scores_mean[-detection_length:]
    )
    auc_scores[i, 1] = roc_auc_score(
        true_anom[-detection_length:], scores_std[-detection_length:]
    )

    print("Simulation {}:".format(i))
    print("     AUC mean = {}".format(auc_scores[i, 0]))
    print("     AUC std = {}".format(auc_scores[i, 1]))

print(auc_scores)

print(np.mean(auc_scores, axis=0))
print(np.std(auc_scores, axis=0))
print(np.percentile(auc_scores, 5, axis=0))
print(np.percentile(auc_scores, 95, axis=0))

np.savetxt(file_name, auc_scores)
