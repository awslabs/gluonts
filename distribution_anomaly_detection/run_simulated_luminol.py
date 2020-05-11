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
import time
import numpy as np
from sklearn.metrics import roc_auc_score

from luminol.anomaly_detector import AnomalyDetector

n_runs = 10
file_name = "results_auc/luminol/test_simulated_ds1_msig_test.txt"
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
    mean_ts = np.sin([2 * math.pi * k / period for k in range(T)])
    sigma_ts = np.ones(T) + noise / 2

    malfunctions_idx = (
        T
        - 1
        - np.random.choice(
            np.arange(detection_length), n_malfunctions, replace=False
        )
    )
    sigma_ts[malfunctions_idx] -= 0.5
    # mean_ts[malfunctions_idx] += 1.

    samples = np.random.normal(mean_ts, sigma_ts, size=(n_obs, T))
    samples = samples.transpose()

    # AD on empirical mean per hour
    ts = np.mean(samples, axis=1)
    ts_mean = {k: ts[k] for k in range(T)}

    my_detector = AnomalyDetector(
        ts_mean, algorithm_name="derivative_detector"
    )
    score = my_detector.get_all_scores()
    scores_mean = np.array([s for (k, s) in score.iteritems()])

    # AD on empirical std
    ts = np.std(samples, axis=1)
    ts_std = {k: ts[k] for k in range(T)}

    my_detector = AnomalyDetector(ts_std, algorithm_name="derivative_detector")
    score = my_detector.get_all_scores()
    scores_std = np.array([s for (k, s) in score.iteritems()])

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
