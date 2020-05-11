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

import os
import sys

import time
import numpy as np
import pandas as pd
import mxnet as mx
import json

from gluonts.dataset.common import ListDataset
from mydeepar import MyDeepAREstimator
from gluonts.trainer import Trainer
from gluonts.distribution.dirichlet_multinomial import (
    DirichletMultinomialOutput,
    DirichletMultinomial,
)

import matplotlib.pyplot as plt

from stats_tools import ecdf, sample_dir_mult
from utils import get_ds_list

from sklearn.metrics import f1_score, roc_auc_score, auc

# -------------------------------
# Parameters of the run
# -------------------------------
np.random.seed(0)
mx.random.seed(0)

# Sub benchmark, this can be one of 4 available be an amazon benchmarks:
#     B1Benchmark: 1 min time granularity
#                  55 time series
#     B2Benchmark: 5 min time granularity
#                  100 time series
#     B3Benchmark: 5 min time granularity
#                  116 time series
benchmark = "B1Benchmark"

assert benchmark in ("B1Benchmark", "B2Benchmark", "B3Benchmark"), print(
    "Benchmark unknown"
)

# Data parameters
start = time.time()

val_name = "target"
anom_name = "anomaly_indicator"

# Proportion of the data to use as test (detection range)
detection_prop = 0.4

# Aggregation time, set to 30 min for both original
# time granularity
if benchmark == "B1Benchmark":
    n_obs = 30
    freq = "30T"
    # Past observations to use for prediction
    lags_seq = [
        1,
        2,
        3,
        4,
        5,
        6,  # Last 3 hours
        46,
        47,
        48,
        49,
        50,  # 1 day ago
        94,
        95,
        96,
        97,
        98,  # 2 days ago
        143,
        144,
        145,  # 3 days ago
    ]
else:
    n_obs = 5
    freq = "30T"
    # Past observations to use for prediction
    lags_seq = [
        1,
        2,
        3,
        4,
        5,
        6,  # Last 3 hours
        46,
        47,
        48,
        49,
        50,  # 1 day ago
        94,
        95,
        96,
        97,
        98,  # 2 days ago
        143,
        144,
        145,  # 3 days ago
        335,
        336,
        337,  # 1 week ago
    ]

# Dimension of the binned distributions
dim = 100
# Type of grid on which to evaluate the ecdf
linear_grid = True

# Restrain the domain to (cap_low_per percentile, cap_hig_per percentile)
cap_low_per = 0.5
cap_high_per = 99.5
boundaries_scale = 2.0

# Model
model = "deepar"

# Trainer parameters
epochs = 200
learning_rate = 1e-2
batch_size = 8
num_batches_per_epoch = 50
prediction_length = 1
context_length = 5

# Prediction parameter
num_samples = 100

# -------------------------------
# Load data
# -------------------------------
data_list = []
n_nans = []
print("Loading data")

if benchmark == "B1Benchmark":
    file_name = "internal_1min/data.json"
else:
    file_name = "internal_5min/data.json"

with open(file_name) as json_file:
    for line in json_file:
        ds = json.loads(line)
        # Keep only time series with labeled anomalies and that are defined at every time point
        if (
            np.sum(np.isnan(np.array(ds[val_name], float))) == 0
            and len(ds[val_name]) > 16000
        ):
            data_list += [ds]

if benchmark == "B2Benchmark":
    data_list = data_list[:100]
elif benchmark == "B3Benchmark":
    data_list.pop(
        205
    )  # Remove time series that is particularly incorrectly labeled
    data_list = data_list[100:]

# -------------------------------
# Pre processing
# -------------------------------
det_lengths = np.zeros(len(data_list), int)
scaled_data_list = []
grids = []

# Get bin the observations
print("Binning the observations")
for s in range(len(data_list)):
    ds = data_list[s]
    ts = np.array(ds[val_name], dtype=float)
    T = len(ts)
    det_lengths[s] = int(detection_prop * T)
    # Truncate values
    cap_max = np.percentile(ts[: int(T * (1 - detection_prop))], cap_high_per)
    cap_min = np.percentile(ts[: int(T * (1 - detection_prop))], cap_low_per)
    # Enlarge boundaries in case of trend
    if cap_min < 0:
        cap_min *= boundaries_scale
    else:
        cap_min /= boundaries_scale
    if cap_max < 0:
        cap_max /= boundaries_scale
    else:
        cap_max *= boundaries_scale
    ts = np.clip(ts, cap_min, cap_max)

    # Get grid of evaluations
    if linear_grid:
        x_grid = np.linspace(cap_min, cap_max, dim + 1)
    else:
        x_grid = np.zeros(dim + 1)
        percentiles = [
            p for p in np.linspace(cap_low_per, cap_high_per, dim + 1)
        ]
        x_grid[0 : dim + 1] = np.percentile(ts, percentiles)
        x_grid[0] = cap_min
        x_grid[dim] = cap_max

    if x_grid[0] != cap_min or x_grid[dim] != cap_max:
        print("Series {}".format(s))
        print("Cap = ({}, {})".format(cap_min, cap_max))
        print("Grid = {}".format(x_grid))

    grids += [x_grid]

    scaled_data = np.zeros((T, dim))
    for t in range(T):
        _, ecdf_t = ecdf([ts[t]], x_grid)
        scaled_data[t, :] = ecdf_t[1:] - ecdf_t[:-1]
        if np.sum(scaled_data[t, :]) == 0:
            print("Series {}, time = {}, ".format(s, t), end="")
            print("Ts value = {}, ".format(ts[t]), end="")
            print("Cap = ({}, {})".format(cap_min, cap_max))

    scaled_data_list += [scaled_data]

# Aggregate data
print("Aggregating data, n_obs = {}".format(n_obs))
s = 0
if n_obs == 1:
    aggregated_data_list = scaled_data_list
else:
    aggregated_data_list = []
    for ds in scaled_data_list:
        T = ds.shape[0]
        agg_ds = np.cumsum(ds, axis=0).take(
            list(range(n_obs - 1, T, n_obs)), axis=0
        )
        T = agg_ds.shape[0]
        agg_ds[1:, :] = agg_ds[1:, :] - agg_ds[:-1, :]
        det_lengths[s] = int(detection_prop * T)
        aggregated_data_list += [agg_ds]
        s += 1

# Divide data into train and test
n_series = len(scaled_data_list)

train_ds = ListDataset(
    [
        {
            "target": aggregated_data_list[s].transpose()[
                :, : -det_lengths[s]
            ],
            "start": data_list[s]["start"],
        }
        for s in range(n_series)
    ],
    freq=freq,
    one_dim_target=False,
)

test_ds = ListDataset(
    [
        {"target": aggregated_data_list[s].transpose(), "start": start}
        for s in range(n_series)
    ],
    freq=freq,
    one_dim_target=False,
)

# -------------------------------
# Fit model
# -------------------------------
estimator = MyDeepAREstimator(
    prediction_length=prediction_length,
    context_length=context_length,
    freq=freq,
    scaling=False,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        learning_rate=learning_rate,
        hybridize=True,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
    ),
    # distr_output=MultivariateGaussianOutput(dim=dim),
    # distr_output=LowrankMultivariateGaussianOutput(dim=dim, rank=20),
    distr_output=DirichletMultinomialOutput(dim=dim, n_trials=n_obs),
    lags_seq=lags_seq,
    pick_incomplete=False,
)

trained_output = estimator.train_model(train_ds)
transf = estimator.create_transformation()
detector = estimator.create_detector(transf, trained_output.trained_net)

# -------------------------------
# Anomaly detection
# -------------------------------
# With ll no samples
mc_samples = 1000
p_values_list = []
p_values_mult_list = []
auc = np.zeros(n_series)
auc_mult = np.zeros(n_series)

tic = time.time()
print("Performing anomaly detection:")
for s in range(n_series):

    T = len(data_list[s][val_name])
    lag = T % n_obs

    # Mimic a streaming setting, get_ds_list returns a list
    # of datasets of length detection length, here:
    #    detection_length = (1-detection_prop)*length_of_ts.
    # Every element of the list is incremented by one observation
    # compared to the preceding element of the list, which corresponds to
    # observing one new observation at a time.
    ts = scaled_data_list[s].transpose()
    agg_ts = aggregated_data_list[s].transpose()
    test_ds = ListDataset(
        [{"target": agg_ts, "start": start}], freq=freq, one_dim_target=False,
    )
    detection_length = det_lengths[s]
    ds_list = get_ds_list(test_ds, detection_length)

    p_values = np.zeros(detection_length * n_obs)
    p_values_mult = np.zeros(detection_length * n_obs)

    data_entry = ListDataset(ds_list, freq=freq, one_dim_target=False,)

    distr_gen = detector.predict(data_entry)

    cap_min = grids[s][0]
    cap_max = grids[s][dim]
    if cap_min < 0:
        cap_min *= 2
    else:
        cap_min /= 2
    if cap_max < 0:
        cap_max /= 2
    else:
        cap_max *= 2

    print("Detection_length = {}".format(detection_length))

    y_true = np.array(
        data_list[s][anom_name][T - detection_length * n_obs - lag : T - lag],
        int,
    )

    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        for k in range(detection_length):

            distr = next(distr_gen).distribution

            distr_cat = DirichletMultinomial(
                dim=distr.dim, n_trials=1, alpha=distr.alpha
            )

            ll_vect = distr_cat.log_prob(mx.nd.eye(dim)).asnumpy().flatten()

            mult_obs = np.zeros(dim)

            # Compute the single observation p_value, which is
            #     p = Prob(Likelihood(X) < likelihood(Obs)).
            # For the single observation p_value we do not need a Monte Carlo
            # scheme since the number of possible outcomes is dim which is typically
            # small enough to allow computing the likelihood of all possible outcomes.
            for t in range(n_obs):
                k_ = k * n_obs + t
                neg_idx = (detection_length - k) * n_obs + t - lag
                ts_val = data_list[s][val_name][-neg_idx]

                # Remark, in the following we suppose that the new observations are already
                # binned when fed to the anomaly detection module, since we already did the
                # binning in the preprocessing. However, in the more realistic situation where
                # the observations come as values (here noted ts_val) and not the bin,
                # in the following whenever appears
                #    ts[:, -neg_idx],
                # it should be replaced by binned_obs that can be defined from the value as:
                #    _, ecdf_t = ecdf([ts_val], x_grid)
                #    binned_obs = ecdf_t[1:] - ecdf_t[:-1]
                obs_bin = np.where(ts[:, -neg_idx])[0][0]

                # If the observation is too small or too big compared
                # to the grid, then assign a fixed p_value depending
                # on the parameters of the grid. Note that such an observation
                # is outside of the 100-cap_low_per-cap_high_per percent total mass
                # region of the training set (so usually outside 99%). Therefore such an
                # observation should be considered anomalous.
                if ts_val < cap_min or ts_val > cap_max:
                    p_values[k_] = cap_low_per / 100 / 2
                else:
                    obs_ll = ll_vect[obs_bin]
                    cdf_ = np.sum(np.exp(ll_vect[ll_vect <= obs_ll]))
                    p_values[k_] = cdf_

                mult_obs += ts[:, -neg_idx]

            # Compute the Multinomial observation p_value, which is
            #     p = Prob(Likelihood({X_1,...,X_n_obs}) < likelihood(Last n_obs Observations)),
            # this is done every n_obs observation, ant NOT with a sliding window (to reduce the computational
            # cost).
            # For the Multinomial observation p_value, the number of possible outcomes is to large (of order dim^n_obs),
            # so we need to use a Monte Carlo estimate of p, with mc_samples random samples from the predicted
            # Dirichlet-Multinomial distribution
            try:
                if n_obs > 1:
                    samples_dirmult = sample_dir_mult(
                        distr.alpha.asnumpy().flatten(), n_obs, mc_samples
                    )
                    samples_dirmult = np.append(
                        samples_dirmult, [mult_obs], axis=0
                    )
                    samples_dirmult = mx.nd.array(samples_dirmult)
                    ll_array = distr.log_prob(samples_dirmult).asnumpy()
                    mult_mc_ll = ll_array[:-1]
                    mult_obs_ll = ll_array[-1]
                    samples_ll = mult_mc_ll[np.isfinite(mult_mc_ll)]

                    p_values_mult[k * n_obs : (k + 1) * n_obs] = np.sum(
                        samples_ll <= mult_obs_ll
                    ) / len(samples_ll)
                else:
                    p_values_mult[k] = p_values[k]

            except:
                print(
                    "Unexpected error on ts {}:".format(s), sys.exc_info()[0]
                )
                pass

    detection_length = det_lengths[s]

    if np.sum(y_true) == 0:
        auc[s] = 0
        auc_mult[s] = 0
    else:
        auc[s] = roc_auc_score(y_true, (1 - p_values))
        auc_mult[s] = roc_auc_score(y_true, (1 - p_values_mult))

    print(
        "Progress: {}\%, average AUC = {}".format(
            s / n_series * 100, np.mean(auc[auc > 0])
        )
    )
    print(
        "Progress: {}\%, average AUC Mult = {}".format(
            s / n_series * 100, np.mean(auc_mult[auc_mult > 0])
        )
    )
    p_values_list += [p_values]
    p_values_mult_list += [p_values_mult]

print(time.time() - tic)

# Computing AUC with both stages
auc = np.zeros(n_series)
auc_mult = np.zeros(n_series)
auc_both1 = np.zeros(n_series)
auc_both2 = np.zeros(n_series)
eps = 1e-32

for s in range(n_series):
    detection_length = det_lengths[s]
    T = len(data_list[s][val_name])
    lag = T % n_obs
    y_true = np.array(
        data_list[s][anom_name][T - detection_length * n_obs - lag : T - lag],
        int,
    )
    p_val = p_values_list[s]
    p_val_mult = p_values_mult_list[s]
    if np.sum(y_true) == 0:
        auc[s] = 0
        auc_mult[s] = 0
        auc_both1[s] = 0
        auc_both2[s] = 0
    else:
        auc[s] = roc_auc_score(y_true, (1 - p_val))
        auc_mult[s] = roc_auc_score(y_true, (1 - p_val_mult))
        auc_both1[s] = roc_auc_score(y_true, (1 - p_val) + (1 - p_val_mult))
        auc_both2[s] = roc_auc_score(
            y_true, -np.log(p_val + eps) - np.log(p_val_mult + eps)
        )

print(
    "auc single = {}, on {}\% of series".format(
        np.mean(auc[auc > 0]), np.sum(auc > 0) / n_series * 100
    )
)
print(
    "auc mult = {}, on {}\% of series".format(
        np.mean(auc_mult[auc > 0]), np.sum(auc > 0) / n_series * 100
    )
)

print(
    "auc both = {}, on {}\% of series".format(
        np.mean(auc_both1[auc > 0]), np.sum(auc > 0) / n_series * 100
    )
)
print(
    "auc both log = {}, on {}\% of series".format(
        np.mean(auc_both2[auc > 0]), np.sum(auc > 0) / n_series * 100
    )
)
