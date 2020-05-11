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

import time
import numpy as np
import pandas as pd
import mxnet as mx

from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer
from gluonts.distribution.dirichlet_multinomial import (
    DirichletMultinomialOutput,
)

import matplotlib.pyplot as plt

from stats_tools import ecdf
from utils import get_ds_list

from sklearn.metrics import f1_score, roc_auc_score, auc


from mydeepar import MyDeepAREstimator


# -------------------------------
# Parameters of the run
# -------------------------------
np.random.seed(0)
mx.random.seed(0)

# Sub benchmark
benchmark = "A1Benchmark"

assert benchmark in (
    "A1Benchmark",
    "A2Benchmark",
    "A3Benchmark",
    "A4Benchmark",
), print("The benchmark has to be a yahoo webscope sub-benchmark")

# Proportion of the time series to use for testing (detection range)
detection_prop = 0.6

# Dimension of the model
dim = 100
# Type of grid to use
linear_grid = True
# Percentiles of the training data to use as boundaries
cap_low_per = 0.5
cap_high_per = 99.5
boundaries_scale = 5

# Trainer parameters
epochs = 200
learning_rate = 1e-4
num_batches_per_epoch = 50
batch_size = 64
prediction_length = 1
context_length = 5

if benchmark == "A1Benchmark":
    epochs = 100
    batch_size = 8

if benchmark == "A4Benchmark":
    dim = 150
    batch_size = 32
    cap_low_per = 0.1
    cap_high_per = 99.9
    boundaries_scale = 7.0

# Data parameters
n_obs = 1
freq = "1H"
start = time.time()
folder = "Yahoo data/ydata-labeled-time-series-anomalies-v1_0/{}/".format(
    benchmark
)
val_name = "value"
if benchmark in ("A1Benchmark", "A2Benchmark"):
    anom_name = "is_anomaly"
else:
    anom_name = "anomaly"

# -------------------------------
# Load data
# -------------------------------
files_names = os.listdir(folder)
if ".json" in files_names:
    files_names.remove(".json")
if "A3Benchmark_all.csv" in files_names:
    files_names.remove("A3Benchmark_all.csv")
if "A4Benchmark_all.csv" in files_names:
    files_names.remove("A4Benchmark_all.csv")

data_list = [
    pd.read_csv(os.path.join(folder, f_name)) for f_name in files_names
]

# -------------------------------
# Pre processing
# -------------------------------
det_lengths = np.zeros(len(data_list), int)
scaled_data_list = []
grids = []

for s in range(len(data_list)):
    ds = data_list[s]
    ts = ds[val_name].values
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

    grids += [x_grid]

    scaled_data = np.zeros((T, dim))
    for t in range(T):
        _, ecdf_t = ecdf([ts[t]], x_grid)
        scaled_data[t, :] = ecdf_t[1:] - ecdf_t[:-1]
        if np.sum(scaled_data[t, :]) == 0:
            print("{}".format(t), end=" ")

    scaled_data_list += [scaled_data]

# Divide data into train and test
n_series = len(scaled_data_list)

train_ds = ListDataset(
    [
        {
            "target": scaled_data_list[s].transpose()[:, : -det_lengths[s]],
            "start": start,
        }
        for s in range(n_series)
    ],
    freq=freq,
    one_dim_target=False,
)

test_ds = ListDataset(
    [
        {"target": scaled_data_list[s].transpose(), "start": start}
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
    distr_output=DirichletMultinomialOutput(dim=dim, n_trials=n_obs),
    lags_seq=[
        1,
        2,
        3,
        4,
        5,
        22,
        23,
        24,
        25,
        26,
        46,
        47,
        48,
        49,
        50,
        70,
        71,
        72,
        73,
        74,
    ],
    pick_incomplete=False,
)

trained_output = estimator.train_model(train_ds)
transf = estimator.create_transformation()
detector = estimator.create_detector(transf, trained_output.trained_net)

# -------------------------------
# Anomaly detection
# -------------------------------
p_values_list = []
auc = np.zeros(n_series)

for s in range(n_series):

    ts = scaled_data_list[s].transpose()
    test_ds = ListDataset(
        [{"target": ts, "start": start}], freq=freq, one_dim_target=False,
    )
    detection_length = det_lengths[s]
    ds_list = get_ds_list(test_ds, detection_length)
    p_values = np.zeros(detection_length)

    data_entry = ListDataset(ds_list, freq=freq, one_dim_target=False,)

    distr_gen = detector.predict(data_entry)

    for k in range(detection_length):

        ts_val = data_list[s][val_name].values[-(detection_length - k)]
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
        if ts_val < cap_min or ts_val > cap_max:
            p_values[k] = cap_low_per / 100 / 2
        else:
            data_entry = ds_list[k]
            distr = next(distr_gen).distribution
            obs_bin = np.where(ts[:, -(detection_length - k)])[0][0]
            ll_vect = distr.log_prob(mx.nd.eye(dim)).asnumpy().flatten()
            obs_ll = ll_vect[obs_bin]
            cdf_ = np.sum(np.exp(ll_vect[ll_vect <= obs_ll]))
            p_values[k] = cdf_

    detection_length = det_lengths[s]
    T = len(data_list[s][val_name].values)
    y_true = data_list[s][anom_name].values[T - detection_length : T]
    if np.sum(y_true) == 0:
        auc[s] = 0
    else:
        auc[s] = roc_auc_score(y_true, (1 - p_values))

    print(
        "Progress: {}\%, average AUC = {}".format(
            s / n_series * 100, np.mean(auc[auc > 0])
        )
    )
    p_values_list += [p_values]

print(
    "auc = {}, on {}\% of series".format(
        np.mean(auc[auc > 0]), np.sum(auc > 0) / n_series * 100
    )
)
