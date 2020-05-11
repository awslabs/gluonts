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
import scipy.stats as scps

from gluonts.dataset.common import ListDataset
from mydeepar import MyDeepAREstimator
from gluonts.trainer import Trainer
from gluonts.distribution.dirichlet import DirichletOutput

from stats_tools import ecdf


n_runs = 10
file_name = "simulated_ds2_test.txt"
# Parameters of to generate the data
T = 4000
detection_length = 2000
n_malfunctions = 60

# Data parameters
freq = "1H"
start = time.time()

# Dimension of the ecdf to model
dim = 30
# Type of grid on which to evaluate the ecdf
linear_grid = True

# Add artificial anomaly
transform = False

# Restrain the domain to (cap_low_per percentile, cap_hig_per percentile)
cap_low_per = 0.001
cap_high_per = 99.999

# Model
model = "deepar"

# Trainer parameters
epochs = 100
learning_rate = 1e-2
batch_size = 10
num_batches_per_epoch = 50
prediction_length = 5

metrics = np.zeros((n_runs, 2))
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

    q_levels = np.linspace(1, 999, 1000) / 1000
    quantiles_matrix = np.array(
        [scps.norm.ppf(q_levels, mean_ts[t], sigma_ts[t]) for t in range(T)]
    )

    # Warning: Clipping values
    cap_min = np.percentile(quantiles_matrix.flatten(), cap_low_per)
    cap_max = np.percentile(quantiles_matrix.flatten(), cap_high_per)
    quantiles_matrix = np.clip(quantiles_matrix, cap_min, cap_max)

    # Get grid of evaluations
    if linear_grid:
        x_grid = np.linspace(cap_min, cap_max, dim + 1)
    else:
        x_grid = np.zeros(dim + 1)
        percentiles = [
            p for p in np.linspace(cap_low_per, cap_high_per, dim + 1)
        ]
        x_grid[0 : dim + 1] = np.percentile(
            quantiles_matrix.flatten(), percentiles
        )

    target_matrix = np.zeros((T, dim))
    for t in range(T):
        _, ecdf_t = ecdf(quantiles_matrix[t, :], x_grid)
        target_matrix[t, :] = ecdf_t[1:] - ecdf_t[:-1]

    data_matrix = target_matrix.transpose()

    epsilon = 1e-8

    data_matrix += epsilon
    data_matrix /= np.expand_dims(np.sum(data_matrix, axis=0), axis=0)

    custom_datasetx = np.zeros((1, dim, T))
    custom_datasetx[0, :, :] = data_matrix

    train_ds = ListDataset(
        [
            {"target": x, "start": start}
            for x in custom_datasetx[:, :, :-detection_length]
        ],
        freq=freq,
        one_dim_target=False,
    )

    test_ds = ListDataset(
        [{"target": x, "start": start} for x in custom_datasetx[:, :, :]],
        freq=freq,
        one_dim_target=False,
    )

    estimator = MyDeepAREstimator(
        prediction_length=prediction_length,
        context_length=2 * prediction_length,
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
        distr_output=DirichletOutput(dim=dim),
        lags_seq=[1, 2, 3, 4, 24, 48, 72, 96],
        pick_incomplete=False,
    )

    trained_output = estimator.train_model(train_ds)
    predictor_dirichlet = trained_output.predictor

    # Standard library imports
    from typing import Dict, Iterator, NamedTuple, Optional, Tuple, Union

    # Third-party imports
    import mxnet as mx

    from gluonts import transform
    from gluonts.dataset.common import DataEntry, Dataset
    from gluonts.dataset.loader import InferenceDataLoader
    from gluonts.support.util import get_hybrid_forward_input_names
    from gluonts.model.estimator import GluonEstimator
    from gluonts.transform import TransformedDataset

    def get_ds_list(dataset: Dataset, detection_length):
        def truncate_target(data, remove_n_item):
            data = data.copy()
            target = data["target"]
            assert (
                target.shape[-1] >= remove_n_item
            )  # handles multivariate case (target_dim, history_length)
            data["target"] = target[..., :-remove_n_item]
            return data

        # TODO filter out time series with target shorter than prediction length
        # TODO or fix the evaluator so it supports missing values instead (all
        # TODO the test set may be gone otherwise with such a filtering)

        list_ds = []
        for k in range(detection_length, 0, -1):
            list_ds += list(
                TransformedDataset(
                    dataset,
                    transformations=[
                        transform.AdhocTransform(
                            lambda ds: truncate_target(ds, k)
                        )
                    ],
                )
            )

        return list_ds

    transf = estimator.create_transformation()
    detector = estimator.create_detector(transf, trained_output.trained_net)

    # detection_length = prediction_length
    mc_samples = 1000
    ds_list = get_ds_list(test_ds, detection_length)
    mc_ll = np.zeros((detection_length, mc_samples))
    obs_ll = np.zeros(detection_length)
    p_values = np.zeros(detection_length)

    for k in range(detection_length):
        data_entry = ds_list[k]
        distr = next(detector.predict([data_entry])).distribution
        mc_ll[k, :] = (
            distr.log_prob(distr.sample(mc_samples)).asnumpy().flatten()
        )
        obs_ll[k] = (
            distr.log_prob(
                mx.nd.array(data_matrix[:, -(detection_length - k)])
            )
            .asnumpy()
            .flatten()
        )
        samples_ll = mc_ll[k, :]
        samples_ll = samples_ll[np.isfinite(samples_ll)]
        p_values[k] = np.sum(samples_ll <= obs_ll[k]) / len(samples_ll)

    n_detected_ma = np.sum(p_values[malfunctions_idx - 2000] < 0.05)
    n_detected_sa = (np.sum(p_values < 0.05) - n_detected_ma) / (
        len(p_values) - n_malfunctions
    )

    metrics[i, 0] = n_detected_sa
    metrics[i, 1] = n_detected_ma

print(metrics)

print(np.mean(metrics, axis=0))
print(np.percentile(metrics, 5, axis=0))
print(np.percentile(metrics, 95, axis=0))
np.savetxt(file_name, metrics)
