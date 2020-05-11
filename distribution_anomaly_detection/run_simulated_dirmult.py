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

from gluonts.dataset.common import ListDataset
from mydeepar import MyDeepAREstimator
from gluonts.trainer import Trainer
from gluonts.distribution.dirichlet_multinomial import (
    DirichletMultinomialOutput,
)

from stats_tools import ecdf
from utils import get_ds_list


n_runs = 10
file_name = "results_auc/simulated_dirmult/test_simulated_ds1_msig_test.txt"
# Parameters of to generate the data
T = 4000
detection_length = 2000
n_malfunctions = 60
n_obs = 60

# Data parameters
freq = "1H"
start = time.time()

# Dimension of the ecdf to model
dim = 10
# Type of grid on which to evaluate the ecdf
linear_grid = False

# Restrain the domain to (cap_low_per percentile, cap_hig_per percentile)
cap_low_per = 0.001
cap_high_per = 99.999

# Model
model = "deepar"

# Trainer parameters
epochs = 50
learning_rate = 1e-2
batch_size = 10
num_batches_per_epoch = 50
prediction_length = 1

metrics = np.zeros((n_runs, 3))
np.random.seed(0)

for i in range(n_runs):
    dates = range(T)

    period = 24
    noise = np.random.normal(scale=0.1, size=T)
    mean_ts = np.sin([2 * math.pi * k / period for k in range(T)]) + noise
    sigma_ts = np.ones(T)

    malfunctions_idx = (
        T
        - 1
        - np.random.choice(
            np.arange(detection_length), n_malfunctions, replace=False
        )
    )
    sigma_ts[malfunctions_idx] -= 0.5

    samples = np.random.normal(mean_ts, sigma_ts, size=(n_obs, T))
    samples = samples.transpose()

    # Warning: Clipping values
    cap_min = np.percentile(
        samples[:, : T - detection_length].flatten(), cap_low_per
    )
    cap_max = np.percentile(
        samples[:, : T - detection_length].flatten(), cap_high_per
    )
    samples = np.clip(samples, cap_min, cap_max)

    # Get grid of evaluations
    if linear_grid:
        x_grid = np.linspace(cap_min, cap_max, dim + 1)
    else:
        x_grid = np.zeros(dim + 1)
        percentiles = [
            p for p in np.linspace(cap_low_per, cap_high_per, dim + 1)
        ]
        x_grid[0 : dim + 1] = np.percentile(samples.flatten(), percentiles)

    target_matrix = np.zeros((T, dim))
    for t in range(T):
        _, ecdf_t = ecdf(samples[t, :], x_grid)
        target_matrix[t, :] = (ecdf_t[1:] - ecdf_t[:-1]) * n_obs

    data_matrix = target_matrix.transpose()

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
        context_length=10 * prediction_length,
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
        lags_seq=[1, 2, 3, 4, 24, 48, 72, 96],
        pick_incomplete=False,
    )

    trained_output = estimator.train_model(train_ds)

    # Third-party imports
    import mxnet as mx

    def sample_one_dir_mult(alpha, n_trials):
        p = np.random.dirichlet(alpha)
        return np.random.multinomial(n_trials, p)

    def sample_dir_mult(alpha, n_trials, n_samples):
        if n_trials == 1:
            return sample_one_dir_mult(alpha, n_trials)
        else:
            return np.array(
                [
                    sample_one_dir_mult(alpha, n_trials)
                    for t in range(n_samples)
                ]
            )

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
        samples_dirmult = mx.nd.array(
            sample_dir_mult(distr.alpha.asnumpy().flatten(), n_obs, mc_samples)
        )
        mc_ll[k, :] = distr.log_prob(samples_dirmult).asnumpy().flatten()
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

    y_true = np.zeros(detection_length)
    y_true[malfunctions_idx - (T - detection_length)] = 1
    metrics[i, 2] = roc_auc_score(y_true, (1 - p_values))

    print("Simulation {}:".format(i))
    print("     Nbre detected malfunctions = {}".format(n_detected_ma))
    print("     Nbre statistical anomalies = {}".format(n_detected_sa))
    print("     AUC = {}".format(metrics[i, 2]))

print(metrics)

print(np.mean(metrics, axis=0))
print(np.std(metrics, axis=0))
print(np.percentile(metrics, 5, axis=0))
print(np.percentile(metrics, 95, axis=0))

np.savetxt(file_name, metrics)
