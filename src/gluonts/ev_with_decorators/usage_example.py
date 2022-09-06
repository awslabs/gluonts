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

import numpy as np
from toolz import take

from gluonts.ev_with_decorators.api import evaluate
from gluonts.ev_with_decorators.metrics import (
    rmse,
    mse,
    mean_mse,
    mape,
    quantile_loss,
    coverage,
    abs_error,
)
from gluonts.model.npts import NPTSPredictor
from gluonts.dataset.repository.datasets import get_dataset

# GET DATA
npts = NPTSPredictor(prediction_length=12, freq="D")
electricity = get_dataset("electricity")
test_data = list(take(10, electricity.test))

# PREPARING DATA FOR EVALUATION
target = np.stack([true_values["target"][-12:] for true_values in test_data])

prediction_mean = []
prediction_median = []
quantile_values = (0.1, 0.5, 0.9)
quantile_predictions = {q: [] for q in quantile_values}

for prediction in npts.predict(test_data):
    prediction_mean.append(prediction.mean)
    prediction_median.append(prediction.median)
    for q in quantile_values:
        quantile_predictions[q].append(prediction.quantile(q))

quantile_predictions_np = dict()
for key, value in quantile_predictions.items():
    quantile_predictions_np[key] = np.stack(value)

# EVALUATION

input_data = {
    "target": target,
    "prediction_mean": np.stack(prediction_mean),
    "prediction_median": np.stack(prediction_median),
    "quantile_predictions": quantile_predictions_np,
}
metrics_to_calculate = [
    abs_error,
    mse,
    rmse,
    mape,
    mean_mse,
    *(quantile_loss(q) for q in quantile_values),
    *(coverage(q) for q in quantile_values),
]
metrics = evaluate(metrics_to_calculate, input_data)
for key, value in metrics.get_all().items():
    print(f"{key} has shape {np.shape(value)}; sum is {np.sum(value)}")

"""
RESULT:

abs_error has shape (10, 12); sum is 6028.869999999999
mse has shape (10,); sum is 64803.03997500001
mape has shape (10,); sum is 5.9675833294209
mean_mse has shape (); sum is 6480.303997500001
quantile_loss[0.1] has shape (10, 12); sum is 2472.2000000000003
quantile_loss[0.5] has shape (10, 12); sum is 5499.0
quantile_loss[0.9] has shape (10, 12); sum is 3039.2
coverage[0.1] has shape (10,); sum is 0.9166666666666667
coverage[0.5] has shape (10,); sum is 4.333333333333333
coverage[0.9] has shape (10,); sum is 8.416666666666666
"""
