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

from gluonts.ev_v3.api import evaluate, ForecastBatch
from gluonts.ev_v3.metrics import (
    AbsError,
    QuantileLoss,
    MSE,
    ND,
    SeasonalError,
)
from gluonts.model.npts import NPTSPredictor
from gluonts.dataset.repository.datasets import get_dataset

# GET DATA
prediction_length = 12
data_entry_count = 10

npts = NPTSPredictor(prediction_length=prediction_length, freq="D")
electricity = get_dataset("electricity")

test_data = list(take(data_entry_count, electricity.test))

# PREPARING DATA FOR EVALUATION
target = np.stack(
    [true_values["target"][-prediction_length:] for true_values in test_data]
)
past_data = np.stack(
    [true_values["target"][:-prediction_length] for true_values in test_data]
)

input_data = {
    "target": target,
    "past_data": past_data,
    "forecast_batch": ForecastBatch(
        prediction_length=prediction_length, batch_size=data_entry_count
    ),
}

quantile_values = (0.1, 0.5, 0.9)
metrics_to_evaluate = [
    AbsError(error_type="p90"),
    MSE(axis=1),
    ND(axis=1),
    *(QuantileLoss(q=q) for q in quantile_values),
    SeasonalError(freq=electricity.metadata.freq, axis=1),
]

# EVALUATION
eval_result = evaluate(metrics=metrics_to_evaluate, data=input_data)

print("RESULT:\n")
for key, value in eval_result.items():
    print(f"metric '{key}' has shape {np.shape(value)}")

"""
RESULT:

metric 'abs_error[p90]' has shape (10, 12)
metric 'mse[mean,axis=1]' has shape (10,)
metric 'ND[median,axis=1]' has shape (10,)
metric 'quantile_loss[0.1]' has shape (10, 12)
metric 'quantile_loss[0.5]' has shape (10, 12)
metric 'quantile_loss[0.9]' has shape (10, 12)
metric 'season_error[seasonality=24,axis=1]' has shape (10,)
"""
