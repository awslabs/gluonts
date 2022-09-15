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

from gluonts.ev_with_decorators.api import evaluate, ForecastBatch
from gluonts.ev_with_decorators.metrics import (
    mse,
    QuantileLoss,
    nd,
    SeasonalError,
    AbsError,
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
    AbsError(),
    mse,
    nd,
    *(QuantileLoss(q=q) for q in quantile_values),
    SeasonalError(freq=electricity.metadata.freq),
]

# EVALUATION
eval_result = evaluate(metrics=metrics_to_evaluate, data=input_data)

print("RESULT")
print("\nBASE METRICS")
base_metrics = eval_result.get_base_metrics()
for key, value in base_metrics.items():
    print(f"'{key}' has shape {np.shape(value)}")

# TODO: how should parametric metrics be retrieved?
# print(base_metrics["abs_error"]["median"])


print("\nAGGREGATED METRICS:")
for key, value in eval_result.get_aggregate_metrics(axis=1).items():
    print(f"'{key}': {value}]")

"""
RESULT

BASE METRICS
'abs_error' has shape ()
'quantile_loss[0.1]' has shape (10, 12)
'quantile_loss[0.9]' has shape (10, 12)
'quantile_loss[0.5]' has shape (10, 12)

AGGREGATED METRICS:
'mse': [6.47952803e+01 1.16923472e+04 5.49424643e+01 1.86574762e+05
 2.59215030e+04 5.30528699e+05 1.77747512e+02 1.06394626e+06
 3.87500220e+04 7.06908010e+04]]
'seasonal_error': [ 5.874899   8.693754   3.8237948 53.10183   21.047922
89.41482  2.6516266 97.68416   36.1574    33.661602 ]]
'nd': [0.937218   0.99556889 0.95046573 0.99899208 0.99619891 0.99955995
 0.96810028 0.99956537 0.99716703 0.99801236]]
"""
