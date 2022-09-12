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
from typing import Dict

import numpy as np
from toolz import take

from gluonts.ev_with_decorators.api import evaluate
from gluonts.ev_with_decorators.metrics import (
    rmse,
    mse,
    QuantileLoss,
    error_wrt_mean,
    error_wrt_median,
    abs_error_wrt_median,
    nd,
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
quantile_predictions: Dict[float, list] = {q: [] for q in quantile_values}

for prediction in npts.predict(test_data):
    prediction_mean.append(prediction.mean)
    prediction_median.append(prediction.median)
    for q in quantile_values:
        quantile_predictions[q].append(prediction.quantile(q))

quantile_predictions_np = dict()
for key, value in quantile_predictions.items():
    quantile_predictions_np[key] = np.stack(value)

input_data = {
    "target": target,
    "prediction_mean": np.stack(prediction_mean),
    "prediction_median": np.stack(prediction_median),
}
for q, quantile_prediction_q in quantile_predictions_np.items():
    input_data[f"prediction_quantile[{q}]"] = quantile_prediction_q

metrics_to_evaluate = [
    abs_error_wrt_median,
    mse,
    nd,
    *(QuantileLoss(q=q) for q in quantile_values),
]

# EVALUATION
eval_result = evaluate(metrics=metrics_to_evaluate, data=input_data)

print("RESULT")
print("\nBASE METRICS (should have two-dimensional shape, like the inputs)")
for key, value in eval_result.get_base_metrics().items():
    print(f"'{key}' has shape {np.shape(value)}")

print("\nAGGREGATED METRICS:")
for key, value in eval_result.get_aggregate_metrics(axis=1).items():
    print(f"'{key}': {value}]")

"""
RESULT:

BASE METRICS (should have two-dimensional shape, like the inputs)
'quantile_loss[0.5]' has shape (10, 12)
'quantile_loss[0.9]' has shape (10, 12)
'quantile_loss[0.1]' has shape (10, 12)
'abs_error_wrt_median' has shape (10, 12)

AGGREGATED METRICS:
'mse': [2.07925175e+02 1.17761600e+02 1.18123333e+00 9.62617850e+03
 7.99396592e+02 1.38035072e+04 8.08445000e+00 1.76820675e+04
 5.03752386e+03 1.56677683e+04]]
'nd': [0.8961039  0.07307692 0.06382979 0.13913558 0.13259958 0.12517289
 0.15337423 0.10590714 0.30637916 0.46169772]]
"""
