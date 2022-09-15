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

from gluonts.ev_v3.api import (
    evaluate,
    get_input_batches,
)
from gluonts.ev_v3.metrics import (
    AbsError,
    QuantileLoss,
    MSE,
    ND,
    SeasonalError,
)
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model.npts import NPTSPredictor
from gluonts.dataset.repository.datasets import get_dataset

prediction_length = 12
data_entry_count = 50
eval_batch_size = 8

# GET DATA
npts = NPTSPredictor(prediction_length=prediction_length, freq="D")
electricity = get_dataset("electricity")
test_data = list(take(data_entry_count, electricity.test))

# PREPARING DATA FOR EVALUATION
forecast_it, _ = make_evaluation_predictions(
    dataset=test_data, predictor=npts, num_samples=10
)

metrics_to_evaluate = [
    MSE(axis=0),
    AbsError(error_type="p90"),
    MSE(axis=1),
    ND(axis=1),
    *(QuantileLoss(quantile=q) for q in (0.1, 0.5, 0.9)),
    SeasonalError(freq=electricity.metadata.freq, axis=1),
]

input_batch_it = get_input_batches(
    iter(test_data), forecast_it, batch_size=eval_batch_size
)

result = evaluate(metrics_to_evaluate, input_batch_it)
for metric_name, value in result.items():
    print(metric_name, np.shape(value))

print(f"\nentry_count is: {result['entry_count']}")

"""
RESULT:

mse[mean,axis=0] (12,)
abs_error[p90] (50, 12)
mse[mean,axis=1] (50,)
ND[median,axis=1] (50,)
QuantileLoss[0.1] (50, 12)
QuantileLoss[0.5] (50, 12)
QuantileLoss[0.9] (50, 12)
season_error[seasonality=24,axis=1] (50,)
entry_count ()

entry_count is: 50
"""