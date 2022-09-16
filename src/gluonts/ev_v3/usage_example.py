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

from gluonts.ev_v3.api import evaluate, evaluate_batches
from gluonts.ev_v3.helpers import get_input_batches, PrimitiveForecastBatch
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


# SCENARIO 1: Entire dataset fits into a single np.ndarray
def scenario_1():
    prediction_length = 12
    data_entry_count = 50

    npts = NPTSPredictor(prediction_length=prediction_length, freq="D")
    electricity = get_dataset("electricity")
    test_data = list(take(data_entry_count, electricity.test))

    # PREPARE FOR EVALUATION
    metrics_to_evaluate = [
        MSE(axis=0),
        AbsError(error_type="p90"),
        MSE(axis=1),
        ND(axis=1),
        *(QuantileLoss(quantile=q) for q in (0.1, 0.5, 0.9)),
        SeasonalError(freq=electricity.metadata.freq, axis=1),
    ]

    target_batch = []
    past_data_batch = []
    for data_entry in test_data:
        target_batch.append(data_entry["target"][-prediction_length:])
        past_data_batch.append(data_entry["target"][:-prediction_length])

    # TODO: not use make_evaluation_predictions to get forecasts
    forecast_it, _ = make_evaluation_predictions(
        dataset=test_data, predictor=npts, num_samples=10
    )

    input_data = {
        "target": np.stack(target_batch),
        "past_data": np.stack(past_data_batch),
        "forecast": PrimitiveForecastBatch(list(forecast_it)),
    }

    # EVALUATE
    result = evaluate(
        metrics_to_evaluate, input_data
    )  # everything is done in one batch

    print("SHAPES OF METRICS:")
    for metric_name, value in result.items():
        print(f"{metric_name} has shape {np.shape(value)}")

    # aggregating to "global" metrics has to be done by the user
    print("\nGLOBAL METRICS:")
    print(f"Mean MSE: {np.mean(result['mse[mean,axis=0]'])}")
    print(f"Mean QuantileLoss[0.9]: {np.mean(result['QuantileLoss[0.9]'])}")
    print("and so on...")


# SCENARIO 2: The model works with some batch size. Evaluation should also
# happen in batches of that size. At the end, the partial results need to
# be combined.
def scenario_2():
    prediction_length = 12
    data_entry_count = 50
    eval_batch_size = 16

    npts = NPTSPredictor(prediction_length=prediction_length, freq="D")
    electricity = get_dataset("electricity")
    test_data = list(take(data_entry_count, electricity.test))

    # PREPARE FOR EVALUATION
    metrics_to_evaluate = [
        MSE(axis=0),
        AbsError(error_type="p90"),
        MSE(axis=1),
        ND(axis=1),
        *(QuantileLoss(quantile=q) for q in (0.1, 0.5, 0.9)),
        SeasonalError(freq=electricity.metadata.freq, axis=1),
    ]

    target_batch = []
    past_data_batch = []
    for data_entry in test_data:
        target_batch.append(data_entry["target"][-prediction_length:])
        past_data_batch.append(data_entry["target"][:-prediction_length])

    # TODO: not use make_evaluation_predictions to get forecasts
    forecast_it, _ = make_evaluation_predictions(
        dataset=test_data, predictor=npts, num_samples=10
    )

    input_batch_it = get_input_batches(
        iter(test_data), forecast_it, batch_size=eval_batch_size
    )

    # EVALUATE
    result = evaluate_batches(metrics_to_evaluate, input_batch_it)

    print("SHAPES OF METRICS:")
    for metric_name, value in result.items():
        print(f"{metric_name} has shape {np.shape(value)}")

    # aggregating to "global" metrics has to be done by the user
    print("\nGLOBAL METRICS:")
    print(f"Mean MSE: {np.mean(result['mse[mean,axis=0]'])}")
    print(f"Mean QuantileLoss[0.9]: {np.mean(result['QuantileLoss[0.9]'])}")
    print("and so on...")


print("SCNEARIO #1 (entire dataset fits into a single np.ndarray):")
scenario_1()
print("\n" + "-" * 20)
print("\nSCNEARIO #2 (evaluation in batches):")
scenario_2()

"""
SCNEARIO #1 (entire dataset fits into a single np.ndarray):
SHAPES OF METRICS:
mse[mean,axis=0] has shape (12,)
abs_error[p90] has shape (50, 12)
mse[mean,axis=1] has shape (50,)
ND[median,axis=1] has shape (50,)
QuantileLoss[0.1] has shape (50, 12)
QuantileLoss[0.5] has shape (50, 12)
QuantileLoss[0.9] has shape (50, 12)
season_error[seasonality=24,axis=1] has shape (50,)
entry_count has shape ()

GLOBAL METRICS:
Mean MSE: 23.819333333333333
Mean QuantileLoss[0.9]: 20.984999999999996
and so on...

--------------------

SCNEARIO #2 (evaluation in batches):
SHAPES OF METRICS:
mse[mean,axis=0] has shape (12,)
abs_error[p90] has shape (50, 12)
mse[mean,axis=1] has shape (50,)
ND[median,axis=1] has shape (50,)
QuantileLoss[0.1] has shape (50, 12)
QuantileLoss[0.5] has shape (50, 12)
QuantileLoss[0.9] has shape (50, 12)
season_error[seasonality=24,axis=1] has shape (50,)
entry_count has shape ()

GLOBAL METRICS:
Mean MSE: 35.6125
Mean QuantileLoss[0.9]: 20.040166666666664
and so on...
"""
