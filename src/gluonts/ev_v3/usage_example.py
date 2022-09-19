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
from more_itertools import take

from gluonts.dataset.split import TestTemplate, OffsetSplitter
from gluonts.ev_v3.evaluator import NewEvaluator
from gluonts.ev_v3.metrics import RMSE
from gluonts.model.npts import NPTSPredictor
from gluonts.dataset.repository.datasets import get_dataset

# DATASET
dataset = get_dataset("electricity")

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq

dataset_test = list(take(100, dataset.test))
test_template = TestTemplate(
    dataset=dataset_test, splitter=OffsetSplitter(offset=-prediction_length)
)

test_dataset = test_template.generate_instances(
    prediction_length=prediction_length
)

predictor = NPTSPredictor(prediction_length=prediction_length, freq=freq)

forecast_it = predictor.predict(dataset=test_dataset.input)


# EVALUATOR
# custom metric functions have to be of this signature
# TODO: is this flexible enough?
def error_plus_two(
    input_data: np.ndarray, prediction_target: np.ndarray, forecast: np.ndarray
) -> np.ndarray:
    error = prediction_target - forecast.mean
    return error + 2


evaluator = NewEvaluator()

eval_result = evaluator(
    dataset=test_dataset,
    forecasts=forecast_it,
    freq=freq,  # TODO: make metadata part of TestDataset
    metrics_per_timestamp=(RMSE(),),
    custom_metrics=[error_plus_two],
)

print("EVALUATION RESULT:")

print("\nBASE METRICS:")
for metric_name, value in eval_result.base_metrics.items():
    print(f"'{metric_name}' has shape {np.shape(value)}")

print("\nMETRICS PER ENTRY:")
for metric_name, value in eval_result.metrics_per_entry.items():
    print(f"'{metric_name}' has shape {np.shape(value)}")

print("\nMETRICS PER TIMESTAMP:")
for metric_name, value in eval_result.metric_per_timestamp.items():
    print(f"'{metric_name}' has shape {np.shape(value)}")

print("\nGLOBAL METRICS:")
for metric_name, value in eval_result.global_metrics.items():
    print(f"'{metric_name}': {value}")

print("\nCUSTOM METRICS:")
for metric_name, value in eval_result.custom_metrics.items():
    print(f"'{metric_name}' has shape {np.shape(value)}")

"""
EVALUATION RESULT:

BASE METRICS:
'Coverage[0.8]' has shape (100, 24)
'Coverage[0.2]' has shape (100, 24)
'Coverage[0.5]' has shape (100, 24)
'Coverage[0.9]' has shape (100, 24)
'Coverage[0.4]' has shape (100, 24)
'QuantileLoss[0.6]' has shape (100, 24)
'Coverage[0.6]' has shape (100, 24)
'QuantileLoss[0.2]' has shape (100, 24)
'QuantileLoss[0.8]' has shape (100, 24)
'QuantileLoss[0.4]' has shape (100, 24)
'QuantileLoss[0.5]' has shape (100, 24)
'QuantileLoss[0.1]' has shape (100, 24)
'AbsPredictionTarget' has shape (100, 24)
'Coverage[0.7]' has shape (100, 24)
'QuantileLoss[0.9]' has shape (100, 24)
'Coverage[0.1]' has shape (100, 24)
'QuantileLoss[0.3]' has shape (100, 24)
'QuantileLoss[0.7]' has shape (100, 24)
'Coverage[0.3]' has shape (100, 24)
'Error[mean]' has shape (100, 24)

METRICS PER ENTRY:
'MSIS[alpha=0.05],freq=1H,seasonality=None]' has shape (100,)
'SeasonalError[seasonality=24]' has shape (100,)
'MAPE[0.5]' has shape (100,)
'ND[0.5]' has shape (100,)
'RMSE[mean]' has shape (100,)
'sMAPE[0.5]' has shape (100,)
'MASE[0.5,freq=1H,seasonality=None]' has shape (100,)
'MSE[mean]' has shape (100,)

METRICS PER TIMESTAMP:
'RMSE[mean]' has shape (24,)

GLOBAL METRICS:
'mean_of_MSIS[alpha=0.05],freq=1H,seasonality=None]': 4.685227692506365
'mean_of_SeasonalError[seasonality=24]': 74.21931457519531
'mean_of_MAPE[0.5]': nan
'mean_of_ND[0.5]': 0.11427785798107187
'mean_of_RMSE[mean]': 89.51677569891281
'mean_of_sMAPE[0.5]': nan
'mean_of_MASE[0.5,freq=1H,seasonality=None]': 0.6866300017750376
'mean_of_MSE[mean]': 31706.2758755

CUSTOM METRICS:
'error_plus_two' has shape (100, 24)
"""
