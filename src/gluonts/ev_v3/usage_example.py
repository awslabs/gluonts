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

from gluonts.dataset.split import TestTemplate, OffsetSplitter
from gluonts.ev_v3.evaluator import NewEvaluator
from gluonts.model.npts import NPTSPredictor
from gluonts.dataset.repository.datasets import get_dataset

# DATASET
dataset = get_dataset("electricity")

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq

test_template = TestTemplate(
    dataset=dataset.test, splitter=OffsetSplitter(offset=prediction_length)
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
