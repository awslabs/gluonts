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

from toolz import take

from gluonts.dataset.split import TestTemplate, OffsetSplitter
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.npts import NPTSPredictor
from gluonts.ev.api import MultiMetricEvaluator, get_evaluation_data_batches
from gluonts.ev.metrics import Coverage, MeanSquaredError, QuantileLoss

dataset = get_dataset("electricity")

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq

dataset_test = list(take(10, dataset.test))

test_template = TestTemplate(
    dataset=dataset_test, splitter=OffsetSplitter(offset=-prediction_length)
)

test_data = test_template.generate_instances(
    prediction_length=prediction_length
)

predictor = NPTSPredictor(prediction_length=prediction_length, freq=freq)
forecast_it = predictor.predict(dataset=test_data.input, num_samples=100)

# --- EVALUATION STARTS HERE ---

batches = get_evaluation_data_batches(test_data, forecast_it)

# OTPION 1:
# mse_evaluator = MeanSquaredError()(axis=1)
# mse = mse_evaluator.evaluate(batches)

# OPTION 2:
metrics_per_entry = [MeanSquaredError(), QuantileLoss()]

multi_metric = MultiMetricEvaluator()
multi_metric.add_metrics(metrics_per_entry, axis=1)
multi_metric.add_metric(Coverage(), axis=None)

res = multi_metric.evaluate(batches)

for name, value in res.items():
    print(f"\n{name}: {value}")
