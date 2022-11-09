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

from gluonts.dataset.split import TestTemplate, OffsetSplitter
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.npts import NPTSPredictor
from gluonts.ev.metrics import MSIS, MSE, SumQuantileLoss

dataset = get_dataset("exchange_rate")

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq

test_template = TestTemplate(
    dataset=dataset.test, splitter=OffsetSplitter(offset=-prediction_length)
)

test_data = test_template.generate_instances(
    prediction_length=prediction_length
)

predictor = NPTSPredictor(prediction_length=prediction_length, freq=freq)

metrics_per_entry = [MSE(), SumQuantileLoss(q=0.9), MSIS()]
evaluation_result = predictor.backtest(
    metrics=metrics_per_entry, test_data=test_data, axis=1, num_samples=100
)
for name, value in evaluation_result.items():
    print(f"\n{name}: {value}")
