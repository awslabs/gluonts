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
from gluonts.ev_v4.api import MetricSpec, evaluate
from gluonts.ev_v4.metrics import mse, msis, mae_coverage
from gluonts.model.npts import NPTSPredictor
from gluonts.dataset.repository.datasets import get_dataset

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

# EVALUATION

metric_specs = (
    MetricSpec(
        name="MSE",
        fn=mse,
        parameters={"axis": 1},
    ),
    MetricSpec(  # we can do metrics per timestep
        name="MSE_per_timestamp",
        fn=mse,
        parameters={"axis": 0},
    ),
    MetricSpec(  # default parameters like 'forecast_type' can be changed
        name="MSE_with_p90_error",
        fn=mse,
        parameters={"forecast_type": "0.9", "axis": 1},
    ),
    MetricSpec(
        name="sMAPE",
        fn=mse,
        parameters={"axis": 1},
    ),
    MetricSpec(
        name="MSIS",
        fn=msis,
        parameters={"freq": freq, "axis": 1},
    ),
    MetricSpec(name="MAE_coverage", fn=mae_coverage, parameters={}),
)

eval_result = evaluate(test_data, forecast_it, metric_specs, batch_size=4)
for metric_name, value in eval_result.items():
    print(f"\n{metric_name}: {value}")

"""
RESULT:

MSE: [1.36719646e+02 7.64413667e+01 2.64516667e-01 1.20087124e+03
 2.88310425e+02 2.47937929e+03 2.83414583e+00 4.11440285e+03
 2.05571107e+03 5.10434000e+03]

MSE_per_timestamp: [1045.43335  981.91202  734.87873  251.49417  504.90605  330.36425
  367.72504 2628.43587 1487.34742  384.64904  563.69332  424.30361
  309.43651 1263.89806 3871.2717  2458.93248  521.8133  3773.91546
 3093.96546 2719.18772 2215.11494 1206.21988  789.50867 5173.85185]

MSE_with_p90_error: [2.84029167e+03 2.34791667e+02 2.16666667e+00 7.40220833e+03
 6.99208333e+02 1.00526250e+04 1.42500000e+01 1.46287083e+04
 1.26820833e+03 6.58812500e+03]

sMAPE: [1.36719646e+02 7.64413667e+01 2.64516667e-01 1.20087124e+03
 2.88310425e+02 2.47937929e+03 2.83414583e+00 4.11440285e+03
 2.05571107e+03 5.10434000e+03]

MSIS: [9.33260746 4.59426162 0.27383605 4.03763773 3.65069411 3.75857656
 1.84820732 4.20151052 9.57894201 3.10243782]

MAE_coverage: 0.07787037037037038

batch_size: 10
"""
