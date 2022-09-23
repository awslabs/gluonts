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
from gluonts.ev_v4.api import gather_inputs, MetricSpec, evaluate
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
        parameters={"forecast_type": "0.9", "axis": 0},
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
    MetricSpec(name="MAE_covergae", fn=mae_coverage, parameters={}),
)

data = gather_inputs(
    test_data=test_data, forecast_it=forecast_it, metric_specs=metric_specs
)

# only NumPy arrays needed for evaluation
eval_result = evaluate(data, metric_specs)
for metric_name, value in eval_result.items():
    print(f"\n{metric_name}: {value}")

"""
RESULT:

MSE: [1.38124313e+02 8.15331000e+01 2.72841667e-01 1.20578845e+03
 3.15274296e+02 2.95207122e+03 2.82054167e+00 5.15698071e+03
 2.06749965e+03 4.38588479e+03]

MSE_per_timestamp: [1509.1703  1109.15853  812.68517  273.12755  537.64231  288.20633
  365.86701 2389.41772 1390.56152  383.41257  629.6229   456.37039
  199.04059 1198.95448 4100.5623  2583.99104  696.75354 4063.59549
 2787.21606 2893.4757  2358.2764  1642.75257  908.37037 5556.76893]

MSE_with_p90_error: [143095.6 145683.4 150937.7 107801.   71526.9  54480.7  43231.3  42721.8
  68347.7  52429.4  30570.9  64439.9  81950.5  86576.3  99728.4 118101.3
 143323.8 130970.1 119324.  109725.3 112578.6 111907.9 111123.4 151992.4]

sMAPE: [ 68800.66666667  31213.58333333  69074.04166667  49509.70833333
  15537.25       179240.70833333  65602.33333333 477296.70833333
  18102.5          5859.29166667]

MSIS: [ 628.39211049  168.85651712  590.5826892   135.03444642    9.4747331
  191.17413774 1008.02789318  289.66750153   12.39590371    3.26292149]

MAE_covergae: 0.17685185185185187
"""
