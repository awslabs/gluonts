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
from gluonts.ev_v4.api import evaluate, MetricSpec
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
    MetricSpec(  # metric name is fn.__name__ if not specified otherwise
        fn=mse,
        parameters={"axis": 1},
    ),
    MetricSpec(  # metrics per timestep are possible, using axis=0
        name="MSE_per_timestamp",
        fn=mse,
        parameters={"axis": 0},
    ),
    MetricSpec(  # default parameters like 'forecast_type' can be changed
        name="MSE_with_p90_error",
        fn=mse,
        parameters={"forecast_type": "0.9", "axis": 1},
    ),
    MetricSpec(fn=mae_coverage),  # fn is the only required argument
    MetricSpec(
        fn=msis,
        parameters={"freq": freq, "axis": 1},
    ),
)

eval_result = evaluate(test_data, forecast_it, metric_specs)
for metric_name, value in eval_result.items():
    print(f"\n{metric_name}: {value}")

"""
RESULT:

mse: [1.70037725e+02 7.53495375e+01 2.81950000e-01 1.26936799e+03
 3.13162196e+02 2.91519228e+03 2.86863750e+00 4.53727826e+03
 2.04621995e+03 4.28342866e+03]

MSE_per_timestamp: [1423.07741 1176.70875  628.24209  248.08788  558.27488  336.70914
  343.80869 2367.45884 1195.56328  435.59824  638.73553  572.21965
  184.32789 1088.52898 4057.31885 2491.17487  686.56598 3162.31304
 2633.25986 2052.32446 2695.25204 1759.17983 1171.43795 5565.48113]

MSE_with_p90_error: [2.99645833e+03 2.03916667e+02 3.41666667e+00 7.24620833e+03
 8.09458333e+02 1.00476667e+04 1.37083333e+01 1.42806667e+04
 1.20904167e+03 6.22516667e+03]

mae_coverage: 0.06851851851851852

msis: [8.98503397 5.77328141 0.27383605 3.71135528 3.59816614 3.44007136
 1.86556138 4.06861266 9.66269812 3.07285557]

batch_size: 10
"""
