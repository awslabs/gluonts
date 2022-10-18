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

from gluonts.ev.api import Evaluator
from gluonts.dataset.split import TestTemplate, OffsetSplitter
from gluonts.ev.metrics import MSE, MSIS, NRMSE, QuantileLossSum
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

# --- EVALUATION STARTS HERE ---

evaluator = Evaluator()
evaluator.add_metrics([MSE(), MSIS()], axis=1)
evaluator.add_metric(NRMSE())
evaluator.add_metric(QuantileLossSum(q=0.9), axis=0)

res = evaluator.evaluate(test_data, forecast_it)

for name, value in res.items():
    print(f"\n{name}: {value}")


"""

MSE[axis=1]: [1.52131758e+02 6.88867542e+01 3.00020833e-01 1.15193784e+03
 3.15931108e+02 2.63484380e+03 2.99962083e+00 4.66641007e+03
 2.05939438e+03 4.50878403e+03]

MSIS[axis=1]: [12.08145374  4.83207589  0.62075673  3.42150118  4.37868274  3.2297932
  3.44080163  3.83119893 10.97729275  5.0347862 ]

NRMSE[axis=None]: 0.14527278126172422

QuantileLossSum[axis=0]: [ 58.3  54.5  33.7  35.   43.1  31.8  31.2  47.1  52.9  55.1  37.7  31.7
  44.7  60.7 176.7 160.4  54.1  29.5  52.6  53.3  47.2  57.6  62.5 134.5]
"""
