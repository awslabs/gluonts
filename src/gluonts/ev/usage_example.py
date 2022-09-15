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
import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.split import split

from gluonts.ev.evaluator import Evaluator
from gluonts.mx import SimpleFeedForwardEstimator, Trainer

# SETUP
dataset = get_dataset(
    "m4_hourly"
).train  # let's pretend, this is the entire dataset
prediction_length = get_dataset("m4_hourly").metadata.prediction_length

training_dataset, test_template = split(
    dataset, date=pd.Period("1750-01-07 00:00:00", freq="1H")
)

test_pairs = test_template.generate_instances(
    prediction_length=prediction_length
)

estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=prediction_length,
    trainer=Trainer(
        ctx="cpu", epochs=1, learning_rate=1e-3, num_batches_per_epoch=100
    ),
)

predictor = estimator.train(training_dataset)
forecast_it = predictor.predict(
    # expected datatype is Dataset so this is not the best way to do it...
    dataset=test_pairs.input
)


# EVALUATION
# let's get the MSE per entry as well as aggregated and also the mean MAPE
# TODO: this is broken because metrics aren't properly sorted

# define a custom aggregation function
def sum_of_last_ten(values: np.ndarray) -> float:
    return np.sum(values[-10:]).item()


evaluator = Evaluator()

metrics = evaluator.apply(test_pairs, forecast_it)

# we can get very detailed metrics (for every time stamp of every series)
global_metrics = metrics.get_point_metrics()
print(np.shape(global_metrics["error"]))  # 2 dimensional values

# local metrics refer to metrics that consist
# of a single number per entry in test dataset
print(
    pd.DataFrame(metrics.get_local_metrics())
    .rename_axis("item_id")
    .reset_index()
)

# global metrics describe the entire dataset in one number
print(metrics.get_global_metrics())

"""
RESULT:

{'mse_mean': 11440645.357514339, 'mape_mean': 0.40203682122671086}
(414, 48)
     item_id            mse
0          0    6409.585938
1          1  504531.250000
2          2   12911.440430
3          3  164485.031250
4          4   65625.437500
..       ...            ...
409      409    1917.923462
410      410    2463.419189
411      411    2342.378906
412      412     132.304184
413      413     359.834381

[414 rows x 2 columns]
{'mse_mean': 11440645.357514339, 'mape_mean': 0.40203682122671086}
"""
