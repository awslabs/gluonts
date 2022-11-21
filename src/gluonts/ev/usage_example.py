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

from gluonts.dataset.split import split
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.npts import NPTSPredictor
from gluonts.ev import SumQuantileLoss, MSE

from gluonts.ev import evaluate


def univariate_case():
    dataset = get_dataset("exchange_rate")

    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq

    _, test_template = split(dataset=dataset.test, offset=-prediction_length)
    test_data = test_template.generate_instances(
        prediction_length=prediction_length
    )

    predictor = NPTSPredictor(prediction_length=prediction_length, freq=freq)

    evaluation_result = predictor.backtest(
        metrics=[SumQuantileLoss(q=0.9), MSE()],
        test_data=test_data,
        axis=1,
    )
    for name, value in evaluation_result.items():
        print(f"\n{name}: {value}")


def multivariate_case():
    batch_count = 2
    shape = (10, 4, 3)  # forecasts_per_batch, prediction length, variate count
    random_data_batches = [
        {
            "label": np.random.random(shape),
            # there is only one seasonal error for the whole prediction length
            "seasonal_error": np.random.random((shape[0], 1, shape[2])),
            "mean": np.random.random(shape),
            "0.9": np.random.random(shape),
        }
        for _ in range(batch_count)
    ]

    evaluation_result = evaluate(
        metrics=[SumQuantileLoss(q=0.9), MSE()],
        data_batches=random_data_batches,
        axis=1,
    )

    for name, value in evaluation_result.items():
        print(f"\n{name}: {value}")


univariate_case()
multivariate_case()
