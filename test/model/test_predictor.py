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

# Third-party imports
import numpy as np

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.predictor import Localizer
from gluonts.model.trivial.mean import MeanEstimator


def test_localizer():
    dataset = ListDataset(
        data_iter=[
            {
                "start": "2012-01-01",
                "target": (np.zeros(20) + i * 0.1 + 0.01),
                "id": f"{i}",
            }
            for i in range(3)
        ],
        freq="1H",
    )

    estimator = MeanEstimator(prediction_length=10, freq="1H", num_samples=50)

    local_pred = Localizer(estimator=estimator)
    agg_metrics, _ = backtest_metrics(
        test_dataset=dataset, predictor=local_pred
    )
