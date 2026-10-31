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
import pytest

from gluonts.dataset.common import ListDataset
from gluonts.model.trivial.mean import MeanEstimator


def train_predictor(targets, prediction_length=2, num_samples=3, freq="D"):
    dataset = ListDataset(
        [{"target": target, "start": "2020"} for target in targets],
        freq=freq,
    )
    estimator = MeanEstimator(
        prediction_length=prediction_length,
        num_samples=num_samples,
    )
    return estimator.train(dataset), dataset


@pytest.mark.parametrize(
    "targets, prediction_length, expected_mean",
    [
        ([[1, 2, 3], [1, 4, 5]], 2, [3.0, 4.0]),
        ([[1, 2, 3]], 3, [1.0, 2.0, 3.0]),
        # NaN in one series must not poison the mean (issue #2175)
        ([[1, 2, np.nan], [1, 4, 5]], 2, [3.0, 5.0]),
        ([[np.nan, np.nan, np.nan], [1, 4, 5]], 2, [4.0, 5.0]),
    ],
)
def test_mean_estimator(targets, prediction_length, expected_mean):
    predictor, dataset = train_predictor(
        targets, prediction_length=prediction_length
    )

    forecast = next(iter(predictor.predict(dataset)))

    assert forecast.samples.shape == (3, prediction_length)
    np.testing.assert_equal(forecast.samples[0], expected_mean)
