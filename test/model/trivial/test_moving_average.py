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
from gluonts.model.trivial.mean import (
    MovingAveragePredictor,
    SkewedMeanPredictor,
)


def get_predictions(
    target, prediction_length=1, context_length=1, freq="D", start="2020"
):
    mp = MovingAveragePredictor(
        prediction_length=prediction_length,
        context_length=context_length,
    )

    ds = ListDataset([{"target": target, "start": start}], freq=freq)
    item = next(iter(ds))
    predictions = mp.predict_item(item).mean

    return predictions


@pytest.mark.parametrize(
    "data, expected_output, prediction_length, context_length",
    [
        ([1, 1, 1], [1], 1, 1),
        ([1, 1, 1], [1, 1], 2, 1),
        ([1, 1, 1], [1, 1, 1], 3, 1),
        ([1, 1, 1], [1], 1, 2),
        ([1, 1, 1], [1, 1], 2, 2),
        ([1, 1, 1], [1, 1, 1], 3, 2),
        ([1, 1, 1], [1], 1, 3),
        ([1, 1, 1], [1, 1], 2, 3),
        ([1, 1, 1], [1, 1, 1], 3, 3),
        ([], [np.nan] * 1, 1, 1),
        ([], [np.nan] * 2, 2, 1),
        ([], [np.nan] * 3, 3, 1),
        ([np.nan], [np.nan] * 1, 1, 1),
        ([1, 3, np.nan], [2], 1, 3),
        ([1, 3, np.nan], [2, 2.5], 2, 3),
        ([1, 3, np.nan], [2, 2.5, 2.25], 3, 3),
        ([1, 2, 3], [3], 1, 1),
        ([1, 2, 3], [3, 3], 2, 1),
        ([1, 2, 3], [3, 3, 3], 3, 1),
        ([1, 2, 3], [2.5], 1, 2),
        ([1, 2, 3], [2.5, 2.75], 2, 2),
        ([1, 2, 3], [2.5, 2.75, 2.625], 3, 2),
        ([1, 2, 3], [2], 1, 3),
        ([1, 2, 3], [2, 7 / 3], 2, 3),
        ([1, 2, 3], [2, 7 / 3, 22 / 9], 3, 3),
        ([1, 1, 1], [1], 1, None),
        ([1, 1, 1], [1, 1], 2, None),
        ([1, 1, 1], [1, 1, 1], 3, None),
        ([1, 3, np.nan], [2], 1, None),
        ([1, 3, np.nan], [2, 2], 2, None),
        ([1, 3, np.nan], [2, 2, 2], 3, None),
    ],
)
def testing(data, expected_output, prediction_length, context_length):
    predictions = get_predictions(
        data,
        prediction_length=prediction_length,
        context_length=context_length,
    )

    np.testing.assert_equal(predictions, expected_output)


@pytest.mark.parametrize(
    "prediction_length, num_samples, skewness, target, expected_mean, expected_std",
    [
        (5, 20, 10, np.array([1.0] * 50), 1.0, 0.0),
        (5, 20, -10, np.array([0.0] * 25 + [3.0] * 25), 1.5, 1.5),
        (5, 20, 0, np.array([2.0] * 49 + [1.5] * 1), 1.99, 0.01),
    ],
)
def test_skewed_mean_predictor(
    prediction_length,
    num_samples,
    skewness,
    target,
    expected_mean,
    expected_std,
):
    predictor = SkewedMeanPredictor(
        prediction_length=prediction_length,
        num_samples=num_samples,
        skewness=skewness,
    )

    item = {"target": target}
    forecast = predictor.predict_item(item)

    assert forecast.samples.shape == (num_samples, prediction_length)
    assert np.isclose(forecast.samples.mean(), expected_mean, atol=0.1)
    assert np.isclose(forecast.samples.std(), expected_std, atol=0.1)
