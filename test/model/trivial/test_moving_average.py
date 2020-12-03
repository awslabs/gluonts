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
from gluonts.model.trivial.mean import MovingAveragePredictor


def get_predictions(
    target, prediction_length=1, context_length=1, freq="D", start="2020"
):
    mp = MovingAveragePredictor(
        prediction_length=prediction_length,
        context_length=context_length,
        freq=freq,
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
