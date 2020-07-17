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

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.model.trivial.mean import MovingAveragePredictor

# Third-party imports
import numpy as np


def set_data(target, freq):
    """
    Sets test data in the right format
    """

    start = "2020"
    ds = ListDataset([{"target": target, "start": start}], freq=freq)
    data = list(ds)

    return data


def get_predictions(data, prediction_length, context_length, freq):
    """
    Gets predictions based on moving average
    """

    mp = MovingAveragePredictor(
        prediction_length=prediction_length,
        context_length=context_length,
        freq=freq,
    )
    predictions = mp.predict_item(data[0]).samples[0]

    return predictions


def check_equality_constant_sequence(
    predictions, constant_value, prediction_length
):
    """
    Checks if prediction values coincide with expected values.  This is for the case where the input is constant:
    expected = [constant_value, constant_value, ..., constant_value]
    """

    expected = [constant_value] * prediction_length

    if np.isnan(constant_value):
        return list(np.isnan(predictions)) == list(np.isnan(expected))
    else:
        return list(predictions) == expected


def run_evaluations(
    data,
    freq,
    constant_value,
    context_length_values=range(1, 10),
    prediction_length_values=range(1, 10),
):
    """
    Executes generic tests based on settings provided by input parameters
    Performs asserts on the output and shape of output.
    """

    for context_length in context_length_values:
        for prediction_length in prediction_length_values:
            predictions = get_predictions(
                data, prediction_length, context_length, freq
            )
            assert check_equality_constant_sequence(
                predictions, constant_value, prediction_length
            )
            assert predictions.shape == (prediction_length,)


def test_constant_sequence():
    constant_value = 1
    target_length = 3
    target = [constant_value] * target_length  # [1, 1, 1]
    freq = "D"
    data = set_data(target, freq)

    run_evaluations(data, freq, constant_value)


def test_length_one_sequence():
    constant_value = 1
    # target_length = 1
    target = [constant_value]
    freq = "D"
    data = set_data(target, freq)

    run_evaluations(data, freq, constant_value)


def test_empty_sequence():
    constant_value = np.nan
    target_length = 0
    target = []
    freq = "D"
    data = set_data(target, freq)

    run_evaluations(data, freq, constant_value)


def test_nan_sequence():
    constant_value = np.nan
    target_length = 3
    target = [constant_value] * target_length  # [1, 1, 1]
    freq = "D"
    data = set_data(target, freq)

    run_evaluations(data, freq, constant_value)
