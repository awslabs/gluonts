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

import pytest
import numpy as np
from contextlib import suppress as do_not_raise

from gluonts.dataset.artificial import ComplexSeasonalTimeSeries


_eps = 1e-6

_length_low = np.array([100])
_min_val = np.array([-10000, 0, 5000])


@pytest.mark.parametrize("num_series", [30])
@pytest.mark.parametrize("prediction_length", [100])
@pytest.mark.parametrize("freq_str", ["M", "W", "D", "H", "min"])
@pytest.mark.parametrize(
    "length_low, length_high", zip(_length_low, _length_low + np.array([50]))
)
@pytest.mark.parametrize(
    "min_val, max_val", zip(_min_val, _min_val + np.array([5000, 2500, 1000]))
)
@pytest.mark.parametrize("is_integer", [True, False])
@pytest.mark.parametrize("proportion_missing_values", [0])
@pytest.mark.parametrize("is_noise", [True])
@pytest.mark.parametrize("is_scale", [True])
@pytest.mark.parametrize("percentage_unique_timestamps", [0.07])
@pytest.mark.parametrize("is_out_of_bounds_date", [True])
@pytest.mark.parametrize("clip_values", [True, False])
def test_complex_seasonal(
    num_series: int,
    prediction_length: int,
    freq_str: str,
    length_low: int,
    length_high: int,
    min_val: float,
    max_val: float,
    is_integer: bool,
    proportion_missing_values: float,
    is_noise: bool,
    is_scale: bool,
    percentage_unique_timestamps: float,
    is_out_of_bounds_date: bool,
    clip_values: bool,
) -> None:
    context = (
        do_not_raise()
        if length_low > prediction_length
        else pytest.raises(AssertionError)
    )
    with context:
        generator = ComplexSeasonalTimeSeries(
            num_series=num_series,
            prediction_length=prediction_length,
            freq_str=freq_str,
            length_low=length_low,
            length_high=length_high,
            min_val=min_val,
            max_val=max_val,
            is_integer=is_integer,
            proportion_missing_values=proportion_missing_values,
            is_noise=is_noise,
            is_scale=is_scale,
            percentage_unique_timestamps=percentage_unique_timestamps,
            is_out_of_bounds_date=is_out_of_bounds_date,
            clip_values=clip_values,
        )

        train = generator.train
        test = generator.test

        assert len(train) == len(test) == num_series

        for ts_train, ts_test in zip(train, test):
            assert ts_train["start"] == ts_test["start"]

            train_values = ts_train["target"]
            test_values = ts_test["target"]

            assert len(test_values) - len(train_values) == prediction_length
            assert length_low <= len(test_values) <= length_high

            if is_integer:
                assert (
                    np.array(train_values).dtype
                    == np.array(test_values).dtype
                    == np.int
                )

            assert np.all(min_val - _eps <= train_values)
            assert np.all(train_values <= max_val + _eps)
            assert np.all(min_val - _eps <= test_values)
            assert np.all(test_values <= max_val + _eps)

            assert np.allclose(train_values, test_values[: len(train_values)])
