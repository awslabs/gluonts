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

from gluonts.core import serde
from gluonts.dataset.common import ListDataset
from gluonts.ext.prophet import ProphetPredictor


@pytest.mark.parametrize(
    "freq",
    [
        "1H",
        "2D",
        "3W",
        "4M",
    ],
)
def test_feat_dynamic_real_success(freq: str):
    params = dict(prediction_length=3, prophet_params=dict(n_changepoints=20))

    dataset = ListDataset(
        data_iter=[
            {
                "start": "2017-01-01",
                "target": np.array([1.0, 2.0, 3.0, 4.0]),
                "feat_dynamic_real": np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    ]
                ),
            }
        ],
        freq=freq,
    )

    predictor = ProphetPredictor(**params)

    act_fcst = next(predictor.predict(dataset))
    exp_fcst = np.arange(5.0, 5.0 + params["prediction_length"])

    assert exp_fcst.shape == act_fcst.quantile(0.1).shape
    assert exp_fcst.shape == act_fcst.quantile(0.5).shape
    assert exp_fcst.shape == act_fcst.quantile(0.9).shape


def test_feat_dynamic_real_bad_size():
    params = dict(prediction_length=3, prophet_params={})

    dataset = ListDataset(
        data_iter=[
            {
                "start": "2017-01-01",
                "target": np.array([1.0, 2.0, 3.0, 4.0]),
                "feat_dynamic_real": np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ]
                ),
            }
        ],
        freq="1D",
    )

    with pytest.raises(AssertionError) as excinfo:
        predictor = ProphetPredictor(**params)
        list(predictor.predict(dataset))

    assert str(excinfo.value) == (
        "Length mismatch for dynamic real-valued feature #0: "
        "expected 7, got 6"
    )


def test_min_obs_error():
    params = dict(prediction_length=10, prophet_params={})

    dataset = ListDataset(
        data_iter=[{"start": "2017-01-01", "target": np.array([1.0])}],
        freq="1D",
    )

    with pytest.raises(ValueError) as excinfo:
        predictor = ProphetPredictor(**params)
        list(predictor.predict(dataset))

    act_error_msg = str(excinfo.value)
    exp_error_msg = "Dataframe has less than 2 non-NaN rows."

    assert act_error_msg == exp_error_msg


def test_prophet_serialization():
    predictor = ProphetPredictor(prediction_length=3)
    assert predictor == serde.decode(serde.encode(predictor))
