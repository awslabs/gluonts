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

import mxnet as mx
import numpy as np
import pandas as pd

from gluonts.model.deepstate.issm import (
    CompositeISSM,
    LevelISSM,
    LevelTrendISSM,
    SeasonalityISSM,
    ZeroFeature,
    _make_block_diagonal,
)
from gluonts.time_feature import DayOfWeekIndex, MonthOfYearIndex


def test_zero_feature():
    zf = ZeroFeature()
    feature = zf(pd.date_range("2020-01-01 21:00:00", periods=10, freq="H"))

    assert isinstance(feature, np.ndarray)
    assert feature.shape == (10,)
    assert feature.dtype == np.float


def test_level_issm_h():
    issm = LevelISSM()

    assert issm.latent_dim() == 1
    assert issm.output_dim() == 1

    time_features = issm.time_features()

    assert len(time_features) == 1

    time_indices = [
        pd.date_range("2020-01-01 21:00:00", periods=10, freq="H"),
        pd.date_range("2020-01-31 22:00:00", periods=10, freq="H"),
    ]

    features = mx.nd.array(
        np.stack(
            [
                np.stack([f(time_index) for f in time_features], axis=-1)
                for time_index in time_indices
            ],
            axis=0,
        )
    )

    emission_coeff, transition_coeff, innovation_coeff = issm.get_issm_coeff(
        features
    )

    assert (
        (
            emission_coeff
            == mx.nd.ones((2, 10, issm.output_dim(), issm.latent_dim()))
        )
        .asnumpy()
        .all()
    )

    assert (
        (
            transition_coeff
            == mx.nd.ones((2, 10, issm.latent_dim(), issm.latent_dim()))
        )
        .asnumpy()
        .all()
    )

    assert (
        (innovation_coeff == mx.nd.ones((2, 10, issm.latent_dim())))
        .asnumpy()
        .all()
    )


def test_level_trend_issm_h():
    issm = LevelTrendISSM()

    assert issm.latent_dim() == 2
    assert issm.output_dim() == 1

    time_features = issm.time_features()

    assert len(time_features) == 1

    time_indices = [
        pd.date_range("2020-01-01 21:00:00", periods=10, freq="H"),
        pd.date_range("2020-01-31 22:00:00", periods=10, freq="H"),
    ]

    features = mx.nd.array(
        np.stack(
            [
                np.stack([f(time_index) for f in time_features], axis=-1)
                for time_index in time_indices
            ],
            axis=0,
        )
    )

    emission_coeff, transition_coeff, innovation_coeff = issm.get_issm_coeff(
        features
    )

    assert (
        (
            emission_coeff
            == mx.nd.ones((2, 10, issm.output_dim(), issm.latent_dim()))
        )
        .asnumpy()
        .all()
    )

    for item in range(2):
        for time in range(10):
            sliced_transition_coeff = mx.nd.slice(
                transition_coeff,
                begin=(item, time, None, None),
                end=(item + 1, time + 1, None, None),
            )
            assert (
                (
                    sliced_transition_coeff
                    == mx.nd.array([[1.0, 1.0], [0.0, 1.0]])
                )
                .asnumpy()
                .all()
            )

    assert (
        (innovation_coeff == mx.nd.ones((2, 10, issm.latent_dim())))
        .asnumpy()
        .all()
    )


def test_seasonality_issm_h():
    issm = SeasonalityISSM(num_seasons=12, time_feature=MonthOfYearIndex())

    assert issm.latent_dim() == 12
    assert issm.output_dim() == 1

    time_features = issm.time_features()

    assert len(time_features) == 1

    time_indices = [
        pd.date_range("2019-01-01 21:00:00", periods=10, freq="H"),
        pd.date_range("2019-01-31 22:00:00", periods=10, freq="H"),
    ]

    features = mx.nd.array(
        np.stack(
            [
                np.stack([f(time_index) for f in time_features], axis=-1)
                for time_index in time_indices
            ],
            axis=0,
        )
    )

    emission_coeff, transition_coeff, innovation_coeff = issm.get_issm_coeff(
        features
    )

    season_indices = [[0] * 10, [0] * 2 + [1] * 8]

    for item in range(2):
        for time in range(10):
            season_indicator = mx.nd.one_hot(
                mx.nd.array([season_indices[item][time]]), 12
            )

            sliced_emission_coeff = mx.nd.slice(
                emission_coeff,
                begin=(item, time, None, None),
                end=(item + 1, time + 1, None, None),
            )
            assert (sliced_emission_coeff == season_indicator).asnumpy().all()

            sliced_transition_coeff = mx.nd.slice(
                transition_coeff,
                begin=(item, time, None, None),
                end=(item + 1, time + 1, None, None),
            )
            assert (sliced_transition_coeff == mx.nd.eye(12)).asnumpy().all()

            sliced_innovation_coeff = mx.nd.slice(
                innovation_coeff,
                begin=(item, time, None),
                end=(item + 1, time + 1, None),
            )
            assert (
                (sliced_innovation_coeff == season_indicator).asnumpy().all()
            )


def test_composite_issm_h():
    issm = CompositeISSM(
        seasonal_issms=[
            SeasonalityISSM(num_seasons=7, time_feature=DayOfWeekIndex()),
            SeasonalityISSM(num_seasons=12, time_feature=MonthOfYearIndex()),
        ],
        add_trend=False,
    )

    assert issm.latent_dim() == 1 + 7 + 12
    assert issm.output_dim() == 1

    time_features = issm.time_features()

    assert len(time_features) == 3

    time_indices = [
        pd.date_range("2020-01-01 21:00:00", periods=10, freq="H"),
        pd.date_range("2020-01-31 22:00:00", periods=10, freq="H"),
    ]

    features = mx.nd.array(
        np.stack(
            [
                np.stack([f(time_index) for f in time_features], axis=-1)
                for time_index in time_indices
            ],
            axis=0,
        )
    )

    emission_coeff, transition_coeff, innovation_coeff = issm.get_issm_coeff(
        features
    )

    season_indices_dow = [[2] * 3 + [3] * 7, [4] * 2 + [5] * 8]

    season_indices_moy = [[0] * 10, [0] * 2 + [1] * 8]

    for item in range(2):
        for time in range(10):
            expected_coeff = mx.nd.concat(
                mx.nd.ones((1, 1)),
                mx.nd.one_hot(
                    mx.nd.array([season_indices_dow[item][time]]), 7
                ),
                mx.nd.one_hot(
                    mx.nd.array([season_indices_moy[item][time]]), 12
                ),
                dim=-1,
            )

            sliced_emission_coeff = mx.nd.slice(
                emission_coeff,
                begin=(item, time, None, None),
                end=(item + 1, time + 1, None, None),
            )
            assert (sliced_emission_coeff == expected_coeff).asnumpy().all()

            sliced_transition_coeff = mx.nd.slice(
                transition_coeff,
                begin=(item, time, None, None),
                end=(item + 1, time + 1, None, None),
            )
            assert (
                (sliced_transition_coeff == mx.nd.eye(1 + 7 + 12))
                .asnumpy()
                .all()
            )

            sliced_innovation_coeff = mx.nd.slice(
                innovation_coeff,
                begin=(item, time, None),
                end=(item + 1, time + 1, None),
            )
            assert (sliced_innovation_coeff == expected_coeff).asnumpy().all()


def test_composite_issm_h_default():
    issm = CompositeISSM.get_from_freq(freq="H")

    assert issm.latent_dim() == 2 + 24 + 7
    assert issm.output_dim() == 1

    time_features = issm.time_features()

    assert len(time_features) == 3

    time_indices = [
        pd.date_range("2020-01-01 21:00:00", periods=10, freq="H"),
        pd.date_range("2020-01-02 03:00:00", periods=10, freq="H"),
    ]

    features = mx.nd.array(
        np.stack(
            [
                np.stack([f(time_index) for f in time_features], axis=-1)
                for time_index in time_indices
            ],
            axis=0,
        )
    )

    emission_coeff, transition_coeff, innovation_coeff = issm.get_issm_coeff(
        features
    )

    season_indices_hod = [
        [21, 22, 23, 0, 1, 2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]

    season_indices_dow = [[2] * 3 + [3] * 7, [3] * 10]

    for item in range(2):
        for time in range(10):
            expected_coeff = mx.nd.concat(
                mx.nd.ones((1, 2)),
                mx.nd.one_hot(
                    mx.nd.array([season_indices_hod[item][time]]), 24
                ),
                mx.nd.one_hot(
                    mx.nd.array([season_indices_dow[item][time]]), 7
                ),
                dim=-1,
            )

            sliced_emission_coeff = mx.nd.slice(
                emission_coeff,
                begin=(item, time, None, None),
                end=(item + 1, time + 1, None, None),
            )
            assert (sliced_emission_coeff == expected_coeff).asnumpy().all()

            sliced_transition_coeff = mx.nd.slice(
                transition_coeff,
                begin=(item, time, None, None),
                end=(item + 1, time + 1, None, None),
            )
            expected_transition_coeff = mx.nd.eye(2 + 24 + 7)
            expected_transition_coeff[0, 1] = 1
            assert (
                (sliced_transition_coeff == expected_transition_coeff)
                .asnumpy()
                .all()
            )

            sliced_innovation_coeff = mx.nd.slice(
                innovation_coeff,
                begin=(item, time, None),
                end=(item + 1, time + 1, None),
            )
            assert (sliced_innovation_coeff == expected_coeff).asnumpy().all()
