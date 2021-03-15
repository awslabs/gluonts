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

from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.distribution.distribution import getF
from gluonts.mx.util import _broadcast_param
from gluonts.time_feature import (
    DayOfWeekIndex,
    HourOfDayIndex,
    MinuteOfHourIndex,
    MonthOfYearIndex,
    TimeFeature,
    WeekOfYearIndex,
    norm_freq_str,
)


def _make_block_diagonal(blocks: List[Tensor]) -> Tensor:
    assert (
        len(blocks) > 0
    ), "You need at least one tensor to make a block-diagonal tensor"

    if len(blocks) == 1:
        return blocks[0]

    F = getF(blocks[0])

    # transition coefficient is block diagonal!
    block_diagonal = _make_2_block_diagonal(F, blocks[0], blocks[1])
    for i in range(2, len(blocks)):
        block_diagonal = _make_2_block_diagonal(
            F=F, left=block_diagonal, right=blocks[i]
        )

    return block_diagonal


def _make_2_block_diagonal(F, left: Tensor, right: Tensor) -> Tensor:
    """
    Creates a block diagonal matrix of shape (batch_size, m+n, m+n) where m and n are the sizes of
    the axis 1 of left and right respectively.

    Parameters
    ----------
    F
    left
        Tensor of shape (batch_size, seq_length, m, m)
    right
        Tensor of shape (batch_size, seq_length, n, n)
    Returns
    -------
    Tensor
        Block diagonal matrix of shape (batch_size, seq_length, m+n, m+n)
    """
    # shape (batch_size, seq_length, m, n)
    zeros_off_diag = F.broadcast_add(
        left.slice_axis(
            axis=-1, begin=0, end=1
        ).zeros_like(),  # shape (batch_size, seq_length, m, 1)
        right.slice_axis(
            axis=-2, begin=0, end=1
        ).zeros_like(),  # shape (batch_size, seq_length, 1, n)
    )

    # shape (batch_size, n, m)
    zeros_off_diag_tr = zeros_off_diag.swapaxes(2, 3)

    # block diagonal: shape (batch_size, seq_length, m+n, m+n)
    _block_diagonal = F.concat(
        F.concat(left, zeros_off_diag, dim=3),
        F.concat(zeros_off_diag_tr, right, dim=3),
        dim=2,
    )

    return _block_diagonal


class ZeroFeature(TimeFeature):
    """
    A feature that is identically zero.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.zeros(index.values.shape)


class ISSM:
    r"""
    An abstract class for providing the basic structure of Innovation State Space Model (ISSM).

    The structure of ISSM is given by

        * dimension of the latent state
        * transition and innovation coefficients of the transition model
        * emission coefficient of the observation model

    """

    @validated()
    def __init__(self):
        pass

    def latent_dim(self) -> int:
        raise NotImplementedError

    def output_dim(self) -> int:
        raise NotImplementedError

    def time_features(self) -> List[TimeFeature]:
        raise NotImplementedError

    def emission_coeff(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def transition_coeff(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def innovation_coeff(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def get_issm_coeff(
        self, features: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.emission_coeff(features),
            self.transition_coeff(features),
            self.innovation_coeff(features),
        )


class LevelISSM(ISSM):
    def latent_dim(self) -> int:
        return 1

    def output_dim(self) -> int:
        return 1

    def time_features(self) -> List[TimeFeature]:
        return [ZeroFeature()]

    def emission_coeff(
        self, feature: Tensor  # (batch_size, time_length, 1)
    ) -> Tensor:
        F = getF(feature)

        _emission_coeff = F.ones(shape=(1, 1, 1, self.latent_dim()))

        # get the right shape: (batch_size, time_length, obs_dim, latent_dim)
        zeros = _broadcast_param(
            feature.squeeze(axis=2),
            axes=[2, 3],
            sizes=[1, self.latent_dim()],
        )

        return _emission_coeff.broadcast_like(zeros)

    def transition_coeff(
        self, feature: Tensor  # (batch_size, time_length, 1)
    ) -> Tensor:
        F = getF(feature)

        _transition_coeff = (
            F.eye(self.latent_dim()).expand_dims(axis=0).expand_dims(axis=0)
        )

        # get the right shape: (batch_size, time_length, latent_dim, latent_dim)
        zeros = _broadcast_param(
            feature.squeeze(axis=2),
            axes=[2, 3],
            sizes=[self.latent_dim(), self.latent_dim()],
        )

        return _transition_coeff.broadcast_like(zeros)

    def innovation_coeff(
        self, feature: Tensor  # (batch_size, time_length, 1)
    ) -> Tensor:
        return self.emission_coeff(feature).squeeze(axis=2)


class LevelTrendISSM(LevelISSM):
    def latent_dim(self) -> int:
        return 2

    def output_dim(self) -> int:
        return 1

    def time_features(self) -> List[TimeFeature]:
        return [ZeroFeature()]

    def transition_coeff(
        self, feature: Tensor  # (batch_size, time_length, 1)
    ) -> Tensor:
        F = getF(feature)

        _transition_coeff = (
            (F.diag(F.ones(shape=(2,)), k=0) + F.diag(F.ones(shape=(1,)), k=1))
            .expand_dims(axis=0)
            .expand_dims(axis=0)
        )

        # get the right shape: (batch_size, time_length, latent_dim, latent_dim)
        zeros = _broadcast_param(
            feature.squeeze(axis=2),
            axes=[2, 3],
            sizes=[self.latent_dim(), self.latent_dim()],
        )

        return _transition_coeff.broadcast_like(zeros)


class SeasonalityISSM(LevelISSM):
    """
    Implements periodic seasonality which is entirely determined by the period `num_seasons`.
    """

    @validated()
    def __init__(self, num_seasons: int, time_feature: TimeFeature) -> None:
        super(SeasonalityISSM, self).__init__()
        self.num_seasons = num_seasons
        self.time_feature = time_feature

    def latent_dim(self) -> int:
        return self.num_seasons

    def output_dim(self) -> int:
        return 1

    def time_features(self) -> List[TimeFeature]:
        return [self.time_feature]

    def emission_coeff(self, feature: Tensor) -> Tensor:
        F = getF(feature)
        return F.one_hot(feature, depth=self.latent_dim())

    def innovation_coeff(self, feature: Tensor) -> Tensor:
        F = getF(feature)
        return F.one_hot(feature, depth=self.latent_dim()).squeeze(axis=2)


def MonthOfYearSeasonalISSM():
    return SeasonalityISSM(num_seasons=12, time_feature=MonthOfYearIndex())


def WeekOfYearSeasonalISSM():
    return SeasonalityISSM(num_seasons=53, time_feature=WeekOfYearIndex())


def DayOfWeekSeasonalISSM():
    return SeasonalityISSM(num_seasons=7, time_feature=DayOfWeekIndex())


def HourOfDaySeasonalISSM():
    return SeasonalityISSM(num_seasons=24, time_feature=HourOfDayIndex())


def MinuteOfHourSeasonalISSM():
    return SeasonalityISSM(num_seasons=60, time_feature=MinuteOfHourIndex())


class CompositeISSM(ISSM):
    DEFAULT_ADD_TREND: bool = True

    @validated()
    def __init__(
        self,
        seasonal_issms: List[SeasonalityISSM],
        add_trend: bool = DEFAULT_ADD_TREND,
    ) -> None:
        super(CompositeISSM, self).__init__()
        self.seasonal_issms = seasonal_issms
        self.nonseasonal_issm = (
            LevelISSM() if add_trend is False else LevelTrendISSM()
        )

    def latent_dim(self) -> int:
        return (
            sum([issm.latent_dim() for issm in self.seasonal_issms])
            + self.nonseasonal_issm.latent_dim()
        )

    def output_dim(self) -> int:
        return self.nonseasonal_issm.output_dim()

    def time_features(self) -> List[TimeFeature]:
        ans = self.nonseasonal_issm.time_features()
        for issm in self.seasonal_issms:
            ans.extend(issm.time_features())
        return ans

    @classmethod
    def get_from_freq(cls, freq: str, add_trend: bool = DEFAULT_ADD_TREND):
        offset = to_offset(freq)

        seasonal_issms: List[SeasonalityISSM] = []

        if offset.name == "M":
            seasonal_issms = [MonthOfYearSeasonalISSM()]
        elif norm_freq_str(offset.name) == "W":
            seasonal_issms = [WeekOfYearSeasonalISSM()]
        elif offset.name == "D":
            seasonal_issms = [DayOfWeekSeasonalISSM()]
        elif offset.name == "B":  # TODO: check this case
            seasonal_issms = [DayOfWeekSeasonalISSM()]
        elif offset.name == "H":
            seasonal_issms = [
                HourOfDaySeasonalISSM(),
                DayOfWeekSeasonalISSM(),
            ]
        elif offset.name == "T":
            seasonal_issms = [
                MinuteOfHourSeasonalISSM(),
                HourOfDaySeasonalISSM(),
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return cls(seasonal_issms=seasonal_issms, add_trend=add_trend)

    def get_issm_coeff(
        self, features: Tensor  # (batch_size, time_length, num_features)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        F = getF(features)
        emission_coeff_ls, transition_coeff_ls, innovation_coeff_ls = zip(
            *[
                issm.get_issm_coeff(
                    features.slice_axis(axis=-1, begin=ix, end=ix + 1)
                )
                for ix, issm in enumerate(
                    [self.nonseasonal_issm] + self.seasonal_issms
                )
            ],
        )

        # stack emission and innovation coefficients
        emission_coeff = F.concat(*emission_coeff_ls, dim=-1)

        innovation_coeff = F.concat(*innovation_coeff_ls, dim=-1)

        # transition coefficient is block diagonal!
        transition_coeff = _make_block_diagonal(transition_coeff_ls)

        return emission_coeff, transition_coeff, innovation_coeff
