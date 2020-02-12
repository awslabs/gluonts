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

# Standard library imports
from typing import List, Tuple

# Third-party imports
from pandas.tseries.frequencies import to_offset

# First-party imports
from gluonts.core.component import validated
from gluonts.distribution.distribution import getF
from gluonts.model.common import Tensor
from gluonts.support.util import _broadcast_param
from gluonts.time_feature import (
    TimeFeature,
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    WeekOfYear,
    MonthOfYear,
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


class ISSM:
    r"""
    An abstract class for providing the basic structure of Innovation State Space Model (ISSM).

    The structure of ISSM is given by

        * dimension of the latent state
        * transition and emission coefficents of the transition model
        * emission coefficient of the observation model

    """

    @validated()
    def __init__(self):
        pass

    def latent_dim(self) -> int:
        raise NotImplemented()

    def output_dim(self) -> int:
        raise NotImplemented()

    def emission_coeff(self, seasonal_indicators: Tensor):
        raise NotImplemented()

    def transition_coeff(self, seasonal_indicators: Tensor):
        raise NotImplemented()

    def innovation_coeff(self, seasonal_indicators: Tensor):
        raise NotImplemented()

    def get_issm_coeff(
        self, seasonal_indicators: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.emission_coeff(seasonal_indicators),
            self.transition_coeff(seasonal_indicators),
            self.innovation_coeff(seasonal_indicators),
        )


class LevelISSM(ISSM):
    def latent_dim(self) -> int:
        return 1

    def output_dim(self) -> int:
        return 1

    def emission_coeff(
        self, seasonal_indicators: Tensor  # (batch_size, time_length)
    ) -> Tensor:
        F = getF(seasonal_indicators)

        _emission_coeff = F.ones(shape=(1, 1, 1, self.latent_dim()))

        # get the right shape: (batch_size, seq_length, obs_dim, latent_dim)
        zeros = _broadcast_param(
            F.zeros_like(
                seasonal_indicators.slice_axis(
                    axis=-1, begin=0, end=1
                ).squeeze(axis=-1)
            ),
            axes=[2, 3],
            sizes=[1, self.latent_dim()],
        )

        return _emission_coeff.broadcast_like(zeros)

    def transition_coeff(
        self, seasonal_indicators: Tensor  # (batch_size, time_length)
    ) -> Tensor:
        F = getF(seasonal_indicators)

        _transition_coeff = (
            F.eye(self.latent_dim()).expand_dims(axis=0).expand_dims(axis=0)
        )

        # get the right shape: (batch_size, seq_length, latent_dim, latent_dim)
        zeros = _broadcast_param(
            F.zeros_like(
                seasonal_indicators.slice_axis(
                    axis=-1, begin=0, end=1
                ).squeeze(axis=-1)
            ),
            axes=[2, 3],
            sizes=[self.latent_dim(), self.latent_dim()],
        )

        return _transition_coeff.broadcast_like(zeros)

    def innovation_coeff(
        self, seasonal_indicators: Tensor  # (batch_size, time_length)
    ) -> Tensor:
        return self.emission_coeff(seasonal_indicators).squeeze(axis=2)


class LevelTrendISSM(LevelISSM):
    def latent_dim(self) -> int:
        return 2

    def output_dim(self) -> int:
        return 1

    def transition_coeff(
        self, seasonal_indicators: Tensor  # (batch_size, time_length)
    ) -> Tensor:
        F = getF(seasonal_indicators)

        _transition_coeff = (
            (F.diag(F.ones(shape=(2,)), k=0) + F.diag(F.ones(shape=(1,)), k=1))
            .expand_dims(axis=0)
            .expand_dims(axis=0)
        )

        # get the right shape: (batch_size, seq_length, latent_dim, latent_dim)
        zeros = _broadcast_param(
            F.zeros_like(
                seasonal_indicators.slice_axis(
                    axis=-1, begin=0, end=1
                ).squeeze(axis=-1)
            ),
            axes=[2, 3],
            sizes=[self.latent_dim(), self.latent_dim()],
        )

        return _transition_coeff.broadcast_like(zeros)


class SeasonalityISSM(LevelISSM):
    """
    Implements periodic seasonality which is entirely determined by the period `num_seasons`.
    """

    @validated()
    def __init__(self, num_seasons: int) -> None:
        super(SeasonalityISSM, self).__init__()
        self.num_seasons = num_seasons

    def latent_dim(self) -> int:
        return self.num_seasons

    def output_dim(self) -> int:
        return 1

    def emission_coeff(self, seasonal_indicators: Tensor) -> Tensor:
        F = getF(seasonal_indicators)
        return F.one_hot(seasonal_indicators, depth=self.latent_dim())

    def innovation_coeff(self, seasonal_indicators: Tensor) -> Tensor:
        F = getF(seasonal_indicators)
        # seasonal_indicators = F.modulo(seasonal_indicators - 1, self.latent_dim)
        return F.one_hot(seasonal_indicators, depth=self.latent_dim()).squeeze(
            axis=2
        )


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

    @classmethod
    def get_from_freq(cls, freq: str, add_trend: bool = DEFAULT_ADD_TREND):
        offset = to_offset(freq)

        seasonal_issms: List[SeasonalityISSM] = []

        if offset.name == "M":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=12)  # month-of-year seasonality
            ]
        elif offset.name == "W-SUN":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=53)  # week-of-year seasonality
            ]
        elif offset.name == "D":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=7)  # day-of-week seasonality
            ]
        elif offset.name == "B":  # TODO: check this case
            seasonal_issms = [
                SeasonalityISSM(num_seasons=7)  # day-of-week seasonality
            ]
        elif offset.name == "H":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
                SeasonalityISSM(num_seasons=7),  # day-of-week seasonality
            ]
        elif offset.name == "T":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=60),  # minute-of-hour seasonality
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return cls(seasonal_issms=seasonal_issms, add_trend=add_trend)

    @classmethod
    def seasonal_features(cls, freq: str) -> List[TimeFeature]:
        offset = to_offset(freq)
        if offset.name == "M":
            return [MonthOfYear(normalized=False)]
        elif offset.name == "W-SUN":
            return [WeekOfYear(normalized=False)]
        elif offset.name == "D":
            return [DayOfWeek(normalized=False)]
        elif offset.name == "B":  # TODO: check this case
            return [DayOfWeek(normalized=False)]
        elif offset.name == "H":
            return [HourOfDay(normalized=False), DayOfWeek(normalized=False)]
        elif offset.name == "T":
            return [
                MinuteOfHour(normalized=False),
                HourOfDay(normalized=False),
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return []

    def get_issm_coeff(
        self, seasonal_indicators: Tensor  # (batch_size, time_length)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        F = getF(seasonal_indicators)
        emission_coeff_ls, transition_coeff_ls, innovation_coeff_ls = zip(
            self.nonseasonal_issm.get_issm_coeff(seasonal_indicators),
            *[
                issm.get_issm_coeff(
                    seasonal_indicators.slice_axis(
                        axis=-1, begin=ix, end=ix + 1
                    )
                )
                for ix, issm in enumerate(self.seasonal_issms)
            ],
        )

        # stack emission and innovation coefficients
        emission_coeff = F.concat(*emission_coeff_ls, dim=-1)

        innovation_coeff = F.concat(*innovation_coeff_ls, dim=-1)

        # transition coefficient is block diagonal!
        transition_coeff = _make_block_diagonal(transition_coeff_ls)

        return emission_coeff, transition_coeff, innovation_coeff
