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
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.dataset.pandas import PandasDataset


class HierarchicalTimeSeries:
    r"""
    Class for representing hierarchical time series.

    The hierarchy is represented by the standard aggregation matrix `S`.
    The time series at the bottom (leaf) level of the hierarchy
    (`ts_at_bottom_level`) are assumed to be given by the columns of
    a single pandas dataframe.

    The ordering of columns of `ts_at_bottom_level` should be consistent
    with the ordering of the columns of `S`.

    Parameters
    ----------
    ts_at_bottom_level
        A single pandas dataframe whose columns are the time series
        corresponding to the leaves of the hierarchy.
    S
        Summation or aggregation matrix whose ordering should be consistent
        with the orderig of the columns of `ts_at_all_levels`.
        In particular, the bottom `k x k` sub-matrix should be identity matrix,
        where `k` is the number of leaves of the hierarchy.
    """

    def __init__(
        self,
        ts_at_bottom_level: pd.DataFrame,
        S: np.ndarray,
    ):
        assert isinstance(ts_at_bottom_level.index, pd.PeriodIndex), (
            "Index of `ts_at_bottom_level` must be an instance of "
            "`pd.PeriodIndex`."
        )

        self._freq = ts_at_bottom_level.index.freqstr

        self._S = S
        self.ts_at_bottom_level = ts_at_bottom_level

        self.ts_aggregated = HierarchicalTimeSeries.aggregate_ts(
            ts_at_bottom_level=self.ts_at_bottom_level,
            S=self._S,
        )

        self._ts_at_all_levels = pd.concat(
            [self.ts_aggregated, self.ts_at_bottom_level],
            axis=1,
        )
        self._ts_at_all_levels.columns = list(range(self.num_ts))

    @property
    def freq(self):
        return self._freq

    @property
    def ts_at_all_levels(self):
        return self._ts_at_all_levels

    @property
    def S(self):
        return self._S

    @property
    def num_ts(self):
        return self._S.shape[0]

    @property
    def num_bottom_ts(self):
        return self._S.shape[1]

    @staticmethod
    def aggregate_ts(
        ts_at_bottom_level: pd.DataFrame,
        S: np.ndarray,
    ) -> pd.DataFrame:
        """
        Constructs aggregated time series according to the
        summation/aggregation matrix `S`.

        Parameters
        ----------
        ts_at_bottom_level
            A single pandas dataframe whose columns are the time series
            corresponding to the leaves of the hierarchy.
        S
            Summation or aggregation matrix whose ordering should be consistent
            with the orderig of the columns of `ts_at_all_levels`.
            In particular, the bottom `k x k` sub-matrix should be an identity
            matrix, where `k` is the number of leaves of the hierarchy.

        Returns
        -------
            A pandas dataframe consisting of aggregated time series
            (at all non-leaf levels).

        """
        num_ts, num_bottom_ts = S.shape
        num_agg_ts = num_ts - num_bottom_ts

        assert ts_at_bottom_level.shape[1] == num_bottom_ts, (
            "Number of columns of the aggregation matrix `S` and "
            "the dataframe `ts_at_bottom_level` should be  same."
            f"But shape of `S`: {S.shape} and shape of `ts_at_bottom_level`: "
            f"{ts_at_bottom_level.shape}."
        )

        # Last `num_bottom_ts` rows contain the identity marix.
        assert (S[num_agg_ts:, :] == np.eye(num_bottom_ts)).all(), (
            f"The last {num_bottom_ts} rows of aggregation matrix `S`"
            f" should contain Identity matrix."
        )

        # First `num_agg_ts` rows contain the aggregation information.
        S_sum = S[:num_agg_ts, :]

        # Construct aggregated time series.
        ts_aggregated = pd.concat(
            {
                f"agg_ts_{i}": ts_at_bottom_level.apply(
                    lambda row: np.dot(row, agg),
                    axis=1,
                )
                for i, agg in enumerate(S_sum)
            },
            axis=1,
        )
        ts_aggregated.set_index(ts_at_bottom_level.index, inplace=True)

        return ts_aggregated

    def to_dataset(
        self,
        feat_dynamic_real: Optional[pd.DataFrame] = None,
    ):
        """
        Convert the hierarchical time series into
        `gluonts.dataset.PandasDataset`.

        Note: Currently only dynamic real features are used by the hierarchical
        model. However, the model internally creates a categorical feature
        to distinguish between different time series of the hierarchy.

        Parameters
        ----------
        feat_dynamic_real
            A pandas dataframe containing dynamic features as columns.
            Note that features of any (or all) time series in the hierachy
            can be passed here, since all time series are considered together
            as a single multivariate time series.

        Returns
        -------
        PandasDataset
            An instance of `PandasDataset`.

        """
        ignore_last_n_targets = 0

        if feat_dynamic_real is not None:
            assert (
                self.ts_at_all_levels.index[0] == feat_dynamic_real.index[0]
            ), (
                "The staring time point of dynamic features should match "
                "with that of the hierarhical time series. "
                f"Start of `feat_dynamic_real`: "
                f"{feat_dynamic_real.index[0]} and "
                f"the start of hierarchical time series: "
                f"{self.ts_at_all_levels.index[0]}."
            )

            assert feat_dynamic_real.index.intersection(
                self.ts_at_all_levels.index
            ).equals(self.ts_at_all_levels.index), (
                "Dynamic features should be provied for all time "
                "points where the target is defined. "
                f"Index of `feat_dynamic_real`: {feat_dynamic_real.index}, \n"
                f"Index of `ts_at_all_levels` of `hts`: "
                f"{self.ts_at_all_levels.index}. \n "
                "Check if the periods of these indices also match. \n"
            )

            feat_dynamic_real.columns = [
                f"feat_dynamic_real_{col}" for col in feat_dynamic_real.columns
            ]
            ignore_last_n_targets = len(feat_dynamic_real.index) - len(
                self.ts_at_all_levels.index
            )
        else:
            feat_dynamic_real = pd.DataFrame()

        pandas_ds = PandasDataset(
            dataframes=pd.concat(
                [self.ts_at_all_levels, feat_dynamic_real],
                axis=1,
            ),
            target=list(self.ts_at_all_levels.columns),
            feat_dynamic_real=list(feat_dynamic_real.columns),
            ignore_last_n_targets=ignore_last_n_targets,
        )

        return pandas_ds
