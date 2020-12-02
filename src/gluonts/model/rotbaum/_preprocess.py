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

import logging
from enum import Enum
from itertools import chain, starmap
from typing import Dict, List, Tuple, Union

import numpy as np

from gluonts.core.component import validated


class CardinalityLabel(str, Enum):
    auto = "auto"
    ignore = "ignore"


Cardinality = Union[List[int], CardinalityLabel]


class PreprocessGeneric:
    """
    Class for the purpose of preprocessing time series. The method
    make_features needs to be custom-made by inherited classes.
    """

    @validated()
    def __init__(
        self,
        context_window_size: int,
        forecast_horizon: int = 1,
        stratify_targets: bool = False,
        n_ignore_last: int = 0,
        max_n_datapts: int = 400000,
        **kwargs
    ):
        """
        Parameters
        ----------
        context_window_size: int
        forecast_horizon: int
        stratify_targets: bool
            If False, each context window translates to one data point in
            feature_data of length the number of features, and one
            datapoint in target_data of length the forecast horizon.
            horizon.
            If True, each context window translates to forecast_horizon
            many datapoints. The resulting datapoints in feature_data are
            of length the number of features plus one, where the last
            coordinate varies between 0 and forecast_horizon - 1, and the
            other coordinates fixed. The respective datapoints in
            target_data are all of length 1 and vary between the first to
            the last targets in the forecast horizon. (Hence the name,
            this stratifies the targets.)
        n_ignore_last: int
            Cut the last n_ignore_last steps of the time series.
        max_n_datapts: int
            Maximal number of context windows to sample from the entire
            dataset.
        """
        assert not (stratify_targets and (forecast_horizon == 1))
        self.context_window_size = context_window_size
        self.forecast_horizon = forecast_horizon
        self.stratify_targets = stratify_targets
        self.n_ignore_last = n_ignore_last
        self.max_n_datapts = max_n_datapts
        self.kwargs = kwargs
        self.num_samples = None
        self.feature_data = None
        self.target_data = None

    def make_features(self, time_series, starting_index):
        """
        Makes features for the context window starting at starting_index.

        Parameters
        ----------
        time_series: list
        starting_index: int
            The index where the context window begins

        Returns
        -------
        list
        """
        raise NotImplementedError()

    def preprocess_from_single_ts(self, time_series: Dict) -> Tuple:
        """
        Takes a single time series, ts_list, and returns preprocessed data.

        Note that the number of features is determined by the implementation
        of make_features. The number of context windows is determined by
        num_samples, see documentation under Parameters.

        If stratify_targets is False, then the length of feature_data is:
        (number of context windows) x (number of features)
        And the length of target_data is:
        (number of context windows) x (forecast_horizon)

        If stratify_targets is False, then the length of feature_data is:
        (number of context windows) * forecast_horizon x (number of features+1)
        And the length of target_data is:
        (number of context windows) * forecast_horizon x 1

        Parameters
        ----------
        time_series: dict
            has 'target' and 'start' keys

        Returns
        -------
        tuple
            list of feature datapoints, list of target datapoints
        """
        altered_time_series = time_series.copy()
        if self.n_ignore_last > 0:
            altered_time_series["target"] = altered_time_series["target"][
                : -self.n_ignore_last
            ]
        feature_data = []
        target_data = []
        max_num_context_windows = (
            len(altered_time_series["target"])
            - self.context_window_size
            - self.forecast_horizon
            + 1
        )
        if max_num_context_windows < 1:
            if not self.use_feat_static_real and not self.cardinality:
                return [], []
            else:
                # will return featurized data containing no target
                return (
                    self.make_features(
                        altered_time_series,
                        len(altered_time_series["target"]),
                    ),
                    [],
                )

        if self.num_samples > 0:
            locations = [
                np.random.randint(max_num_context_windows)
                for _ in range(self.num_samples)
            ]
        else:
            locations = range(max_num_context_windows)
        for starting_index in locations:
            if self.stratify_targets:
                featurized_data = self.make_features(
                    altered_time_series, starting_index
                )
                for forecast_horizon_index in range(self.forecast_horizon):
                    feature_data.append(
                        list(featurized_data) + [forecast_horizon_index]
                    )
                    target_data.append(
                        [
                            time_series["target"][
                                starting_index
                                + self.context_window_size
                                + forecast_horizon_index
                            ]
                        ]
                    )
            else:
                featurized_data = self.make_features(
                    altered_time_series, starting_index
                )
                feature_data.append(featurized_data)
                target_data.append(
                    time_series["target"][
                        starting_index
                        + self.context_window_size : starting_index
                        + self.context_window_size
                        + self.forecast_horizon
                    ]
                )
        return feature_data, target_data

    def preprocess_from_list(
        self, ts_list, change_internal_variables: bool = True
    ) -> Tuple:
        """
        Applies self.preprocess_from_single_ts for each time series in ts_list,
        and collates the results into self.feature_data and self.target_data

        Parameters
        ----------
        ts_list: list
            List of time series, each a dict with 'target' and 'start' keys.
        change_internal_variables: bool
            If True, keep results in self.feature_data, self.target_data and
            return None.

        Returns
        -------
        tuple
            If change_internal_variables is False, then returns:
            list of feature datapoints, list of target datapoints
        """
        feature_data, target_data = [], []
        self.num_samples = self.get_num_samples(ts_list)

        if isinstance(self.cardinality, str):
            self.cardinality = (
                self.infer_cardinalities(ts_list)
                if self.cardinality == "auto"
                else []
            )

        for time_series in ts_list:
            ts_feature_data, ts_target_data = self.preprocess_from_single_ts(
                time_series=time_series
            )
            feature_data += list(ts_feature_data)
            target_data += list(ts_target_data)
        logging.info(
            "Done preprocessing. Resulting number of datapoints is: {}".format(
                len(feature_data)
            )
        )
        if change_internal_variables:
            self.feature_data, self.target_data = feature_data, target_data
        else:
            return feature_data, target_data

    def get_num_samples(self, ts_list) -> int:
        """
        Outputs a reasonable choice for number of windows to sample from
        each time series at training time.
        """
        n_time_series = sum(
            [
                len(time_series["target"])
                - self.context_window_size
                - self.forecast_horizon
                >= 0
                for time_series in ts_list
            ]
        )
        max_size_ts = max(
            [len(time_series["target"]) for time_series in ts_list]
        )
        n_windows_per_time_series = self.max_n_datapts // n_time_series
        if n_time_series * 1000 < n_windows_per_time_series:
            n_windows_per_time_series = n_time_series * 1000
        elif n_windows_per_time_series == 0:
            n_windows_per_time_series = 1
        elif n_windows_per_time_series > max_size_ts:
            n_windows_per_time_series = -1
        return n_windows_per_time_series

    def infer_cardinalities(self):
        raise NotImplementedError


class PreprocessOnlyLagFeatures(PreprocessGeneric):
    def __init__(
        self,
        context_window_size,
        forecast_horizon=1,
        stratify_targets=False,
        n_ignore_last=0,
        num_samples=-1,
        use_feat_static_real=False,
        use_feat_dynamic_real=False,
        use_feat_dynamic_cat=False,
        cardinality: Cardinality = CardinalityLabel.auto,
        one_hot_encode: bool = True,  # Should improve accuracy but will slow down model
        **kwargs
    ):

        if one_hot_encode:
            assert cardinality != "ignore" or (
                isinstance(cardinality, List)
                and all(c > 0 for c in cardinality)
            ), "You should set `one_hot_encode=True` if and only if cardinality is a valid list or not ignored: {}"

        super().__init__(
            context_window_size=context_window_size,
            forecast_horizon=forecast_horizon,
            stratify_targets=stratify_targets,
            n_ignore_last=n_ignore_last,
            num_samples=num_samples,
            **kwargs
        )

        self.use_feat_static_real = use_feat_static_real
        self.cardinality = cardinality
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_dynamic_cat = use_feat_dynamic_cat
        self.one_hot_encode = one_hot_encode

    @classmethod
    def _pre_transform(cls, time_series_window) -> Tuple:
        """
        Makes features given time series window. Returns list of features,
        one for every step of the lag (equaling mean-adjusted lag features);
        and a dictionary of statistics features (one for mean and one for
        standard deviation).

        Parameters
        ----------
        time_series_window: list

        Returns
        -------------
        tuple
            trasnformed time series, dictionary with transformation data
        return (time_series_window - np.mean(time_series_window)), {
            'mean': np.mean(time_series_window),
            'std': np.std(time_series_window)
        }
        """
        mean_value = np.mean(time_series_window)
        return (
            (time_series_window - mean_value),
            {
                "mean": mean_value,
                "std": np.std(time_series_window),
                "n_lag_features": len(time_series_window),
            },
        )

    def encode_one_hot(self, feat: int, cardinality: int) -> List[int]:
        result = [0] * cardinality
        result[feat] = 1
        return result

    def encode_one_hot_all(self, feat_list: List):
        # asserts that the categorical features are label encoded
        np_feat_list = np.array(feat_list)
        assert all(np.floor(np_feat_list) == np_feat_list)

        encoded = starmap(
            self.encode_one_hot, zip(feat_list, self.cardinality)
        )
        encoded_chain = chain.from_iterable(encoded)
        return list(encoded_chain)

    def infer_cardinalities(self, time_series):
        if "feat_static_cat" not in time_series[0]:
            return []
        mat = np.array(
            [elem["feat_static_cat"] for elem in time_series], dtype=int
        )
        return [len(set(xs)) for xs in mat.T]

    def make_features(self, time_series: Dict, starting_index: int) -> List:
        """
        Makes features for the context window starting at starting_index.

        Parameters
        ----------
        time_series: dict
            has 'target' and 'start' keys
        starting_index: int
            The index where the context window begins

        Returns
        -------
        list
        """
        end_index = starting_index + self.context_window_size
        if starting_index < 0:
            prefix = [None] * abs(starting_index)
        else:
            prefix = []
        time_series_window = time_series["target"][starting_index:end_index]
        only_lag_features, transform_dict = self._pre_transform(
            time_series_window
        )

        feat_static_real = (
            list(time_series["feat_static_real"])
            if self.use_feat_static_real
            else []
        )
        if self.cardinality:
            feat_static_cat = (
                self.encode_one_hot_all(time_series["feat_static_cat"])
                if self.one_hot_encode
                else list(time_series["feat_static_cat"])
            )
        else:
            feat_static_cat = []

        feat_dynamic_real = (
            [elem for ent in time_series["feat_dynamic_real"] for elem in ent]
            if self.use_feat_dynamic_real
            else []
        )
        feat_dynamic_cat = (
            [elem for ent in time_series["feat_dynamic_cat"] for elem in ent]
            if self.use_feat_dynamic_cat
            else []
        )

        # these two assertions check that the categorical features are encoded
        np_feat_static_cat = np.array(feat_static_cat)
        assert (not feat_static_cat) or all(
            np.floor(np_feat_static_cat) == np_feat_static_cat
        )

        np_feat_dynamic_cat = np.array(feat_dynamic_cat)
        assert (not feat_dynamic_cat) or all(
            np.floor(np_feat_dynamic_cat) == np_feat_dynamic_cat
        )

        feat_dynamics = feat_dynamic_real + feat_dynamic_cat
        feat_statics = feat_static_real + feat_static_cat
        only_lag_features = list(only_lag_features)
        return (
            prefix
            + only_lag_features
            + list(transform_dict.values())
            + feat_statics
            + feat_dynamics
        )
