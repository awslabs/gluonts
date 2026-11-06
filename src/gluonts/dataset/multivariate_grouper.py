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
from typing import Callable, Optional

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset, ListDataset
from gluonts.dataset.field_names import FieldName


class MultivariateGrouper:
    """
    The MultivariateGrouper takes a univariate dataset and groups it into a
    single multivariate time series. Therefore, this class allows the user to
    convert a univariate dataset into a multivariate dataset without making a
    separate copy of the dataset.

    The Multivariate Grouper has two different modes:

    Training: For training data, the univariate time series get aligned to the
    earliest time stamp in the dataset. Time series will be left and right
    padded to produce an array of shape (dim, num_time_steps)

    Test: The test dataset might have multiple start dates (usually because
    the test dataset mimics a rolling evaluation scenario). In this case,
    the univariate dataset will be split into n multivariate time series,
    where n is the number of evaluation dates. Again, the
    time series will be grouped but only left padded. Note that the
    padded value will influence the prediction if the context length is
    longer than the length of the time series.

    Rules for padding for training and test datasets can be specified by the
    user.

    Parameters
    ----------
    max_target_dim
        Set maximum dimensionality (for faster testing or when hitting
        constraints of multivariate model). Takes the last max_target_dim
        time series and groups them to multivariate time series.
    num_test_dates
        Number of test dates in the test set. This can be more than one if
        the test set contains more than one forecast start date (often the
        case in a rolling evaluation scenario). Must be set to convert test
        data.
    train_fill_rule
        Implements the rule that fills missing data after alignment of the
        time series for the training dataset.
    test_fill_rule
        Implements the rule that fills missing data after alignment of the
        time series for the test dataset.
    """

    @validated()
    def __init__(
        self,
        max_target_dim: Optional[int] = None,
        num_test_dates: Optional[int] = None,
        train_fill_rule: Callable = np.mean,
        test_fill_rule: Callable = lambda x: 0.0,
    ) -> None:
        self.num_test_dates = num_test_dates
        self.max_target_dimension = max_target_dim
        self.train_fill_function = train_fill_rule
        self.test_fill_rule = test_fill_rule

        self.first_timestamp = None
        self.last_timestamp = None
        self.frequency = ""

    def __call__(self, dataset: Dataset) -> Dataset:
        try:
            from gluonts.dataset.split import (
                InputDataset,
                LabelDataset,
                TestData,
                TestTemplate,
            )

            if isinstance(dataset, TestData):
                if (
                    self.num_test_dates is not None
                    and self.num_test_dates != dataset.windows
                ):
                    raise ValueError(
                        f"num_test_dates ({self.num_test_dates}) must match "
                        f"dataset.windows ({dataset.windows})"
                    )
                self._preprocess(dataset.dataset)
                grouped_base = self._prepare_train_data(dataset.dataset)
                return TestTemplate(
                    grouped_base, dataset.splitter
                ).generate_instances(
                    prediction_length=dataset.prediction_length,
                    windows=dataset.windows,
                    distance=dataset.distance,
                    max_history=dataset.max_history,
                )
            if isinstance(dataset, InputDataset):
                return self(dataset.test_data).input
            if isinstance(dataset, LabelDataset):
                return self(dataset.test_data).label
        except ImportError:
            pass

        dataset_list = list(dataset)
        if not dataset_list:
            return dataset_list

        first_entry = dataset_list[0]
        if isinstance(first_entry, (tuple, list)) and len(first_entry) == 2:
            inputs = [entry[0] for entry in dataset_list]
            labels = [entry[1] for entry in dataset_list]
            grouped_inputs = self(inputs)

            num_dates = (
                self.num_test_dates if self.num_test_dates is not None else 1
            )
            assert len(labels) % num_dates == 0
            num_series = len(labels) // num_dates

            is_series_major = False
            if num_dates > 1 and num_series > 1:
                end_0 = (
                    inputs[0][FieldName.START]
                    + len(inputs[0][FieldName.TARGET])
                    - 1
                )
                end_1 = (
                    inputs[1][FieldName.START]
                    + len(inputs[1][FieldName.TARGET])
                    - 1
                )
                if end_0 != end_1 or len(inputs[0][FieldName.TARGET]) != len(
                    inputs[1][FieldName.TARGET]
                ):
                    is_series_major = True

            all_grouped_labels = []
            for d in range(num_dates):
                if is_series_major:
                    indices = [d + s * num_dates for s in range(num_series)]
                else:
                    indices = [d * num_series + s for s in range(num_series)]
                date_labels = [labels[idx] for idx in indices]
                lbl_dict = dict(date_labels[0])
                lbl_dict[FieldName.TARGET] = np.vstack(
                    [l[FieldName.TARGET] for l in date_labels]
                )
                lbl_dict = self._restrict_max_dimensionality(lbl_dict)
                all_grouped_labels.append(lbl_dict)

            return list(zip(grouped_inputs, all_grouped_labels))

        self._preprocess(dataset_list)
        return self._group_all(dataset_list)

    def _preprocess(self, dataset: Dataset) -> None:
        """
        The preprocess function iterates over the dataset to gather data that
        is necessary for alignment.

        This includes:
             1. Storing first/last timestamp in the dataset
             2. Storing the frequency of the dataset
        """
        self.first_timestamp = None
        self.last_timestamp = None
        for data in dataset:
            timestamp = data[FieldName.START]

            if self.first_timestamp is None:
                self.first_timestamp = timestamp

            if self.last_timestamp is None:
                self.last_timestamp = timestamp

            assert self.first_timestamp is not None
            assert self.last_timestamp is not None

            self.first_timestamp = min(self.first_timestamp, timestamp)
            self.last_timestamp = max(
                self.last_timestamp,
                timestamp + len(data[FieldName.TARGET]) - 1,
            )
            self.frequency = timestamp.freq

        logging.info(
            "first/last timestamp found: "
            f"{self.first_timestamp}/{self.last_timestamp}"
        )

    def _group_all(self, dataset: Dataset) -> Dataset:
        if self.num_test_dates is None:
            grouped_dataset = self._prepare_train_data(dataset)
        else:
            grouped_dataset = self._prepare_test_data(dataset)
        return grouped_dataset

    def _prepare_train_data(self, dataset: Dataset) -> Dataset:
        logging.info("group training time series to datasets")

        # Creates a single multivariate time series from the
        # univariate series in the dataset
        grouped_data = self._transform_target(self._align_data_entry, dataset)
        grouped_data[FieldName.TARGET] = np.vstack(
            grouped_data[FieldName.TARGET]
        )

        fields = next(iter(dataset), {}).keys()
        if FieldName.FEAT_DYNAMIC_REAL in fields:
            grouped_data[FieldName.FEAT_DYNAMIC_REAL] = np.vstack(
                [data[FieldName.FEAT_DYNAMIC_REAL] for data in dataset],
            )
        grouped_data = self._restrict_max_dimensionality(grouped_data)
        grouped_data[FieldName.START] = self.first_timestamp
        grouped_data[FieldName.FEAT_STATIC_CAT] = [0]

        return ListDataset(
            [grouped_data], freq=self.frequency, one_dim_target=False
        )

    def _prepare_test_data(self, dataset: Dataset) -> Dataset:
        assert self.num_test_dates is not None

        logging.info("group test time series to datasets")

        dataset_list = list(dataset)
        grouped_data = self._transform_target(
            self._left_pad_data, dataset_list
        )

        # Splits test dataset with rolling date into N R^d time series,
        # where N is the number of rolling evaluation dates
        assert len(grouped_data[FieldName.TARGET]) % self.num_test_dates == 0
        num_dates = self.num_test_dates
        num_series = len(grouped_data[FieldName.TARGET]) // num_dates
        targets = grouped_data[FieldName.TARGET]

        is_series_major = False
        if num_dates > 1 and num_series > 1:
            end_0 = (
                dataset_list[0][FieldName.START]
                + len(dataset_list[0][FieldName.TARGET])
                - 1
            )
            end_1 = (
                dataset_list[1][FieldName.START]
                + len(dataset_list[1][FieldName.TARGET])
                - 1
            )
            if end_0 != end_1 or len(targets[0]) != len(targets[1]):
                is_series_major = True

        fields = next(iter(dataset_list), {}).keys()
        all_entries = list()
        for d in range(num_dates):
            if is_series_major:
                indices = [d + s * num_dates for s in range(num_series)]
            else:
                indices = [d * num_series + s for s in range(num_series)]

            dataset_at_test_date = [targets[idx] for idx in indices]
            entries_at_test_date = [dataset_list[idx] for idx in indices]

            grouped_data = dict()
            grouped_data[FieldName.TARGET] = np.vstack(dataset_at_test_date)

            if FieldName.FEAT_DYNAMIC_REAL in fields:
                grouped_data[FieldName.FEAT_DYNAMIC_REAL] = np.vstack(
                    [
                        data[FieldName.FEAT_DYNAMIC_REAL]
                        for data in entries_at_test_date
                    ],
                )
            grouped_data = self._restrict_max_dimensionality(grouped_data)
            grouped_data[FieldName.START] = self.first_timestamp
            grouped_data[FieldName.FEAT_STATIC_CAT] = [0]
            all_entries.append(grouped_data)

        return ListDataset(
            all_entries, freq=self.frequency, one_dim_target=False
        )

    def _align_data_entry(self, data: DataEntry) -> np.ndarray:
        ts = self.to_ts(data)
        return ts.reindex(
            pd.period_range(
                start=self.first_timestamp,
                end=self.last_timestamp,
                freq=data[FieldName.START].freq,
            ),
            fill_value=self.train_fill_function(ts),
        ).values

    def _left_pad_data(self, data: DataEntry) -> np.ndarray:
        ts = self.to_ts(data)
        return ts.reindex(
            pd.period_range(
                start=self.first_timestamp,
                end=ts.index[-1],
                freq=data[FieldName.START].freq,
            ),
            fill_value=self.test_fill_rule(ts),
        ).values

    @staticmethod
    def _transform_target(funcs, dataset: Dataset) -> DataEntry:
        return {FieldName.TARGET: [funcs(data) for data in dataset]}

    def _restrict_max_dimensionality(self, data: DataEntry) -> DataEntry:
        """
        Takes the last max_target_dimension dimensions from a multivariate
        dataentry.

        Parameters
        ----------
        data
            multivariate data entry with (dim, num_timesteps) target field

        Returns
        -------
        DataEntry
            data multivariate data entry with
            (max_target_dimension, num_timesteps) target field
        """

        if self.max_target_dimension is not None:
            # restrict maximum dimensionality (for faster testing)
            data[FieldName.TARGET] = data[FieldName.TARGET][
                -self.max_target_dimension :, :
            ]
            if FieldName.FEAT_DYNAMIC_REAL in data.keys():
                data[FieldName.FEAT_DYNAMIC_REAL] = data[
                    FieldName.FEAT_DYNAMIC_REAL
                ][-self.max_target_dimension :, :]
        return data

    @staticmethod
    def to_ts(data: DataEntry) -> pd.Series:
        return pd.Series(
            data[FieldName.TARGET],
            index=pd.period_range(
                start=data[FieldName.START],
                periods=len(data[FieldName.TARGET]),
                freq=data[FieldName.START].freq,
            ),
        )
