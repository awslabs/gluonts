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
import logging
import numpy as np
import pandas as pd
from typing import Callable, Optional

# First-party imports
from gluonts.dataset.common import ListDataset, DataEntry, Dataset
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName


OLDEST_SUPPORTED_TIMESTAMP = pd.Timestamp(1800, 1, 1, 12)
LATEST_SUPPORTED_TIMESTAMP = pd.Timestamp(2200, 1, 1, 12)


class MultivariateGrouper:
    """
    The MultivariateGrouper takes a univariate dataset and groups it into a
    single multivariate time series. Therefore, this class allows the user
    to convert a univariate dataset into a multivariate dataset without making
    a separate copy of the dataset.

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

        self.first_timestamp = LATEST_SUPPORTED_TIMESTAMP
        self.last_timestamp = OLDEST_SUPPORTED_TIMESTAMP
        self.frequency = ""

    def __call__(self, dataset: Dataset) -> Dataset:
        self._preprocess(dataset)
        return self._group_all(dataset)

    def _preprocess(self, dataset: Dataset) -> None:
        """
        The preprocess function iterates over the dataset to gather data that
        is necessary for alignment.
        This includes
            1) Storing first/last timestamp in the dataset
            2) Storing the frequency of the dataset
        """
        for data in dataset:
            timestamp = data[FieldName.START]
            self.first_timestamp = min(self.first_timestamp, timestamp)
            self.last_timestamp = max(
                self.last_timestamp,
                timestamp + (len(data[FieldName.TARGET]) - 1) * timestamp.freq,
            )
            self.frequency = timestamp.freq
        logging.info(
            f"first/last timestamp found: "
            f"{self.first_timestamp}/{self.last_timestamp}"
        )

    def _group_all(self, dataset: Dataset) -> Dataset:
        if self.num_test_dates is None:
            grouped_dataset = self._prepare_train_data(dataset)
        else:
            grouped_dataset = self._prepare_test_data(dataset)
        return grouped_dataset

    def _prepare_train_data(self, dataset: Dataset) -> ListDataset:
        logging.info("group training time-series to datasets")

        grouped_data = self._transform_target(self._align_data_entry, dataset)
        grouped_data = self._restrict_max_dimensionality(grouped_data)
        grouped_data[FieldName.START] = self.first_timestamp
        grouped_data[FieldName.FEAT_STATIC_CAT] = [0]

        return ListDataset(
            [grouped_data], freq=self.frequency, one_dim_target=False
        )

    def _prepare_test_data(self, dataset: Dataset) -> ListDataset:
        logging.info("group test time-series to datasets")

        grouped_data = self._transform_target(self._left_pad_data, dataset)
        # splits test dataset with rolling date into N R^d time series where
        # N is the number of rolling evaluation dates
        split_dataset = np.split(
            grouped_data[FieldName.TARGET], self.num_test_dates
        )

        all_entries = list()
        for dataset_at_test_date in split_dataset:
            grouped_data = dict()
            grouped_data[FieldName.TARGET] = np.array(
                list(dataset_at_test_date), dtype=np.float32
            )
            grouped_data = self._restrict_max_dimensionality(grouped_data)
            grouped_data[FieldName.START] = self.first_timestamp
            grouped_data[FieldName.FEAT_STATIC_CAT] = [0]
            all_entries.append(grouped_data)

        return ListDataset(
            all_entries, freq=self.frequency, one_dim_target=False
        )

    def _align_data_entry(self, data: DataEntry) -> np.array:
        ts = self.to_ts(data)
        return ts.reindex(
            pd.date_range(
                start=self.first_timestamp,
                end=self.last_timestamp,
                freq=data[FieldName.START].freq,
            ),
            fill_value=self.train_fill_function(ts),
        ).values

    def _left_pad_data(self, data: DataEntry) -> np.array:
        ts = self.to_ts(data)
        return ts.reindex(
            pd.date_range(
                start=self.first_timestamp,
                end=ts.index[-1],
                freq=data[FieldName.START].freq,
            ),
            fill_value=self.test_fill_rule(ts),
        ).values

    @staticmethod
    def _transform_target(funcs, dataset: Dataset) -> DataEntry:
        return {FieldName.TARGET: np.array([funcs(data) for data in dataset])}

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
        return data

    @staticmethod
    def to_ts(data: DataEntry) -> pd.Series:
        return pd.Series(
            data[FieldName.TARGET],
            index=pd.date_range(
                start=data[FieldName.START],
                periods=len(data[FieldName.TARGET]),
                freq=data[FieldName.START].freq,
            ),
        )
