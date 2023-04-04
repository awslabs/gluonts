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

import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset, DataEntry, Dataset


class Grouper:
    # todo the contract of this grouper is missing from the documentation, what it does when, how it pads values etc
    def __init__(
        self,
        fill_value: float = 0.0,
        max_target_dim: int = None,
        align_data: bool = True,
        num_test_dates: int = None,
    ) -> None:
        self.fill_value = fill_value

        self.first_timestamp = pd.Timestamp(2200, 1, 1, 12)
        self.last_timestamp = pd.Timestamp(1800, 1, 1, 12)
        self.frequency = None
        self.align_data = align_data
        self.max_target_length = 0
        self.num_test_dates = num_test_dates
        self.max_target_dimension = max_target_dim

    def __call__(self, dataset: Dataset) -> Dataset:
        self._preprocess(dataset)
        return self._group_all(dataset)

    def _group_all(self, dataset: Dataset) -> Dataset:
        if self.align_data:
            funcs = {"target": self._align_data_entry}

        if self.num_test_dates is None:
            grouped_dataset = self._prepare_train_data(dataset, funcs)
        else:
            grouped_dataset = self._prepare_test_data(dataset)
        return grouped_dataset

    def to_ts(self, data: DataEntry):
        return pd.Series(
            data["target"],
            index=pd.date_range(
                start=data["start"],
                periods=len(data["target"]),
                freq=data["start"].freq,
            ),
        )

    def _align_data_entry(self, data: DataEntry) -> DataEntry:
        d = data.copy()
        # fill target invidually if we want to fill all of them, we should use a dataframe
        ts = self.to_ts(data)
        d["target"] = ts.reindex(
            pd.date_range(
                start=self.first_timestamp,
                end=self.last_timestamp,
                freq=d["start"].freq,
            ),
            fill_value=ts.mean(),
        )
        d["start"] = self.first_timestamp
        return d

    def _preprocess(self, dataset: Dataset) -> None:
        """
        The preprocess function iterates over the dataset to gather data that
        is necessary for grouping.
        This includes:
            1) Storing first/last timestamp in the dataset
            2) Aligning time series
            3) Calculating groups
        """
        for data in dataset:
            timestamp = data["start"]
            self.first_timestamp = min(self.first_timestamp, timestamp)

            self.frequency = (
                timestamp.freq if self.frequency is None else self.frequency
            )
            self.last_timestamp = max(
                self.last_timestamp,
                timestamp + len(data["target"]) * self.frequency,
            )

            # todo
            self.max_target_length = max(
                self.max_target_length, len(data["target"])
            )
        logging.info(
            f"first/last timestamp found: {self.first_timestamp}/{self.last_timestamp}"
        )

    def _prepare_train_data(self, dataset, funcs):
        logging.info("group training time-series to datasets")
        grouped_data = {}
        for key in funcs.keys():
            grouped_entry = [funcs[key](data)[key] for data in dataset]

            # we check that each time-series has the same length
            assert (
                len(set([len(x) for x in grouped_entry])) == 1
            ), f"alignement did not work as expected more than on length found: {set([len(x) for x in grouped_entry])}"
            grouped_data[key] = np.array(grouped_entry)
        if self.max_target_dimension is not None:
            # targets are often sorted by incr amplitude, use the last one when restricted number is asked
            grouped_data["target"] = grouped_data["target"][
                -self.max_target_dimension :, :
            ]
        grouped_data["item_id"] = "all_items"
        grouped_data["start"] = self.first_timestamp
        grouped_data["feat_static_cat"] = [0]
        return ListDataset(
            [grouped_data], freq=self.frequency, one_dim_target=False
        )

    def _prepare_test_data(self, dataset):
        logging.info("group test time-series to datasets")

        def left_pad_data(data: DataEntry):
            ts = self.to_ts(data)
            filled_ts = ts.reindex(
                pd.date_range(
                    start=self.first_timestamp,
                    end=ts.index[-1],
                    freq=data["start"].freq,
                ),
                fill_value=0.0,
            )
            return filled_ts.values

        grouped_entry = [left_pad_data(data) for data in dataset]
        grouped_entry = np.array(grouped_entry)

        split_dataset = np.split(grouped_entry, self.num_test_dates)

        all_entries = list()
        for dataset_at_test_date in split_dataset:
            grouped_data = dict()
            assert (
                len(set([len(x) for x in dataset_at_test_date])) == 1
            ), "all test time-series should have the same length"
            grouped_data["target"] = np.array(
                list(dataset_at_test_date), dtype=np.float32
            )
            if self.max_target_dimension is not None:
                grouped_data["target"] = grouped_data["target"][
                    -self.max_target_dimension :, :
                ]
            grouped_data["item_id"] = "all_items"
            grouped_data["start"] = self.first_timestamp
            grouped_data["feat_static_cat"] = [0]
            all_entries.append(grouped_data)

        return ListDataset(
            all_entries, freq=self.frequency, one_dim_target=False
        )
