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

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, fields
from typing import Iterator, List, Optional
import torch
from torch.utils.data import IterableDataset, Dataset

from .dataset import TimeSeries, TimeSeriesDataset


class WeightedIndexIterator:
    """Iterator that caches a number of indices sampled according to given weights.

    This gives a great performance speedup since np.random.choice is the bottleneck
    of the data loading. This class samples and caches a certain number of indices and
    return them until new ones need to be sampled.
    """

    def __init__(self, weights: np.ndarray, num_cache: int = 1024):
        """
        Args:
            weights:  np.ndarray, containing the sample weights
            num_cache: the number of indices to cache
        """
        self.weights = weights
        self.num_cache = num_cache

    def __iter__(self):
        self.i = self.num_cache
        return self

    def __next__(self):
        if self.i >= self.num_cache:
            self.idx = np.random.choice(
                len(self.weights), p=self.weights, size=self.num_cache
            )
            self.i = 0

        idx = self.idx[self.i]
        self.i += 1
        return idx


@dataclass
class Triplet:
    """
    A triplet is composed of a support set, observed queries, and corresponding (unobserved) future queries.
    """

    support_set: List[
        TimeSeries
    ]  # length: support_set_size, length TimeSeries: context_length
    query_past: List[
        TimeSeries
    ]  # length: num_queries, length TimeSeries: context_length
    query_future: List[
        TimeSeries
    ]  # length: num_queries, length TimeSeries: prediction_length

    def __iter__(self):
        return (getattr(self, field.name) for field in fields(self))


class TripletDataset(Dataset[Triplet]):
    """
    The triplet dataset gets a list of queries and corresponding support set and returns them as triplets.
    """

    def __init__(
        self, queries: TimeSeriesDataset, support_sets: List[List[TimeSeries]]
    ):
        self.queries = queries
        self.support_sets = support_sets
        assert len(queries) == len(
            support_sets
        ), "For each query there must be exactly one support set"

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> Triplet:
        pl = self.queries.prediction_length
        ts_l = len(self.queries[index])
        query_past = [self.queries[index][0:-pl]]
        query_future = [self.queries[index][ts_l - pl : ts_l]]
        return Triplet(self.support_sets[index], query_past, query_future)


class SamplingTripletDataset(IterableDataset[Triplet]):  # type: ignore
    """
    The sampling triplet dataset randomly samples support sets and past queries
    along with their future prediction horizon.
    All three sets consist of time series windows sliced from the original time series. The support set time series
    end before the prediction horizon begins to avoid time leakage.
    The dataset yields infinitely many items. Support set time series length is for now context_length.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        support_set_size: int,
        num_queries: int,
        context_length: int,
        support_length: int,
        prediction_length: int,
        catch22_nn: Optional[np.ndarray] = None,
        cheat: float = 0.0,
    ):
        """
        Args:
            dataset: The dataset to sample from.
            support_set_size: The size of the support set.
            num_queries: The number of queries.
            context_length: The length of the context.
            support_length: The length of the support time series.
            prediction_length: The length of the prediction.
            catch22_nn: Contains for each index its 100 nearest neighbors w.r.t. catch22 distance.
                If not None, slices from the closest `support_set_size` time series are chosen as support set.
            cheat: If true, the query (time series to be predicted) shifted by the prediction length
                is contained in the support set, i.e. the ground truth is in the support set.
        """

        # Initialize
        super().__init__()
        self.dataset = dataset
        self.support_set_size = support_set_size
        self.num_queries = num_queries
        self.context_length = context_length
        self.support_length = support_length
        self.prediction_length = prediction_length
        self.cheat = cheat
        assert (
            not cheat or self.num_queries == 1
        ), "Cheat sampling only allows num_queries = 1"

        time_series_lengths = np.array([len(s) for s in dataset])
        time_series_weights = time_series_lengths / time_series_lengths.sum()
        self.index_iterator = iter(WeightedIndexIterator(time_series_weights))

        self.catch22_nn = catch22_nn
        assert not (
            self.catch22_nn is not None and num_queries > 1
        ), "catch22 support set selection only works with num_queries equal to one"

    def __iter__(self) -> Iterator[Triplet]:
        while True:
            (
                query_past,
                query_future,
                cheat_query,
                query_idx,
            ) = self._sample_queries()

            # We do not use the qsplit option for training
            support_set = sample_supps(
                supps_size=self.support_set_size,
                length=self.support_length,
                dataset=self.dataset,
                cheat_query=cheat_query[0]
                if np.random.rand() < self.cheat
                else None,
                index_iterator=self.index_iterator
                if self.catch22_nn is None
                else iter(self.catch22_nn[query_idx]),
            )
            yield Triplet(support_set, query_past, query_future)

    def _sample_queries(self):
        query_past = []
        query_future = []
        query_cheat = []
        for _ in range(self.num_queries):
            # First, sample a time series with probability relative to its length
            idx = next(self.index_iterator)
            series = self.dataset[idx]

            # Then, sample a slice with uniform probability
            # context should be at least prediction length long
            split_point = np.random.choice(
                np.arange(
                    self.prediction_length,
                    len(series) - self.prediction_length + 1,
                )
            )
            prediction = series[
                split_point : split_point + self.prediction_length
            ]
            context_start = max(0, split_point - self.context_length)
            context = series[context_start:split_point]

            query_past.append(context)
            query_future.append(prediction)

            # sample the start of the cheat time series
            cheat_earliest_start = max(
                0,
                context_start
                - self.support_length
                + self.context_length
                + self.prediction_length,
            )
            cheat_start = np.random.choice(
                np.arange(cheat_earliest_start, context_start + 1)
            )
            cheat_end = min(len(series), cheat_start + self.support_length)
            query_cheat.append(series[cheat_start:cheat_end])
        return query_past, query_future, query_cheat, idx


class SequentialTripletDataset(Dataset[Triplet]):  # type: ignore
    """
    The sequential triplet dataset traverses the dataset and uses the last prediction length slice as future query.
    The support set is sampled randomly. The length of dataset is the number of times series
    divided by the number of queries.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        support_set_size: int,
        num_queries: int,
        context_length: int,
        support_length: int,
        prediction_length: int,
        support_dataset: TimeSeriesDataset = None,
        seed: Optional[int] = None,
        catch22_nn: Optional[np.ndarray] = None,
        cheat: bool = False,
    ):
        """
        Args:
            dataset: The dataset to sample from.
            support_set_size: The size of the support set.
            num_queries: The number of queries.
            context_length: The length of the context.
            support_length: The length of the support time series.
            prediction_length: The length of the prediction.
            support_dataset: The dataset to choose the support set from. If not provided `dataset` is used.
                This is used to choose the support set for the test split from the val split (technical reasons).
            seed: The random seed for sampling the support set
            catch22_nn: Contains for each index its 100 nearest neighbors w.r.t. catch22 distance.
                If not None, slices from the closest `support_set_size` time series are chosen as support set.
            cheat: If true, the query (time series to be predicted) shifted by the prediction length
                is contained in the support set, i.e. the ground truth is in the support set.
        """

        super().__init__()
        self.dataset = dataset
        self.support_set_size = support_set_size
        self.num_queries = num_queries
        self.context_length = context_length
        self.support_length = support_length
        self.prediction_length = prediction_length
        self.cheat = cheat
        assert (
            not cheat or self.num_queries == 1
        ), "Cheat sampling only allows num_queries = 1"

        self.support_dataset = support_dataset or dataset
        self.seed = seed
        if catch22_nn is None:
            time_series_lengths = np.array(
                [len(s) for s in self.support_dataset]
            )
            time_series_weights = (
                time_series_lengths / time_series_lengths.sum()
            )
            self.index_iterator = iter(
                WeightedIndexIterator(time_series_weights)
            )
        self.catch22_nn = catch22_nn
        assert not (
            self.catch22_nn is not None and num_queries > 1
        ), "catch22 support set selection only works with num_queries equal to one"

    def __len__(self) -> int:
        return len(self.dataset) // self.num_queries

    def __getitem__(self, index: int) -> Triplet:
        start = index * self.num_queries
        end = start + self.num_queries
        query_past, query_future = zip(
            *(self._last_slice(self.dataset[i]) for i in range(start, end))
        )
        # support time series should end before earliest start of future queries
        q_split = min(q.start_date for q in query_future)
        query = self.dataset[index]
        cheat_query = query[
            max(0, len(query) - self.support_length) : len(query)
        ]
        support_set = sample_supps(
            supps_size=self.support_set_size,
            length=self.support_length,
            dataset=self.support_dataset,
            q_split=q_split,
            # TODO: The seeding does not work anymore, we could use torch again in the validation split
            # but this will be slower
            # seed=self.seed + index if self.seed else None,
            cheat_query=cheat_query if np.random.rand() < self.cheat else None,
            index_iterator=self.index_iterator
            if self.catch22_nn is None
            else iter(self.catch22_nn[query_past[0].item_id]),
        )
        return Triplet(support_set, query_past, query_future)

    def _last_slice(self, series: TimeSeries):
        split_point = len(series) - self.dataset.prediction_length
        prediction = series[split_point : split_point + self.prediction_length]
        context = series[
            max(0, split_point - self.context_length) : split_point
        ]
        return context, prediction


def sample_supps(
    supps_size: int,
    length: int,
    dataset: TimeSeriesDataset,
    index_iterator: Iterator,
    q_split: Optional[pd.Timestamp] = None,
    # seed: Optional[int] = None,
    cheat_query: Optional[TimeSeries] = None,
):
    """
    Args:
        supps_size: The number of support time series
        length: The length of the support time series slice
        dataset: The dataset to choose support time series slices from
        index_iterator: An iterator that returns indices of the dataset sampled w.r.t. some weights.
        q_split: The latest possible end time of all support time series
        cheat_query: If not None, the cheat query is contained in the support set at a random position.
    """

    support_set = []
    for i in range(supps_size):
        series = dataset[next(index_iterator)]
        if q_split:
            # support set slice is the one closest to the prediction point of the query
            freq = series.start_date.freq
            end_point = min(
                (
                    q_split.to_period(freq) - series.start_date.to_period(freq)
                ).n,
                len(series),
            )
            if end_point <= 0:
                # this should basically never happen and is only there to not break the training in case it happens
                support_set.append(
                    TimeSeries(
                        dataset_name=series.dataset_name,
                        start_date=None,
                        values=torch.zeros(1, 1),
                        scale=torch.as_tensor([0, 1]),
                    )
                )
                continue
        else:
            # if possible choose a full-length slice
            end_point = np.random.randint(
                low=min(len(series), length), high=len(series) + 1
            )
        support_ts = series[max(0, end_point - length) : end_point]
        support_set.append(support_ts)
    if cheat_query is not None:
        support_set[np.random.choice(supps_size)] = cheat_query
    return support_set


class SuperSamplingTripletDataset(IterableDataset[Triplet]):  # type: ignore
    """
    The super sampling triplet dataset randomly samples support sets and past queries
    along with their future prediction horizon from a list of sampling datasets.
    First a sampling dataset is randomly chosen.
    Then the chosen triplet dataset samples support, query past and query future set.
    The dataset yields infinitely many items.
    """

    def __init__(
        self, datasets: List[SamplingTripletDataset], dataset_sampling: str
    ):
        """
        Args:
            datasets: The base datasets to sample from.
        """

        # Initialize
        super().__init__()
        datasets = list(datasets)
        self.datasets = [iter(dataset) for dataset in datasets]

        if dataset_sampling == "weighted":
            # Sample a dataset with probability relative to the total number of observations in it
            self.dataset_weights = [
                sampling_dataset.dataset.number_of_time_steps
                for sampling_dataset in datasets
            ]
            self.dataset_weights = np.array(self.dataset_weights) / sum(
                self.dataset_weights
            )
        elif dataset_sampling == "uniform":
            self.dataset_weights = None
        else:
            raise ValueError(
                f"dataset sampling {dataset_sampling} is not implemented."
            )

    def __iter__(self) -> Iterator[Triplet]:
        while True:
            index = np.random.choice(
                len(self.datasets), p=self.dataset_weights
            )
            # Return item of chosen base dataset
            yield next(self.datasets[index])
