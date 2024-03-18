# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from typing import Iterable, Iterator, List, Optional

import numpy as np

import sklearn

from ncad.ts import TimeSeries, TimeSeriesDataset


class TimeSeriesTransform(object):
    def __call__(self, ts_iterable: Iterable[TimeSeries]) -> Iterator[TimeSeries]:
        for ts in ts_iterable:
            yield self.transform(ts.copy())

    def transform(self, ts: TimeSeries) -> TimeSeries:
        raise NotImplementedError()

    def __add__(self, other: "TimeSeriesTransform") -> "TimeSeriesTransform":
        return Chain([self, other])


class Chain(TimeSeriesTransform):
    def __init__(
        self,
        ts_transforms: List[TimeSeriesTransform],
    ) -> None:
        self.ts_transforms = []
        for trans in ts_transforms:
            # flatten chains
            if isinstance(trans, Chain):
                self.ts_transforms.extend(trans.ts_transforms)
            else:
                self.ts_transforms.append(trans)

    def __call__(self, data_it: Iterable[TimeSeries]) -> Iterator[TimeSeries]:
        tmp = data_it
        for trans in self.ts_transforms:
            tmp = trans(tmp)
        return tmp

    def transform(self, ts: TimeSeries) -> TimeSeries:
        tmp = ts.copy()
        for trans in self.ts_transforms:
            tmp = trans.transform(tmp)
        return tmp


class IdentityTransform(TimeSeriesTransform):
    def transform(self, ts: TimeSeries) -> TimeSeries:
        return ts


class ApplyWithProbability(TimeSeriesTransform):
    def __init__(
        self,
        base_transform: TimeSeriesTransform,
        p: float,
    ) -> None:
        self.base_transform = base_transform
        self.p = p

    def transform(self, ts: TimeSeries) -> TimeSeries:
        if np.random.uniform() > self.p:
            return ts
        return self.base_transform.transform(ts)


class TimeSeriesScaler(TimeSeriesTransform):
    def __init__(
        self,
        type: str = ["standard", "robust"][1],
    ) -> None:
        super().__init__()

        self.type = type
        if self.type == "standard":
            self.scaler = sklearn.preprocessing.StandardScaler()
        if self.type == "robust":
            self.scaler = sklearn.preprocessing.RobustScaler()

    def transform(self, ts: TimeSeries) -> TimeSeries:
        if ts.shape[0] > 0:
            ts.values = self.scaler.fit_transform(ts.values.reshape(ts.shape)).squeeze()

        return ts


class RandomPickListTransforms(TimeSeriesTransform):
    def __init__(
        self,
        l_transforms: List[TimeSeriesTransform],
        number_picks: int = 1,
        mixture_proportions: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.l_transforms = l_transforms
        self.number_picks = number_picks
        if mixture_proportions is None:
            mixture_proportions = np.ones(len(l_transforms)) / len(l_transforms)
        self.mixture_proportions = mixture_proportions.astype(float)

        assert len(mixture_proportions) == len(l_transforms)

    def transform(self, ts: TimeSeries) -> TimeSeries:
        draws = np.random.choice(
            range(len(self.l_transforms)), self.number_picks, p=self.mixture_proportions
        )
        for k in draws:
            ts = self.l_transforms[k].transform(ts)

        return ts


class ShortALongB(TimeSeriesTransform):
    """Apply different transforms for TimeSeries shorter and longer than a length threshold

    If the time series is shorter than length_threshold, it applies TimeSeriesTransform A,
    otherwise, applies transform B.
    """

    def __init__(
        self,
        length_threshold: int = 0,
        A: TimeSeriesTransform = IdentityTransform(),
        B: TimeSeriesTransform = IdentityTransform(),
    ):
        self.length_threshold = length_threshold
        self.A = A
        self.B = B

    def transform(self, ts: TimeSeries) -> TimeSeries:
        if ts.shape[0] < self.length_threshold:
            return self.A.transform(ts)

        return self.B.transform(ts)


class LabelNoise(TimeSeriesTransform):
    def __init__(self, p_flip_1_to_0: float) -> None:
        assert 0 <= p_flip_1_to_0 <= 1
        self.p_flip_1_to_0 = p_flip_1_to_0

    def transform(self, ts: TimeSeries) -> TimeSeries:
        if self.p_flip_1_to_0 == 0:
            return ts

        if np.random.uniform() > self.p_flip_1_to_0:
            return ts

        ts = ts.copy()
        ts.labels = np.zeros_like(ts.labels)
        return ts


def apply_transform_to_dataset(
    dataset: TimeSeriesDataset,
    transform: TimeSeriesTransform,
) -> TimeSeriesDataset:
    new_dataset = TimeSeriesDataset()
    for ts in dataset:
        new_dataset.append(transform.transform(ts))
    return new_dataset


def get_magnitude(metric):
    return np.nanmean(metric)
