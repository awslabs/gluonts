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
from typing import Optional, List, Tuple, Any
from itertools import repeat

from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
import numpy as np

from .timeseries import TimeSeries, TimeSeriesCorpus


class WindowsDataset(TorchDataset):
    def __init__(
        self,
        corpus: TimeSeriesCorpus,
        windows: List[Tuple[int, int, int]],
    ) -> None:
        super(WindowsDataset, self).__init__()
        self.corpus = corpus
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def _fetch(self, index: int) -> Tuple[TimeSeries, slice]:
        scope_id, start, length = self.windows[index]
        window = slice(start, start + length)
        return self.corpus[scope_id], window

    def __getitem__(self, index: int) -> TimeSeries:
        ts, window = self._fetch(index)
        subseq = ts[window]
        return subseq

    @classmethod
    def sliding_windows(
        cls,
        corpus: TimeSeriesCorpus,
        window_size: int,
        window_size_plus: int = 0,
        window_size_minus: int = 0,
        window_shift: int = 1,
        seed: int = 42,
    ) -> WindowsDataset:
        max_window_size = window_size + window_size_plus
        min_window_size = window_size - window_size_minus

        windows = []
        np.random.seed(seed)
        for scope_id, series in tqdm(
            enumerate(corpus),
            desc="Building dataset",
            total=len(corpus),
        ):
            stop = len(series) - min_window_size
            index = np.arange(stop, 0, -window_shift)
            window_length = np.random.randint(
                min_window_size, max_window_size + 1, size=index.shape
            )
            window_length = np.minimum(len(series) - index, window_length)
            windows.extend(
                zip(repeat(scope_id, len(index)), index, window_length)
            )
        return cls(corpus, windows)

    @classmethod
    def random_windows(
        cls,
        corpus: TimeSeriesCorpus,
        n_windows: int,
        window_size: int,
        window_size_plus: int = 0,
        window_size_minus: int = 0,
        window_shift: int = 1,
        seed: int = 42,
    ) -> WindowsDataset:
        max_window_size = window_size + window_size_plus
        min_window_size = window_size - window_size_minus

        windows = []
        np.random.seed(seed)
        for scope_id, series in tqdm(
            enumerate(corpus),
            desc="Building dataset",
            total=len(corpus),
        ):
            stop = len(series) - min_window_size
            index = np.arange(stop, 0, -window_shift)
            window_length = np.random.randint(
                min_window_size, max_window_size + 1, size=index.shape
            )
            window_length = np.minimum(len(series) - index, window_length)
            windows.extend(
                zip(repeat(scope_id, len(index)), index, window_length)
            )
        window_idx = np.random.choice(
            len(windows),
            size=n_windows,
            replace=len(windows) < n_windows,
        )
        windows = [windows[i] for i in window_idx]
        return cls(corpus, windows)
