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
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

from .timeseries import TimeSeries, TimeSeriesCorpus
from .windows import WindowsDataset


class ForecastingDataset(WindowsDataset):
    def __init__(
        self,
        corpus: TimeSeriesCorpus,
        windows: List[Tuple[int, int]],
        horizon: int,
        context: Optional[int] = None,
        gap: int = 0,
    ) -> None:
        super(ForecastingDataset, self).__init__(
            corpus=corpus,
            windows=windows,
        )
        self.horizon = horizon
        self.context = context
        self.gap = gap

    def __getitem__(self, index: int) -> Tuple[TimeSeries, TimeSeries]:
        ts, ctx_window, tgt_window = self._fetch(index)
        ctx = ts[ctx_window]
        tgt = ts[tgt_window]
        return ctx, tgt

    def _fetch(self, index: int) -> Tuple[TimeSeries, slice, slice]:
        scope_id, tgt_head = self.windows[index]
        tgt_tail = tgt_head + self.horizon
        ctx_tail = tgt_head - self.gap
        ctx_head = None if self.context is None else ctx_tail - self.context
        ctx_window = slice(ctx_head, ctx_tail)
        tgt_window = slice(tgt_head, tgt_tail)
        return self.corpus[scope_id], ctx_window, tgt_window

    @classmethod
    def sliding_windows(
        cls,
        corpus: TimeSeriesCorpus,
        horizon: int,
        context: Optional[int] = None,
        gap: int = 0,
        shift: Optional[int] = None,
    ):
        shift = shift or horizon
        windows = []
        for scope_id, series in tqdm(
            enumerate(corpus),
            desc="Building dataset",
            total=len(corpus),
        ):
            stop = len(series) - horizon
            start = gap
            if context is not None:
                start += context
            windows.extend(
                [(scope_id, index) for index in np.arange(stop, start, -shift)]
            )
        return cls(
            corpus,
            windows=windows,
            horizon=horizon,
            context=context,
            gap=gap,
        )

    @classmethod
    def random_windows(
        cls,
        corpus: TimeSeriesCorpus,
        n_windows: int,
        horizon: int,
        context: Optional[int] = None,
        gap: int = 0,
        shift: int = 1,
        cold_start: bool = False,
        favor_recent: bool = False,
    ):
        all_windows = []
        weights = []
        for scope_id, series in tqdm(
            enumerate(corpus),
            desc="Building dataset",
            total=len(corpus),
        ):
            stop = len(series) - horizon
            start = gap
            if context is not None and not cold_start:
                start = start + context
            else:
                start = start + 1
            windows = [
                (scope_id, index) for index in range(stop, start, -shift)
            ]
            all_windows.extend(windows)
            if favor_recent:
                weights.extend(range(stop, start, -shift))
            else:
                weights.extend([1.0] * len(windows))
        weights = np.array(weights) / np.sum(weights)
        np.random.seed(42)
        windows_idx = np.random.choice(
            len(all_windows),
            size=n_windows,
            replace=len(all_windows) < n_windows,
            p=weights,
        )
        windows = [all_windows[i] for i in windows_idx]
        return cls(
            corpus,
            windows=windows,
            horizon=horizon,
            context=context,
            gap=gap,
        )
