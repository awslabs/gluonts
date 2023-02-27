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

from typing import Optional

import numpy as np


def fill_forward(a: np.ndarray, fill_start=None) -> np.ndarray:
    """
    Forward fill an array. If `fill_start` is not None, then the
    NaNs in the beginning of the array will be filled with `fill_start`.
    """
    a = np.array(a)
    assert a.ndim == 1

    # forward fill labels
    idx = np.where(~np.isnan(a), np.arange(a.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]

    # nan values at the beginning of window are left
    if fill_start is not None:
        a[~np.isfinite(a)] = fill_start

    return a


def labels_filter(
    a: np.ndarray,
    num_open: int,
    num_clear: int,
    initial_label: int = 0,
    forward_fill: bool = False,
):
    if forward_fill:
        a = fill_forward(a)

    state = initial_label
    count_high = 0
    count_low = 0
    output = []
    for label in a:
        if label and not np.isnan(label):
            count_low = 0
            count_high += 1
        else:
            count_high = 0
            count_low += 1
        if state and count_low == num_clear:
            state = 0
        if not state and count_high == num_open:
            state = 1
        output.append(state)
    return np.array(output)


def n_k_filter(
    a: np.ndarray,
    num_open: int,
    num_clear: int,
    num_open_suff: Optional[int] = None,
    num_clear_suff: Optional[int] = None,
    initial_label: int = 0,
    forward_fill: bool = False,
) -> np.ndarray:
    """
    Implements the (N, K)-filter that opens/closes anomalies based on
    observing K or greater labels in the last N time steps.

    Parameters
    ----------
    a
        input array, type boolean
    num_open
        number of time steps to look back for opening an anomaly
    num_clear
        number of time steps to look back for clearing an anomaly
    num_open_suff
        number of positive labels in the lookback period sufficient
        to open an anomaly. defaults to num_open.
    num_clear_suff
        number of positive labels in the lookback period sufficient
        to clear an anomaly. defaults to num_clear.
    initial_label
        1 if the initial state is an anomaly, 0 if not. default 0.
    """
    num_open_suff = num_open_suff or num_open
    num_clear_suff = num_clear_suff or num_clear

    if forward_fill:
        a = fill_forward(a)

    def causal_conv(a, f):
        if len(f) > 1:
            return np.convolve(a, f)[: -(len(f) - 1)]
        return np.convolve(a, f)

    up = causal_conv(a > 0, np.ones(num_open)) >= num_open_suff
    down = causal_conv(a == 0, np.ones(num_clear)) >= num_clear_suff

    s = initial_label
    out = []
    for i in range(len(up)):
        s = up[i] if s == 0 else ~down[i]
        out.append(s)
    return np.array(out, dtype=int)
