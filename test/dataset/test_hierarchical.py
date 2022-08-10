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


# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.dataset.hierarchical import HierarchicalTimeSeries


def random_ts(num_ts: int, periods: int, freq: str):
    index = pd.period_range(start="22-03-2020", periods=periods, freq=freq)

    return pd.concat(
        [
            pd.Series(data=np.random.random(size=len(index)), index=index)
            for _ in range(num_ts)
        ],
        axis=1,
    )


def test_three_level_hierarchy():
    # Simple three-level hierarchy containing 4 leaf nodes.
    S = np.vstack(([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]], np.eye(4)))

    num_ts, num_bottom_ts = S.shape
    periods = 24
    ts_at_bottom_level = random_ts(
        num_ts=num_bottom_ts,
        periods=periods,
        freq="H",
    )

    hts = HierarchicalTimeSeries(ts_at_bottom_level=ts_at_bottom_level, S=S)

    ts_at_all_levels = hts.ts_at_all_levels
    assert (ts_at_all_levels.index == ts_at_bottom_level.index).all(), (
        "The index of dataframe `ts_at_all_levels` does not match "
        "with that of  `ts_at_bottom_level`:\n"
        f"Index of `ts_at_bottom_level`: {ts_at_bottom_level.index}, \n "
        f"Index of `ts_at_all_levels`: {ts_at_all_levels.index}."
    )

    assert ts_at_all_levels.shape == (periods, num_ts), (
        "Hierarchical time series do not have the right shape. "
        f"Expected: {(periods, num_ts)}, "
        f"Obtained: {ts_at_bottom_level.shape}!"
    )

    root_level = hts.ts_at_all_levels.iloc[:, 0]
    root_level_expected = ts_at_bottom_level.sum(axis=1)
    np.testing.assert_array_almost_equal(
        root_level.values,
        root_level_expected.values,
        err_msg="Values of the time series at the root"
        "level are not correctly computed.",
    )

    level_1 = hts.ts_at_all_levels.iloc[:, 1:3]
    level_1_expected = pd.concat(
        [
            ts_at_bottom_level.iloc[:, :2].sum(axis=1),
            ts_at_bottom_level.iloc[:, 2:].sum(axis=1),
        ],
        axis=1,
    )
    np.testing.assert_array_almost_equal(
        level_1.values,
        level_1_expected.values,
        err_msg="Values of the time series at the first"
        "aggregated level (after the root) are not "
        "correctly computed.",
    )

    leaf_level = hts.ts_at_all_levels.iloc[:, 3:]
    np.testing.assert_array_almost_equal(
        ts_at_bottom_level.values,
        leaf_level.values,
        err_msg="Values of the time series at the bottom "
        "level do not agree with the given inputs.",
    )
