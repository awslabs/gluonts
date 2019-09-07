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
import unittest
from typing import cast

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.exception import GluonTSDataError
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.stat import (
    DatasetStatistics,
    ScaleHistogram,
    calculate_dataset_statistics,
)


def make_dummy_dynamic_feat(target, num_features) -> np.ndarray:
    # gives dummy dynamic_feat constructed from the target
    return np.vstack([target * (i + 1) for i in range(num_features)])


# default values for TimeSeries field
start = pd.Timestamp("1985-01-02", freq="1D")
target = np.random.randint(0, 10, 20)
fsc = [0, 1]
fsr = [0.1, 0.2]


def make_time_series(
    start=start,
    target=target,
    feat_static_cat=fsc,
    feat_static_real=fsr,
    num_feat_dynamic_cat=1,
    num_feat_dynamic_real=1,
) -> DataEntry:
    feat_dynamic_cat = (
        make_dummy_dynamic_feat(target, num_feat_dynamic_cat).astype("int64")
        if num_feat_dynamic_cat > 0
        else None
    )
    feat_dynamic_real = (
        make_dummy_dynamic_feat(target, num_feat_dynamic_real).astype("float")
        if num_feat_dynamic_real > 0
        else None
    )
    data = {
        "start": start,
        "target": target,
        "feat_static_cat": feat_static_cat,
        "feat_static_real": feat_static_real,
        "feat_dynamic_cat": feat_dynamic_cat,
        "feat_dynamic_real": feat_dynamic_real,
    }
    return data


def ts(
    start,
    target,
    feat_static_cat=None,
    feat_static_real=None,
    feat_dynamic_cat=None,
    feat_dynamic_real=None,
) -> DataEntry:
    d = {"start": start, "target": target}
    if feat_static_cat is not None:
        d["feat_static_cat"] = feat_static_cat
    if feat_static_real is not None:
        d["feat_static_real"] = feat_static_real
    if feat_dynamic_cat is not None:
        d["feat_dynamic_cat"] = feat_dynamic_cat
    if feat_dynamic_real is not None:
        d["feat_dynamic_real"] = feat_dynamic_real
    return d


class DatasetStatisticsTest(unittest.TestCase):
    def test_dataset_statistics(self) -> None:

        n = 2
        T = 10

        # use integers to avoid float conversion that can fail comparison
        np.random.seed(0)
        targets = np.random.randint(0, 10, (n, T))

        scale_histogram = ScaleHistogram()
        for i in range(n):
            scale_histogram.add(targets[i, :])

        scale_histogram.add([])

        expected = DatasetStatistics(
            integer_dataset=True,
            num_time_series=n + 1,
            num_time_observations=targets.size,
            mean_target_length=T * 2 / 3,
            min_target=targets.min(),
            mean_target=targets.mean(),
            mean_abs_target=targets.mean(),
            max_target=targets.max(),
            feat_static_real=[{0.1}, {0.2, 0.3}],
            feat_static_cat=[{1}, {2, 3}],
            num_feat_dynamic_real=2,
            num_feat_dynamic_cat=2,
            num_missing_values=0,
            scale_histogram=scale_histogram,
        )

        # FIXME: the cast below is a hack to make mypy happy
        timeseries = cast(
            Dataset,
            [
                make_time_series(
                    target=targets[0, :],
                    feat_static_cat=[1, 2],
                    feat_static_real=[0.1, 0.2],
                    num_feat_dynamic_cat=2,
                    num_feat_dynamic_real=2,
                ),
                make_time_series(
                    target=targets[1, :],
                    feat_static_cat=[1, 3],
                    feat_static_real=[0.1, 0.3],
                    num_feat_dynamic_cat=2,
                    num_feat_dynamic_real=2,
                ),
                make_time_series(
                    target=np.array([]),
                    feat_static_cat=[1, 3],
                    feat_static_real=[0.1, 0.3],
                    num_feat_dynamic_cat=2,
                    num_feat_dynamic_real=2,
                ),
            ],
        )

        found = calculate_dataset_statistics(timeseries)

        assert expected == found

    def test_dataset_histogram(self) -> None:

        # generates 2 ** N - 1 timeseries with constant increasing values
        N = 6
        n = 2 ** N - 1
        T = 5
        targets = np.ones((n, T))
        for i in range(0, n):
            targets[i, :] = targets[i, :] * i

        # FIXME: the cast below is a hack to make mypy happy
        timeseries = cast(
            Dataset, [make_time_series(target=targets[i, :]) for i in range(n)]
        )

        found = calculate_dataset_statistics(timeseries)

        hist = found.scale_histogram.bin_counts
        for i in range(0, N):
            assert i in hist
            assert hist[i] == 2 ** i


class DatasetStatisticsExceptions(unittest.TestCase):
    def test_dataset_statistics_exceptions(self) -> None:
        def check_error_message(expected_regex, dataset) -> None:
            with self.assertRaisesRegex(GluonTSDataError, expected_regex):
                calculate_dataset_statistics(dataset)

        check_error_message("Time series dataset is empty!", [])

        check_error_message(
            "Only empty time series found in the dataset!",
            [make_time_series(target=np.random.randint(0, 10, 0))],
        )

        # infinite target
        # check_error_message(
        #     "Target values have to be finite (e.g., not inf, -inf, "
        #     "or None) and cannot exceed single precision floating "
        #     "point range.",
        #     [make_time_series(target=np.full(20, np.inf))]
        # )

        # different number of feat_dynamic_{cat, real}
        check_error_message(
            "Found instances with different number of features in "
            "feat_dynamic_cat, found one with 2 and another with 1.",
            [
                make_time_series(num_feat_dynamic_cat=2),
                make_time_series(num_feat_dynamic_cat=1),
            ],
        )
        check_error_message(
            "Found instances with different number of features in "
            "feat_dynamic_cat, found one with 0 and another with 1.",
            [
                make_time_series(num_feat_dynamic_cat=0),
                make_time_series(num_feat_dynamic_cat=1),
            ],
        )
        check_error_message(
            "feat_dynamic_cat was found for some instances but not others.",
            [
                make_time_series(num_feat_dynamic_cat=1),
                make_time_series(num_feat_dynamic_cat=0),
            ],
        )
        check_error_message(
            "Found instances with different number of features in "
            "feat_dynamic_real, found one with 2 and another with 1.",
            [
                make_time_series(num_feat_dynamic_real=2),
                make_time_series(num_feat_dynamic_real=1),
            ],
        )
        check_error_message(
            "Found instances with different number of features in "
            "feat_dynamic_real, found one with 0 and another with 1.",
            [
                make_time_series(num_feat_dynamic_real=0),
                make_time_series(num_feat_dynamic_real=1),
            ],
        )
        check_error_message(
            "feat_dynamic_real was found for some instances but not others.",
            [
                make_time_series(num_feat_dynamic_real=1),
                make_time_series(num_feat_dynamic_real=0),
            ],
        )

        # infinite feat_dynamic_{cat,real}
        inf_dynamic_feat = np.full((2, len(target)), np.inf)
        check_error_message(
            "Features values have to be finite and cannot exceed single "
            "precision floating point range.",
            [
                ts(
                    start,
                    target,
                    feat_dynamic_cat=inf_dynamic_feat,
                    feat_static_cat=[0, 1],
                )
            ],
        )
        check_error_message(
            "Features values have to be finite and cannot exceed single "
            "precision floating point range.",
            [
                ts(
                    start,
                    target,
                    feat_dynamic_real=inf_dynamic_feat,
                    feat_static_cat=[0, 1],
                )
            ],
        )

        # feat_dynamic_{cat, real} different length from target
        check_error_message(
            "Each feature in feat_dynamic_cat has to have the same length as the "
            "target. Found an instance with feat_dynamic_cat of length 1 and a "
            "target of length 20.",
            [
                ts(
                    start=start,
                    target=target,
                    feat_static_cat=[0, 1],
                    feat_dynamic_cat=np.ones((1, 1)),
                )
            ],
        )
        check_error_message(
            "Each feature in feat_dynamic_real has to have the same length as the "
            "target. Found an instance with feat_dynamic_real of length 1 and a "
            "target of length 20.",
            [
                ts(
                    start=start,
                    target=target,
                    feat_static_cat=[0, 1],
                    feat_dynamic_real=np.ones((1, 1)),
                )
            ],
        )

        # feat_static_{cat, real} different length
        check_error_message(
            "Not all feat_static_cat vectors have the same length 2 != 1.",
            [
                ts(start=start, target=target, feat_static_cat=[0, 1]),
                ts(start=start, target=target, feat_static_cat=[1]),
            ],
        )
        check_error_message(
            "Not all feat_static_real vectors have the same length 2 != 1.",
            [
                ts(start=start, target=target, feat_static_real=[0, 1]),
                ts(start=start, target=target, feat_static_real=[1]),
            ],
        )

        calculate_dataset_statistics(
            # FIXME: the cast below is a hack to make mypy happy
            cast(
                Dataset,
                [
                    make_time_series(num_feat_dynamic_cat=2),
                    make_time_series(num_feat_dynamic_cat=2),
                ],
            )
        )

        calculate_dataset_statistics(
            # FIXME: the cast below is a hack to make mypy happy
            cast(
                Dataset,
                [
                    make_time_series(num_feat_dynamic_cat=0),
                    make_time_series(num_feat_dynamic_cat=0),
                ],
            )
        )
