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
start = pd.Timestamp('1985-01-02', freq="1D")
target = np.random.randint(0, 10, 20)
cat = [0, 1]


def make_time_series(
    start=start, target=target, cat=cat, num_dynamic_feat=1
) -> DataEntry:
    dynamic_feat = (
        make_dummy_dynamic_feat(target, num_dynamic_feat)
        if num_dynamic_feat > 0
        else None
    )
    data = {
        'start': start,
        'target': target,
        'cat': cat,
        'dynamic_feat': dynamic_feat,
    }
    return data


def ts(start, target, cat=None, dynamic_feat=None) -> DataEntry:
    d = {'start': start, 'target': target}
    if cat is not None:
        d['cat'] = cat
    if dynamic_feat is not None:
        d['dynamic_feat'] = dynamic_feat
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
            cats=[{0}, {1, 2}],
            num_dynamic_feat=2,
            num_missing_values=0,
            scale_histogram=scale_histogram,
        )

        # FIXME: the cast below is a hack to make mypy happy
        timeseries = cast(
            Dataset,
            [
                make_time_series(
                    target=targets[0, :], cat=[0, 1], num_dynamic_feat=2
                ),
                make_time_series(
                    target=targets[1, :], cat=[0, 2], num_dynamic_feat=2
                ),
                make_time_series(
                    target=np.array([]), cat=[0, 2], num_dynamic_feat=2
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

        check_error_message('Time series dataset is empty!', [])

        check_error_message(
            'Only empty time series found in the dataset!',
            [make_time_series(target=np.random.randint(0, 10, 0))],
        )

        # different number of dynamic_feat
        check_error_message(
            'Found instances with different number of features in '
            'dynamic_feat, found one with 2 and another with 1.',
            [
                make_time_series(num_dynamic_feat=2),
                make_time_series(num_dynamic_feat=1),
            ],
        )

        # different number of dynamic_feat
        check_error_message(
            'Found instances with different number of features in '
            'dynamic_feat, found one with 0 and another with 1.',
            [
                make_time_series(num_dynamic_feat=0),
                make_time_series(num_dynamic_feat=1),
            ],
        )

        # different number of dynamic_feat
        check_error_message(
            'dynamic_feat was found for some instances but not others.',
            [
                make_time_series(num_dynamic_feat=1),
                make_time_series(num_dynamic_feat=0),
            ],
        )

        # infinite target
        # check_error_message(
        #     'Target values have to be finite (e.g., not "inf", "-inf", '
        #     '"nan", or null) and cannot exceed single precision floating '
        #     'point range.',
        #     [make_time_series(target=np.full(20, np.inf))]
        # )

        # infinite dynamic_feat
        inf_dynamic_feat = np.full((2, len(target)), np.inf)
        check_error_message(
            'Features values have to be finite and cannot exceed single '
            'precision floating point range.',
            [
                ts(
                    start=start,
                    target=target,
                    dynamic_feat=inf_dynamic_feat,
                    cat=[0, 1],
                )
            ],
        )

        # cat different length
        check_error_message(
            'Not all cat vectors have the same length 2 != 1.',
            [ts(start, target, [0, 1]), ts(start, target, [1])],
        )

        # cat different length
        check_error_message(
            'Each feature in dynamic_feat has to have the same length as the '
            'target. Found an instance with dynamic_feat of length 1 and a '
            'target of length 20.',
            [
                ts(start, target, [0, 1], dynamic_feat=np.ones((1, 1))),
                ts(start, target, [1], dynamic_feat=np.ones((1, 1))),
            ],
        )

        calculate_dataset_statistics(
            # FIXME: the cast below is a hack to make mypy happy
            cast(
                Dataset,
                [
                    make_time_series(num_dynamic_feat=2),
                    make_time_series(num_dynamic_feat=2),
                ],
            )
        )

        calculate_dataset_statistics(
            # FIXME: the cast below is a hack to make mypy happy
            cast(
                Dataset,
                [
                    make_time_series(num_dynamic_feat=0),
                    make_time_series(num_dynamic_feat=0),
                ],
            )
        )
