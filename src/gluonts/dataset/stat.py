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
import math
from collections import defaultdict
from typing import Any, List, NamedTuple, Optional, Set

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.core.exception import assert_data_error
from gluonts.gluonts_tqdm import tqdm
from gluonts.dataset.field_names import FieldName


class ScaleHistogram:
    """
    Scale histogram of a timeseries dataset

    This counts the number of timeseries whose mean of absolute values is in
    the `[base ** i, base ** (i+1)]` range for all possible `i`.
    The number of entries with empty target is counted separately.

    Parameters
    ----------
    base
        Log-width of the histogram's buckets.
    bin_counts
    empty_target_count
    """

    @validated()
    def __init__(
        self,
        base: float = 2.0,
        bin_counts: Optional[dict] = None,
        empty_target_count: int = 0,
    ) -> None:
        self._base = base
        self.bin_counts = defaultdict(
            int, {} if bin_counts is None else bin_counts
        )
        self.empty_target_count = empty_target_count
        self.__init_args__ = dict(
            base=self._base,
            bin_counts=self.bin_counts,
            empty_target_count=empty_target_count,
        )

    def bucket_index(self, target_values):
        assert len(target_values) > 0
        scale = np.mean(np.abs(target_values))
        scale_bin = int(math.log(scale + 1.0, self._base))
        return scale_bin

    def add(self, target_values):
        if len(target_values) > 0:
            bucket = self.bucket_index(target_values)
            self.bin_counts[bucket] = self.bin_counts[bucket] + 1
        else:
            self.empty_target_count = self.empty_target_count + 1

    def count(self, target):
        if len(target) > 0:
            return self.bin_counts[self.bucket_index(target)]
        else:
            return self.empty_target_count

    def __len__(self):
        return self.empty_target_count + sum(self.bin_counts.values())

    def __eq__(self, other):
        return (
            isinstance(other, ScaleHistogram)
            and self.bin_counts == other.bin_counts
            and self.empty_target_count == other.empty_target_count
            and self._base == other._base
        )

    def __str__(self):
        string_repr = [
            "count of scales in {min}-{max}:{count}".format(
                min=self._base ** base_index - 1,
                max=self._base ** (base_index + 1) - 1,
                count=count,
            )
            for base_index, count in sorted(
                self.bin_counts.items(), key=lambda x: x[0]
            )
        ]
        return "\n".join(string_repr)


class DatasetStatistics(NamedTuple):
    """
    A NamedTuple to store the statistics of a Dataset.
    """

    integer_dataset: bool
    max_target: float
    mean_abs_target: float
    mean_target: float
    mean_target_length: float
    min_target: float
    feat_static_real: List[Set[float]]
    feat_static_cat: List[Set[int]]
    num_feat_dynamic_real: Optional[int]
    num_feat_dynamic_cat: Optional[int]
    num_missing_values: int
    num_time_observations: int
    num_time_series: int
    scale_histogram: ScaleHistogram

    # DO NOT override the __str__ method, since we rely that we can load
    # DatasetStatistics again; i.e. stats == eval(str(stats))

    def __eq__(self, other):
        for x, y in zip(self._asdict().values(), other._asdict().values()):
            if isinstance(x, float):
                if abs(x - y) > abs(0.0001 * x):
                    return False
            elif x != y:
                return False
        return True


# TODO: reorganize modules to avoid circular dependency
# TODO: and substitute Any with Dataset
def calculate_dataset_statistics(ts_dataset: Any) -> DatasetStatistics:
    """
    Computes the statistics of a given Dataset.

    Parameters
    ----------
    ts_dataset
        Dataset of which to compute the statistics.

    Returns
    -------
    DatasetStatistics
        NamedTuple containing the statistics.
    """
    num_time_observations = 0
    num_time_series = 0
    min_target = 1e20
    max_target = -1e20
    sum_target = 0.0
    sum_abs_target = 0.0
    integer_dataset = True
    observed_feat_static_cat: Optional[List[Set[int]]] = None
    observed_feat_static_real: Optional[List[Set[float]]] = None
    num_feat_static_real: Optional[int] = None
    num_feat_static_cat: Optional[int] = None
    num_feat_dynamic_real: Optional[int] = None
    num_feat_dynamic_cat: Optional[int] = None
    num_missing_values = 0

    scale_histogram = ScaleHistogram()

    with tqdm(enumerate(ts_dataset, start=1), total=len(ts_dataset)) as it:
        for num_time_series, ts in it:

            # TARGET
            target = ts[FieldName.TARGET]
            observed_target = target[~np.isnan(target)]
            num_observations = len(observed_target)

            if num_observations > 0:
                # 'nan' is handled in observed_target definition
                assert_data_error(
                    np.all(np.isfinite(observed_target)),
                    "Target values have to be finite (e.g., not inf, -inf, "
                    "or None) and cannot exceed single precision floating "
                    "point range.",
                )

                num_time_observations += num_observations
                min_target = float(min(min_target, observed_target.min()))
                max_target = float(max(max_target, observed_target.max()))
                num_missing_values += int(np.isnan(target).sum())
                sum_target += float(observed_target.sum())
                sum_abs_target += float(np.abs(observed_target).sum())
                integer_dataset = integer_dataset and bool(
                    np.all(np.mod(observed_target, 1) == 0)
                )

            scale_histogram.add(
                observed_target
            )  # after checks for inf and None

            # FEAT_STATIC_CAT
            feat_static_cat = (
                ts[FieldName.FEAT_STATIC_CAT]
                if FieldName.FEAT_STATIC_CAT in ts
                else []
            )

            if num_feat_static_cat is None:
                num_feat_static_cat = len(feat_static_cat)
                observed_feat_static_cat = [
                    set() for _ in range(num_feat_static_cat)
                ]

            # needed to type check
            assert num_feat_static_cat is not None
            assert observed_feat_static_cat is not None

            assert_data_error(
                num_feat_static_cat == len(feat_static_cat),
                "Not all feat_static_cat vectors have the same length {} != {}.",
                num_feat_static_cat,
                len(feat_static_cat),
            )
            for i, c in enumerate(feat_static_cat):
                observed_feat_static_cat[i].add(c)

            # FEAT_STATIC_REAL
            feat_static_real = (
                ts[FieldName.FEAT_STATIC_REAL]
                if FieldName.FEAT_STATIC_REAL in ts
                else []
            )

            if num_feat_static_real is None:
                num_feat_static_real = len(feat_static_real)
                observed_feat_static_real = [
                    set() for _ in range(num_feat_static_real)
                ]

            # needed to type check
            assert num_feat_static_real is not None
            assert observed_feat_static_real is not None

            assert_data_error(
                num_feat_static_real == len(feat_static_real),
                "Not all feat_static_real vectors have the same length {} != {}.",
                num_feat_static_real,
                len(feat_static_real),
            )
            for i, c in enumerate(feat_static_real):
                observed_feat_static_real[i].add(c)

            # FEAT_DYNAMIC_CAT
            feat_dynamic_cat = (
                ts[FieldName.FEAT_DYNAMIC_CAT]
                if FieldName.FEAT_DYNAMIC_CAT in ts
                else None
            )

            if feat_dynamic_cat is None:
                # feat_dynamic_cat not found, check it was the first ts we encounter or
                # that feat_dynamic_cat were seen before
                assert_data_error(
                    num_feat_dynamic_cat is None or num_feat_dynamic_cat == 0,
                    "feat_dynamic_cat was found for some instances but not others.",
                )
                num_feat_dynamic_cat = 0
            else:
                if num_feat_dynamic_cat is None:
                    # first num_feat_dynamic_cat found
                    num_feat_dynamic_cat = feat_dynamic_cat.shape[0]
                else:
                    assert_data_error(
                        num_feat_dynamic_cat == feat_dynamic_cat.shape[0],
                        "Found instances with different number of features in "
                        "feat_dynamic_cat, found one with {} and another with {}.",
                        num_feat_dynamic_cat,
                        feat_dynamic_cat.shape[0],
                    )

                assert_data_error(
                    np.all(np.isfinite(feat_dynamic_cat)),
                    "Features values have to be finite and cannot exceed single "
                    "precision floating point range.",
                )
                num_feat_dynamic_cat_time_steps = feat_dynamic_cat.shape[1]
                assert_data_error(
                    num_feat_dynamic_cat_time_steps == len(target),
                    "Each feature in feat_dynamic_cat has to have the same length as "
                    "the target. Found an instance with feat_dynamic_cat of length {} "
                    "and a target of length {}.",
                    num_feat_dynamic_cat_time_steps,
                    len(target),
                )

            # FEAT_DYNAMIC_REAL
            feat_dynamic_real = (
                ts[FieldName.FEAT_DYNAMIC_REAL]
                if FieldName.FEAT_DYNAMIC_REAL in ts
                else None
            )

            if feat_dynamic_real is None:
                # feat_dynamic_real not found, check it was the first ts we encounter or
                # that feat_dynamic_real were seen before
                assert_data_error(
                    num_feat_dynamic_real is None
                    or num_feat_dynamic_real == 0,
                    "feat_dynamic_real was found for some instances but not others.",
                )
                num_feat_dynamic_real = 0
            else:
                if num_feat_dynamic_real is None:
                    # first num_feat_dynamic_real found
                    num_feat_dynamic_real = feat_dynamic_real.shape[0]
                else:
                    assert_data_error(
                        num_feat_dynamic_real == feat_dynamic_real.shape[0],
                        "Found instances with different number of features in "
                        "feat_dynamic_real, found one with {} and another with {}.",
                        num_feat_dynamic_real,
                        feat_dynamic_real.shape[0],
                    )

                assert_data_error(
                    np.all(np.isfinite(feat_dynamic_real)),
                    "Features values have to be finite and cannot exceed single "
                    "precision floating point range.",
                )
                num_feat_dynamic_real_time_steps = feat_dynamic_real.shape[1]
                assert_data_error(
                    num_feat_dynamic_real_time_steps == len(target),
                    "Each feature in feat_dynamic_real has to have the same length as "
                    "the target. Found an instance with feat_dynamic_real of length {} "
                    "and a target of length {}.",
                    num_feat_dynamic_real_time_steps,
                    len(target),
                )

    assert_data_error(num_time_series > 0, "Time series dataset is empty!")
    assert_data_error(
        num_time_observations > 0,
        "Only empty time series found in the dataset!",
    )

    # note this require the above assumption to avoid a division by zero
    # runtime error
    mean_target_length = num_time_observations / num_time_series

    # note this require the above assumption to avoid a division by zero
    # runtime error
    mean_target = sum_target / num_time_observations
    mean_abs_target = sum_abs_target / num_time_observations

    integer_dataset = integer_dataset and min_target >= 0.0

    assert len(scale_histogram) == num_time_series

    return DatasetStatistics(
        integer_dataset=integer_dataset,
        max_target=max_target,
        mean_abs_target=mean_abs_target,
        mean_target=mean_target,
        mean_target_length=mean_target_length,
        min_target=min_target,
        num_missing_values=num_missing_values,
        feat_static_real=observed_feat_static_real
        if observed_feat_static_real
        else [],
        feat_static_cat=observed_feat_static_cat
        if observed_feat_static_cat
        else [],
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_feat_dynamic_cat=num_feat_dynamic_cat,
        num_time_observations=num_time_observations,
        num_time_series=num_time_series,
        scale_histogram=scale_histogram,
    )
