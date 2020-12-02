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

from typing import Dict, List, Tuple

import numpy as np
import pytest
from sklearn.metrics import auc

from gluonts.nursery.anomaly_detection.supervised_metrics import (
    aggregate_precision_recall_curve,
    buffered_precision_recall,
    segment_precision_recall,
)
from gluonts.nursery.anomaly_detection.supervised_metrics._buffered_precision_recall import (
    extend_ranges,
)
from gluonts.nursery.anomaly_detection.supervised_metrics.bounded_pr_auc import (
    bounded_pr_auc,
)
from gluonts.nursery.anomaly_detection.supervised_metrics.utils import (
    labels_to_ranges,
    range_overlap,
)

TEST_CASES = [
    # edge cases
    {
        "real_ranges": [],
        "pred_ranges": [],
        "range_precision": 0.0,
        "segment_precision": 0.0,
        "range_recall": 0.0,
        "segment_recall": 0.0,
    },
    {
        "real_ranges": [],
        "pred_ranges": [range(1, 3)],
        "range_precision": 0.0,
        "segment_precision": 0.0,
        "range_recall": 0.0,
        "segment_recall": 0.0,
    },
    {
        "real_ranges": [range(1, 4)],
        "pred_ranges": [],
        "range_precision": 0.0,
        "segment_precision": 0.0,
        "range_recall": 0.0,
        "segment_recall": 0.0,
    },
    {
        "real_ranges": [],
        "pred_ranges": [range(0, 0)],  # still empty range
        "range_precision": 0.0,
        "segment_precision": 0.0,
        "range_recall": 0.0,
        "segment_recall": 0.0,
    },
    # perfect precision and recall
    {
        "real_ranges": [range(2, 4), range(7, 8), range(11, 20)],
        "pred_ranges": [range(2, 4), range(7, 8), range(11, 20)],
        "range_precision": 1.0,
        "segment_precision": 1.0,
        "range_recall": 1.0,
        "segment_recall": 1.0,
    },
    # worst precision and recall
    {
        "real_ranges": [range(2, 4), range(7, 8), range(11, 20)],
        "pred_ranges": [range(0, 2), range(4, 7), range(8, 11)],
        "range_precision": 0.0,
        "segment_precision": 0.0,
        "range_recall": 0.0,
        "segment_recall": 0.0,
    },
    # perfect recall but worst precision
    {
        "real_ranges": [range(2, 4), range(7, 8), range(11, 20)],
        "pred_ranges": [range(0, 20)],
        "range_precision": 0.2,  # 1.0 * (1.0 / 3.0 * (12 / 20.0)), with cardinality_factor = 1/3
        "segment_precision": 0.5,  # All the three normal ranges are falsely classified as positive
        "range_recall": 1.0,
        "segment_recall": 1.0,
    },
    # perfect precision but worst recall (only one element of the largest range detected)
    {
        "real_ranges": [range(2, 4), range(7, 8), range(11, 20)],
        "pred_ranges": [range(11, 12)],
        "range_precision": 1.0,
        "segment_precision": 1.0,
        "range_recall": 1.0
        / 27.0,  # 1.0 / 3.0 * (0.0 + 0.0 + 1.0 / 9.0), with cardinality_factor = 1 (not shown)
        "segment_recall": 1.0 / 3.0,
    },
    # perfect precision but bad recall (only one true small range detected)
    {
        "real_ranges": [range(2, 4), range(7, 8), range(11, 20)],
        "pred_ranges": [range(7, 8)],
        "range_precision": 1.0,
        "segment_precision": 1.0,
        "range_recall": 1.0
        / 3.0,  # 1.0 / 3.0 * (0.0 + 1.0 + 0.0), with cardinality_factor = 1 (not shown)
        "segment_recall": 1.0 / 3.0,
    },
    # perfect precision but not so good recall
    {
        "real_ranges": [range(2, 4), range(7, 8), range(11, 20)],
        "pred_ranges": [range(7, 8), range(11, 15)],
        "range_precision": 1.0,
        "segment_precision": 1.0,
        "range_recall": 13.0
        / 27.0,  # 1.0 / 3.0 * (0.0 + 1.0 + 4.0 / 9.0), with cardinality_factor = 1 (not shown)
        "segment_recall": 2.0 / 3.0,
    },
    # perfect precision and reasonably good recall
    {
        "real_ranges": [range(2, 4), range(7, 8), range(11, 20)],
        "pred_ranges": [range(3, 4), range(7, 8), range(11, 15)],
        "range_precision": 1.0,
        "segment_precision": 1.0,
        "range_recall": 35.0
        / 54.0,  # 1.0 / 3.0 * (1.0 / 2.0 + 1.0 + 4.0 / 9.0), cardinality_factor = 1 (not shown)
        "segment_recall": 1.0,
    },
    # perfect precision but not so good recall (fragmented detection)
    {
        "real_ranges": [range(11, 20)],
        "pred_ranges": [range(12, 14), range(17, 18), range(19, 20)],
        "range_precision": 1.0,
        "segment_precision": 1.0,
        "range_recall": 4.0
        / 27.0,  # 1.0 * (1.0 / 3.0 * 4.0 / 9.0), with cardinality_factor = 1/3
        "segment_recall": 1.0,
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_segment_precision_recall(test_case: Dict):
    (
        obtained_segment_precision,
        obtained_segment_recall,
    ) = segment_precision_recall(
        test_case["real_ranges"], test_case["pred_ranges"]
    )

    expected_segment_precision = test_case["segment_precision"]
    np.testing.assert_almost_equal(
        obtained_segment_precision, expected_segment_precision
    ), f"Expected segment precision: {expected_segment_precision}, " f"obtained segment precision: {obtained_segment_precision}"

    expected_segment_recall = test_case["segment_recall"]
    np.testing.assert_almost_equal(
        obtained_segment_recall, expected_segment_recall
    ), f"Expected segment recall: {expected_segment_recall}, " f"obtained segment recall: {obtained_segment_recall}"


TEST_CASES_LABELS_TO_RANGES = [
    {
        "labels": [0],
        "ranges": [],
    },
    {
        "labels": [1],
        "ranges": [range(0, 1)],
    },
    {
        "labels": [0] * 10,
        "ranges": [],
    },
    {
        "labels": [1] * 10,
        "ranges": [range(0, 10)],
    },
    {
        "labels": [1, 0],
        "ranges": [range(0, 1)],
    },
    {
        "labels": [1, 0, 1, 1],
        "ranges": [range(0, 1), range(2, 4)],
    },
    {
        "labels": [1, 0, 1, 1, 0],
        "ranges": [range(0, 1), range(2, 4)],
    },
    {
        "labels": [0, 1, 1, 0],
        "ranges": [range(1, 3)],
    },
    {
        "labels": [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        "ranges": [range(2, 4), range(6, 7), range(8, 11), range(12, 13)],
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES_LABELS_TO_RANGES)
def test_labels_to_ranges(test_case: Dict):
    expected_ranges_ls = test_case["ranges"]
    obtained_ranges_ls = labels_to_ranges(test_case["labels"])
    assert expected_ranges_ls == obtained_ranges_ls, (
        "Obtained list of ranges do not match the expected list. "
        f"Expected: f{expected_ranges_ls}, obtained: f{obtained_ranges_ls}"
    )


EXTEND_RANGES_TEST_CASES = [
    {
        "input": [range(2, 4), range(8, 10), range(15, 16)],
        "window_length": 2,
        "right_output": [range(2, 6), range(8, 12), range(15, 18)],
        "left_output": [range(0, 4), range(6, 10), range(13, 16)],
    },
    {
        "input": [range(2, 4), range(8, 10), range(18, 26)],
        "window_length": 4,
        "right_output": [range(2, 8), range(8, 14), range(18, 30)],
        "left_output": [range(0, 4), range(4, 10), range(14, 26)],
    },
    {
        "input": [range(2, 4), range(8, 10), range(18, 26)],
        "window_length": 5,
        "right_output": [range(2, 8), range(8, 15), range(18, 31)],
        "left_output": [range(0, 4), range(4, 10), range(13, 26)],
    },
    {
        "input": [range(2, 4), range(8, 10), range(18, 26)],
        "window_length": 20,
        "right_output": [range(2, 8), range(8, 18), range(18, 46)],
        "left_output": [range(0, 4), range(4, 10), range(10, 26)],
    },
    {"input": [], "window_length": 4, "right_output": [], "left_output": []},
]


@pytest.mark.parametrize("test_case", EXTEND_RANGES_TEST_CASES)
def test_extend_ranges(test_case):
    # extend_range working properly
    assert (
        extend_ranges(test_case["input"], test_case["window_length"])
        == test_case["right_output"]
    )

    # extend_range left working properly
    assert (
        extend_ranges(
            test_case["input"], test_case["window_length"], direction="left"
        )
        == test_case["left_output"]
    )

    # 0 behaves as the "identity"
    assert extend_ranges(test_case["input"], 0) == test_case["input"]


RANGE_OVERLAP_TEST_CASES = [
    {"input": [range(3, 7), range(7, 9)], "output": False},
    {"input": [range(3, 7), range(17, 19)], "output": False},
    {"input": [range(3, 7), range(6, 9)], "output": True},
    {"input": [range(19, 21), range(0, 200)], "output": True},
    {"input": [range(19, 28), range(10, 20)], "output": True},
]


@pytest.mark.parametrize("test_case", RANGE_OVERLAP_TEST_CASES)
def test_range_overlap(test_case):
    (a, b), o = test_case["input"], test_case["output"]
    assert range_overlap(a, b) == o
    assert range_overlap(b, a) == o


BUFFERED_PRECISION_RECALL_TEST_CASES = [
    {
        "real_ranges": [],
        "pred_ranges": [],
        "buffer_lengths": [0, 1, 2, 3],
        "precision": [1, 1, 1, 1],
        "recall": [1, 1, 1, 1],
    },
    {
        "real_ranges": [
            range(5, 8),
            range(20, 26),
            range(100, 155),
            range(200, 201),
        ],
        "pred_ranges": [],
        "buffer_lengths": [0, 1, 2, 3],
        "precision": [1, 1, 1, 1],
        "recall": [0, 0, 0, 0],
    },
    {
        "real_ranges": [],
        "pred_ranges": [
            range(5, 8),
            range(20, 26),
            range(100, 155),
            range(200, 201),
        ],
        "buffer_lengths": [0, 1, 2, 3],
        "precision": [0, 0, 0, 0],
        "recall": [1, 1, 1, 1],
    },
    {
        "real_ranges": [
            range(5, 8),
            range(20, 26),
            range(100, 155),
            range(200, 201),
        ],
        "pred_ranges": [range(9, 12)],
        "buffer_lengths": [0, 1, 2, 3],
        "precision": [0, 0, 1.0, 1.0],
        "recall": [0, 0, 0.25, 0.25],
    },
    {
        "real_ranges": [
            range(5, 8),
            range(20, 26),
            range(100, 155),
            range(200, 201),
        ],
        "pred_ranges": [
            range(10, 12),
            range(27, 32),
            range(156, 191),
            range(202, 208),
        ],
        "buffer_lengths": [0, 1, 2, 3],
        "precision": [0, 0, 0.75, 1.0],
        "recall": [0, 0, 0.75, 1.0],
    },
    {
        "real_ranges": [
            range(5, 8),
            range(20, 26),
            range(100, 155),
            range(200, 201),
        ],
        "pred_ranges": [range(3, 205)],
        "buffer_lengths": [0, 1, 2, 3],
        "precision": [1.0, 1.0, 1.0, 1.0],
        "recall": [1.0, 1.0, 1.0, 1.0],
    },
]


@pytest.mark.parametrize("test_case", BUFFERED_PRECISION_RECALL_TEST_CASES)
def test_buffered_precision_recall(test_case):
    real_ranges, pred_ranges = (
        test_case["real_ranges"],
        test_case["pred_ranges"],
    )
    for i, bl in enumerate(test_case["buffer_lengths"]):
        assert buffered_precision_recall(
            real_ranges, pred_ranges, buffer_length=bl
        ) == (test_case["precision"][i], test_case["recall"][i])


@pytest.fixture
def labels_and_scores() -> List[Tuple[np.array, np.array]]:
    label1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    scores1 = np.array(
        [
            0.2,
            0.3,
            0.5,
            0.7,
            4,
            2.5,
            0.3,
            0.2,
            0.7,
            0.3,
            0.2,
            4,
            3,
            8,
            0.2,
            0.1,
        ]
    )

    label2 = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    scores2 = np.array(
        [0.2, 0.3, 0.5, 7, 4, 2.5, 0.3, 0.2, 0.7, 0.3, 0.2, 4, 3, 8, 0.2, 0.1]
    )

    label3 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    scores3 = np.array(
        [0.2, 0.3, 5, 7, 4, 2.5, 0.3, 0.2, 0.7, 0.3, 0.2, 4, 3, 8, 0.2, 0.1]
    )
    return [(label1, scores1), (label2, scores2), (label3, scores3)]


def test_aggregate_precision_recall_curve(labels_and_scores):
    precisions, recalls, thresholds = aggregate_precision_recall_curve(
        labels_and_scores, precision_recall_fn=segment_precision_recall
    )
    sorted_pr = list(
        zip(*sorted(zip(recalls, precisions), key=lambda x: (x[0], x[1])))
    )
    pr_auc = auc(*sorted_pr)
    np.testing.assert_almost_equal(pr_auc, 0.49960, 5)


def test_bounded_pr_auc(labels_and_scores):
    precisions, recalls, thresholds = aggregate_precision_recall_curve(
        labels_and_scores, precision_recall_fn=segment_precision_recall
    )
    pr_auc = bounded_pr_auc(precisions, recalls)
    np.testing.assert_almost_equal(pr_auc, 0.49960, 5)

    pr_auc_1 = bounded_pr_auc(precisions, recalls, 0.2)
    np.testing.assert_almost_equal(pr_auc_1, 0.32460, 5)
