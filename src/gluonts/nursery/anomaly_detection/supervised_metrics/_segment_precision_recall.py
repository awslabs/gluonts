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

from typing import List, Tuple

import numpy as np

from .utils import labels_to_ranges, range_overlap


def segment_precision_recall(
    real_ranges: List[range], pred_ranges: List[range]
) -> Tuple[float, float]:
    """
    Segment based metric is less lenient than the range based metric.

    This metric counts

        * a ground truth anomaly range as true-positive as long as there is an overlapping predicted
            anomaly range; it does not penalize the position of the overlap. If there is no overlapping predicted range
            for given ground truth range, then it counts as one false-negative irrespective of its size.

        * a predicted anomaly range as false positive if it has a nonempty overlap with a "normal" range.

    Parameters
    ----------
    real_ranges
        List of ranges corresponding to ground truth anomalies.
    pred_ranges
        List of predicted anomaly ranges.

    Returns
    -------
    Tuple containing segment based precision and recall.

    """
    # Compute the number of true positives and false negatives by going over each of the ground truth anomaly ranges
    tp, fn = 0, 0
    for tr in real_ranges:
        tp_found = False
        for pr in pred_ranges:
            if range_overlap(tr, pr):
                tp_found = True
                break

        if tp_found:
            tp += 1
        else:
            fn += 1

    # Deduce the ranges corresponding to normal behavior
    if real_ranges and real_ranges[-1]:
        max_ix_real_range = real_ranges[-1][-1]
    else:
        max_ix_real_range = -1

    if pred_ranges and pred_ranges[-1]:
        max_ix_pred_range = pred_ranges[-1][-1]
    else:
        max_ix_pred_range = -1

    num_time_points = max(max_ix_real_range, max_ix_pred_range) + 1
    normal_ind = np.array([True] * num_time_points)
    for tr in real_ranges:
        normal_ind[tr] = False
    normal_ranges = labels_to_ranges(normal_ind)

    # Compute the number of false positives and true negatives by going over each of the ground truth normal ranges
    fp, tn = 0, 0
    for nr in normal_ranges:
        fp_found = False
        for pr in pred_ranges:
            if range_overlap(nr, pr):
                fp_found = True
                fp += 1

        if not fp_found:
            tn += 1

    # avoid division by zero.
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
