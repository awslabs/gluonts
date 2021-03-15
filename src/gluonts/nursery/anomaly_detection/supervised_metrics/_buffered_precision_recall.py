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

from .utils import range_overlap


def extend_ranges(
    ranges: List[range], extension_length: int, direction: str = "right"
) -> List[range]:
    """
    Extends a list of ranges by `extension_length` steps.

    Parameters
    ----------
    ranges: List[range]
        Ranges to be extended, non-overlapping and in sorted order.
    extension_length: int
        Positive integer, the length of the extension.
    direction: str
        Direction of the extension. If "right" ("left"), ranges are extended towards
        the right (left). Default "right".
    Returns
    -------
    out_ranges: List[range]
    """
    if not ranges:
        return []
    assert extension_length >= 0, "`extension_length` must be zero or above"
    assert direction in [
        "left",
        "right",
    ], "`direction` must be one of 'left' or 'right'."

    out_ranges = []

    # iterate over real ranges (ground truth anomalies) and "extend" them to include
    # slack windows where it's OK if they're caught
    for i, old_range in enumerate(ranges):
        if direction == "right":
            range_end_ub = (
                ranges[i + 1].start
                if i < len(ranges) - 1
                else ranges[-1].stop + extension_length + 1
            )
            new_range = range(
                old_range.start,
                min(range_end_ub, old_range.stop + extension_length),
            )
        else:
            range_start_lb = ranges[i - 1].stop if i > 0 else 0
            new_range = range(
                max(range_start_lb, old_range.start - extension_length),
                old_range.stop,
            )

        out_ranges.append(new_range)

    return out_ranges


def buffered_precision_recall(
    real_ranges: List[range],
    pred_ranges: List[range],
    buffer_length: int = 5,
) -> Tuple[float, float]:
    """
    Implements a new range-based precision recall metric, that measures how well anomalies
    (`real_ranges`) are caught with labels (`pred_ranges`).

    We extend anomaly ranges by a number of time steps (`buffer_length`) to accomodate
    those raised with a lag. For example, if an annotator has marked `range(5, 9)`, and the
    model has labeled `range(11, 13)` as an anomaly, we would often like to mark this as a
    correctly raised anomaly. There are two reasons for this. (i) Human annotators often draw
    boxes around anomalies with a certain "margin," i.e., with a lead and a lag around the
    true anomaly. (ii) The low-pass filter raises anomalies with a certain latency.

    Therefore, this function looks for intersections between "extended" anomaly ranges,
    those with a buffer of `buffer_length` added after the annotated range, and the predicted
    ranges. Any intersection is counted as a success. More precisely,

    - If an "extended" anomaly range intersects with any labeled range, it's "caught." If an
      anomaly intersects with no predicted range, it's not caught. Recall is,
      `n_caught_anomalies / n_all_anomalies`.
    - If a predicted range intersects with any "extended" anomaly range, it's a good alarm.
      Precision is `n_good_pred_ranges / n_pred_ranges`.

    Note that the numerators (numbers of true positives) are different for precision
    and recall. This is since an anomaly can be caught by multiple pred ranges, as well as a
    pred range marking two separate anomalies. This function allows for this behavior.
    Moreover, a prediction range is either "good" (it intersects with an extended anomaly range,
    and is a "true positive predicted range") or or "bad" (false positive). This is different
    than `segment_precision_recall`, since there a predicted range is counted towards true
    positives and false positives at the same time if it spans the intersection of an
    anomaly segment and a non-anomaly segment.

    Parameters
    ----------
    real_ranges: List[range]
        Python range objects representing ground truth anomalies (e.g., as annotated by
        human labelers). Ranges must ve non-overlapping and in sorted order.
    pred_ranges: List[range]
        Python range objects representing labels produced by the model. Ranges must be
        non-overlapping and in sorted order.
    buffer_length: int
        The number of time periods which a predicted range is allowed lag after an
        anomaly, for which it will be marked as a "good" raise. For example, if the
        actual range is `range(5,7)` and the predicted range is `range(8, 9)`, this prediction
        will be deemed accurate with a buffer length of 2 or above.

    Returns
    -------
    precision: float
        Precision. Ratio of predicted ranges that overlap with an (extended) anomaly range.
    recall: float
        Recall. Ratio of (extended) anomaly ranges that were caught by (overlaps with)
        at least one prediction range.
    """
    if len(real_ranges) == 0 and len(pred_ranges) == 0:
        return 1.0, 1.0
    if len(pred_ranges) == 0:
        return 1.0, 0.0
    if len(real_ranges) == 0:
        return 0.0, 1.0

    extended_ranges = extend_ranges(real_ranges, buffer_length)

    recall_tp, fn = 0, 0
    for tr in extended_ranges:
        # any labels raised this target range?
        raised = any(range_overlap(tr, pr) for pr in pred_ranges)
        recall_tp += raised
        fn += 1 - raised

    precision_tp, fp = 0, 0
    for pr in pred_ranges:
        # any anomalies caught by the label?
        caught = any(range_overlap(tr, pr) for tr in extended_ranges)
        precision_tp += caught
        fp += 1 - caught

    return precision_tp / len(pred_ranges), recall_tp / len(extended_ranges)
