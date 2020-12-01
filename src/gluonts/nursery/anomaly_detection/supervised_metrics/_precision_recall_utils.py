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

from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from . import buffered_precision_recall
from .utils import labels_to_ranges


class PrecisionRecallAndWeights(NamedTuple):
    precisions: np.array
    recalls: np.array
    precision_weights: np.array
    recall_weights: np.array


def singleton_precision_recall(
    true_labels,
    pred_labels,
) -> Tuple[float, float]:
    """

    Parameters
    ----------
    true_labels
        Binary array of true labels
    pred_labels
        Binary array of predicted labels

    Returns
    -------
    precision: float
    recall: float
    """
    precision = 0.0
    recall = 0

    tp = np.sum(true_labels * pred_labels)
    true_cond_p = np.sum(true_labels)
    pred_cond_p = np.sum(pred_labels)

    if pred_cond_p > 0:
        precision = tp / pred_cond_p
    if true_cond_p > 0:
        recall = tp / true_cond_p

    return precision, recall


def precision_recall_curve_per_ts(
    labels: List[bool],
    scores: List[float],
    thresholds: np.array,
    partial_filter: Optional[Callable] = None,
    singleton_curve: bool = False,
    precision_recall_fn: Callable = buffered_precision_recall,
) -> PrecisionRecallAndWeights:
    true_ranges = labels_to_ranges(labels)
    precisions = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))

    precision_weights, recall_weights = (
        np.zeros(len(thresholds)),
        np.zeros(len(thresholds)),
    )

    for ix, th in enumerate(thresholds):

        if partial_filter is None:
            pred_labels = scores >= th
        else:
            pred_labels = partial_filter(th)

        if singleton_curve:
            true_labels_np = np.array(labels, dtype=float)
            pred_labels_np = np.array(pred_labels, dtype=float)
            _prec, _reca = singleton_precision_recall(
                true_labels_np, pred_labels_np
            )
            _prec_w, _reca_w = np.sum(pred_labels_np), np.sum(true_labels_np)
        else:
            pred_ranges = labels_to_ranges(pred_labels)
            _prec, _reca = precision_recall_fn(true_ranges, pred_ranges)
            _prec_w, _reca_w = len(pred_ranges), len(true_ranges)

        precisions[ix] += _prec * _prec_w
        recalls[ix] += _reca * _reca_w

        precision_weights[ix] += _prec_w
        recall_weights[ix] += _reca_w

    return PrecisionRecallAndWeights(
        precisions, recalls, precision_weights, recall_weights
    )


def aggregate_precision_recall_curve(
    label_score_iterable: Iterable,
    thresholds: Optional[np.array] = None,
    partial_filter: Optional[Callable] = None,
    singleton_curve: bool = False,
    precision_recall_fn: Callable = buffered_precision_recall,
    n_jobs: int = -1,
):
    """
    Computes aggregate range-based precision recall curves over a data set, iterating over
    individual time series. Optionally takes partially constructed filter that converts given scores/thresholds to
    anomaly labels. See `gluonts.nursery.anomaly_detection.supervised_metrics.filters` for example filters.

    Parameters
    ----------
    label_score_iterable: Iterable
        An iterable that gives 2-tuples of np.arrays (of identical length),
        corresponding to `true_labels` and `pred_scores` respectively.
    thresholds: np.array
        An np.array of score thresholds for which to compute precision recall values.
        If the filter_type argument is provided, these are the threshold values of
        the filter. If not, they will be applied as a single step hard threshold to
        predicted scores.
    partial_filter: Callable
        Partial constructor for a "filter" object. If provided, this function can be called with a "score_threshold" to
        return labels used for precision and recall computation. If not provided, labels will be assigned with a hard
        threshold.
        See `gluonts.nursery.anomaly_detection.supervised_metrics.filters` for example filters.
    singleton_curve: bool
        If true, range-based precision recall will not be computed
    precision_recall_fn:
        Function to call in order to get the precision, recall metrics.
    n_jobs: int
        How many concurrent threads for parallelization, default is -1 (use all cpu available)

    Returns
    -------
    (Same as output of `sklearn.metrics.precision_recall_curve`)
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : array, shape = [n_thresholds <= len(np.unique(scores))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    if thresholds is None:
        thresholds = np.unique(
            np.concatenate([scores for _, scores in label_score_iterable])
        )

    all_metrics = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(precision_recall_curve_per_ts)(
            labels,
            scores,
            thresholds,
            partial_filter,
            singleton_curve,
            precision_recall_fn,
        )
        for labels, scores in label_score_iterable
    )
    (
        all_precisions,
        all_recalls,
        all_precision_weights,
        all_recall_weights,
    ) = zip(*all_metrics)

    precisions = np.sum(all_precisions, axis=0)
    recalls = np.sum(all_recalls, axis=0)
    precision_weights = np.sum(all_precision_weights, axis=0)
    recall_weights = np.sum(all_recall_weights, axis=0)

    # normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = np.where(
            precision_weights > 0, precisions / precision_weights, 0.0
        )
        recalls = np.where(recall_weights > 0, recalls / recall_weights, 0.0)

    # Start from the latest threshold where the full recall is attained.
    perfect_recall_ixs = np.where(recalls == 1.0)[0]
    first_ind = perfect_recall_ixs[-1] if len(perfect_recall_ixs) > 0 else 0
    return (
        np.r_[precisions[first_ind:], 1],
        np.r_[recalls[first_ind:], 0],
        thresholds[first_ind:],
    )


def aggregate_precision_recall(
    labels_pred_iterable: Iterable,
    precision_recall_fn: Callable = buffered_precision_recall,
) -> Tuple[float, float]:
    """
    Computes aggregate range-based precision recall metrics for the given prediction labels.

    Parameters
    ----------
    labels_pred_iterable
        An iterable that gives 2-tuples of boolean lists corresponding to `true_labels` and
        `pred_labels` respectively.
    precision_recall_fn
        Function to call in order to get the precision, recall metrics.

    Returns
    -------
    A tuple containing average precision and recall in that order.
    """
    total_prec, total_reca, total_prec_w, total_reca_w = 0.0, 0.0, 0.0, 0.0
    for true_labels, pred_labels in labels_pred_iterable:
        true_ranges = labels_to_ranges(true_labels)
        pred_ranges = labels_to_ranges(pred_labels)

        _prec, _reca = precision_recall_fn(true_ranges, pred_ranges)
        _prec_w, _reca_w = len(pred_ranges), len(true_ranges)
        total_prec += _prec * _prec_w
        total_prec_w += _prec_w

        total_reca += _reca * _reca_w
        total_reca_w += _reca_w

    return (
        total_prec / total_prec_w if total_prec_w > 0 else 0,
        total_reca / total_reca_w if total_reca_w > 0 else 0,
    )
