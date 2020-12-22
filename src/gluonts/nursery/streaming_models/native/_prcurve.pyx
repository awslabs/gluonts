# cython: language_level=3, boundscheck=False
from functools import partial

import cython
import numpy as np

cimport numpy as cnp

from typing import Callable, List, Tuple

from cpython.mem cimport PyMem_Free, PyMem_Malloc

# bias function type
ctypedef int (*pyx_bias_fptr)(const int, const int)
# ctypedef int (*pyx_bias_fptr)(int, int) except -1

# --------------------------------------------------------
# utilities for working with range objects
# --------------------------------------------------------

# struct that shadows a Python `rangeobject`
ctypedef struct pyx_range:
    int start
    int stop
    int step
    int length


cdef pyx_range _convert_range(range_in: range):
    """
    Helper function to converts a Python range object to a
    `pyx_range`.
    """
    cdef pyx_range rval
    rval = pyx_range(
        start = range_in.start,
        stop = range_in.stop,
        step = range_in.step,
        length = len(range_in)
    )
    return rval


@cython.wraparound(False)
cdef void _convert_ranges(
    ranges: List[range],
    pyx_range *out_ranges_arr,
    int out_ranges_size
):
    """
    Convert python ranges to pyx_ranges, and fill them to given
    array. Assumes `len(ranges) == out_ranges_size`, though this is
    not explicitly asserted for performance reasons. 
    
    Parameters
    ----------
    ranges
        List of python ranges to be converted to pyx_ranges
    out_ranges_arr
    out_ranges_size
    """
    cdef:
        int i = 0

    # len(ranges) == ref_ranges_size assumed
    for i in range(out_ranges_size):
        curr_range = ranges[i]

        out_ranges_arr[i] = pyx_range(
            start = curr_range.start,
            stop = curr_range.stop,
            step = curr_range.step,
            length = len(curr_range)
        )


@cython.cdivision(True)
cdef int _index_in_pyx_range(const int index, const pyx_range *target):
    """
    Assert that an index is in the given range. 
    """
    cdef int diff = index - target.start
    if diff < 0:
        return 0
    if (diff / target.step) < target.length:
        return 1
    return 0


@cython.wraparound(False)
def pyx_labels_to_ranges(
    labels: List[bool]
) -> List[range]:
    """
    Compiled version of `range_precision_recall.labels_to_ranges`.
    """
    ranges_ls = []
    cdef:
        int ix
        int num_labels = len(labels)
        int start_ix = -1
        int stop_ix = -1

    for ix in range(num_labels):
        label = labels[ix]
        if label:
            # this might be the last positive label in the anomaly range
            stop_ix = ix + 1
            if start_ix == -1:
                # a new positive label is seen
                start_ix = ix
        elif start_ix != -1:
            # a consecutive sequence of positive labels is ended
            assert stop_ix != -1
            ranges_ls.append(range(start_ix, stop_ix))
            start_ix = -1

    if start_ix != -1:
        # the last element of `labels` is True, we never ended that range
        assert stop_ix != -1
        ranges_ls.append(range(start_ix, stop_ix))

    return ranges_ls


# --------------------------------------------------------
# cython equivalents of helpers, bias functions, etc
# --------------------------------------------------------

@cython.cdivision(True)
cdef inline double _gamma_fn(const int x):
    return 1 / <double>x


cdef inline int _flat_bias(const int position, const int anomaly_length):
    # assert 1 <= position <= anomaly_length, \
    #     "position must be larger than or equal to one and smaller than or equal to the length of the anomaly."
    return 1


cdef inline int _front_end_bias(const int position, const int anomaly_length):
    # assert 1 <= position <= anomaly_length, \
    #     "position must be larger than or equal to one and smaller than or equal to the length of the anomaly."
    return anomaly_length - position + 1


cdef inline int _back_end_bias(const int position, const int anomaly_length):
    # assert 1 <= position <= anomaly_length, \
    #     "position must be larger than or equal to one and smaller than or equal to the length of the anomaly."
    return position


@cython.cdivision(True)
cdef inline int _middle_bias(const int position, const int anomaly_length):
    # assert 1 <= position <= anomaly_length, \
    #     "position must be larger than or equal to one and smaller than or equal to the length of the anomaly."
    if position <= anomaly_length / 2:
        return position
    else:
        return anomaly_length - position + 1


@cython.cdivision(True)
cdef double _overlap_size(
    const pyx_range *target_range,
    const pyx_range *overlap_set,
    pyx_bias_fptr positional_bias_fn,
):
    cdef:
        int i
        int bias
        double my_value = 0
        double max_value = 0
        int anomaly_length = target_range.length

    for i in range(1, anomaly_length + 1):
        bias = positional_bias_fn(i, anomaly_length)
        max_value += bias
        curr_target_index = (
            target_range.start + (i - 1) * target_range.step
        )
        if _index_in_pyx_range(curr_target_index, overlap_set):
            my_value += bias
    return my_value / max_value


# -----------------------------------------------------
# ports of RangeMetric static functions and subclass
# (RangePrecision, RangeRecall) methods
# -----------------------------------------------------

cdef int _rm__range_intersects(
    const pyx_range *left_range,
    const pyx_range *right_range,
):
    """
    Return 1 if two ranges of step size 1 intersect.
    """
    # FIXME: this function assumes steps of input ranges are 1
    cdef:
        int start = max(left_range.start, right_range.start)
        int stop = min(left_range.stop, right_range.stop)
    if stop - start > 0:
        return 1
    return 0


cdef int _rm__num_overlaps(
    const pyx_range *target_range,
    const pyx_range *ref_ranges,
    const int ref_ranges_size,
    int *first_overlap_out
):
    """
    Compute the number of overlaps between a target range and 
    an array of reference ranges. This function assumes that 
    the reference ranges are in ascending sorted order (the 
    start of the subsequent range is greater than the stop of the
    preceding range).  
    
    Parameters
    ----------
    target_range
        pointer to target range
    ref_ranges
        array of reference ranges
    ref_ranges_size
        length of reference ranges
    first_overlap_out
        pointer to address where the index of the first
        overlapping range is written. this helps the caller
        when iterating over overlapping ranges. 

    Returns
    -------
    num_overlaps: int
        the number of overlaps
    """
    cdef:
        int i = 0
        int num_overlaps = 0
        int first_encountered = 0
        int target_start = target_range.start
        int target_stop = target_range.stop

    for i in range(ref_ranges_size):
        if ref_ranges[i].start >= target_stop:
            break
        if ref_ranges[i].stop < target_start:
            continue
        if _rm__range_intersects(target_range, &ref_ranges[i]) > 0:
            num_overlaps += 1
            if not first_encountered:
                first_overlap_out[0] = i
                first_encountered = 1

    return num_overlaps


cdef double _rm_existence_reward(
    const pyx_range *target_range,
    const pyx_range *ref_ranges,
    const int ref_ranges_size
):
    cdef:
        int i = 0
        int target_start = target_range.start

    for i in range(ref_ranges_size):
        if ref_ranges[i].stop < target_start:
            continue
        if _rm__range_intersects(target_range, &ref_ranges[i]) > 0:
            return 1.0

    return 0.0


cdef double _rm_overlap_reward(
    const pyx_range *target_range,
    const pyx_range *ref_ranges,
    const int ref_ranges_size
):
    """
    Cython port of RangeMetric.overlap_reward that fixes the overlap_size_fn and positional_bias_fn
    arguments to pyx_overlap_size and pyx__flat_bias respectively.
    """
    cdef:
        int i
        int num_overlaps
        int first_overlap = 0
        int target_start = target_range.start
        int target_stop = target_range.stop
        double cardinality_factor = 0
        double total_overlap_size = 0

    # Eq. 7 in the paper.
    num_overlaps = _rm__num_overlaps(
        target_range, ref_ranges, ref_ranges_size, &first_overlap
    )
    cardinality_factor = _gamma_fn(num_overlaps) if num_overlaps > 1 else 1.0

    for i in range(first_overlap, first_overlap + num_overlaps):
        total_overlap_size += _overlap_size(
            target_range, &ref_ranges[i], _flat_bias
        )

    return cardinality_factor * total_overlap_size


cdef double _rm__recall_of_single_target_range(
        const pyx_range *target_range,
        const pyx_range *pred_ranges,
        const int pred_ranges_size,
        const double alpha
    ):
        cdef:
            double existence_reward
            double overlap_reward

        existence_reward = _rm_existence_reward(
            target_range, pred_ranges, pred_ranges_size
        )
        overlap_reward = _rm_overlap_reward(
            target_range, pred_ranges, pred_ranges_size
        )

        return alpha * existence_reward + (1.0 - alpha) * overlap_reward


cdef double _rm__precision_of_single_target_range(
        const pyx_range *target_range,
        const pyx_range *real_ranges,
        const int real_ranges_size
    ):
        return _rm_overlap_reward(
            target_range, real_ranges, real_ranges_size
        )


@cython.wraparound(False)
def pyx_calculate_rewards(
    target_range: range,
    ref_ranges: List[range],
) -> Tuple[float, float]:
    cdef:
        int ref_ranges_size = len(ref_ranges)
        pyx_range target_range_pyx
        pyx_range* ref_ranges_arr

    target_range_pyx = _convert_range(target_range)
    ref_ranges_arr = <pyx_range *>PyMem_Malloc(ref_ranges_size * cython.sizeof(pyx_range))

    _convert_ranges(ref_ranges, ref_ranges_arr, ref_ranges_size)

    existence_reward = _rm_existence_reward(
        &target_range_pyx, ref_ranges_arr, ref_ranges_size
    )
    overlap_reward = _rm_overlap_reward(
        &target_range_pyx, ref_ranges_arr, ref_ranges_size
    )

    PyMem_Free(ref_ranges_arr)

    return existence_reward, overlap_reward


@cython.cdivision(True)
def pyx_range_precision_recall(
    real_ranges: List[range],
    pred_ranges: List[range],
    double recall_alpha,
    # positional_bias_fn: Callable,  # TODO
) -> Tuple[float, float, float, float]:
    """
    Given a list of python ranges, compute the range precision and recall
    values.

    Parameters
    ----------
    real_ranges
    pred_ranges
    recall_alpha

    Returns
    -------
    precision
    recall
    """
    cdef:
        int i, j
        double recall = 0.0
        double precision = 0.0
        pyx_range *real_ranges_arr
        pyx_range *pred_ranges_arr
        int real_ranges_size = len(real_ranges)
        int pred_ranges_size = len(pred_ranges)

    # TODO: assert all step 1 and start stop ok -- real_ranges
    # TODO: assert all step 1 and start stop ok -- pred_ranges

    real_ranges_arr = <pyx_range *>PyMem_Malloc(real_ranges_size * cython.sizeof(pyx_range))
    _convert_ranges(real_ranges, real_ranges_arr, real_ranges_size)

    pred_ranges_arr = <pyx_range *>PyMem_Malloc(pred_ranges_size * cython.sizeof(pyx_range))
    _convert_ranges(pred_ranges, pred_ranges_arr, pred_ranges_size)

    for i in range(real_ranges_size):
        recall += _rm__recall_of_single_target_range(
            &real_ranges_arr[i],
            pred_ranges_arr,
            pred_ranges_size,
            recall_alpha
        )

    for j in range(pred_ranges_size):
        precision += _rm__precision_of_single_target_range(
            &pred_ranges_arr[j],
            real_ranges_arr,
            real_ranges_size
        )

    PyMem_Free(real_ranges_arr)
    PyMem_Free(pred_ranges_arr)

    return (
        precision / pred_ranges_size if pred_ranges_size > 0 else 0.0,
        recall / real_ranges_size if real_ranges_size > 0 else 0.0,
        pred_ranges_size,
        real_ranges_size
    )


@cython.cdivision(True)
def pyx_singleton_precision_recall(
    cnp.ndarray[dtype=cnp.float64_t] true_labels,
    cnp.ndarray[dtype=cnp.float64_t] pred_labels,
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
    cdef:
        double precision = 0, recall = 0
        double tp, true_cond_p, pred_cond_p

    tp = np.sum(true_labels * pred_labels)
    true_cond_p = np.sum(true_labels)
    pred_cond_p = np.sum(pred_labels)

    if pred_cond_p > 0:
        precision = tp / pred_cond_p
    if true_cond_p > 0:
        recall = tp / true_cond_p

    return precision, recall


def pyx_precision_recall_curve(
    true_labels: np.array,
    pred_scores: np.array,
    thresholds: np.array = None,
    precision_recall_fn: Callable = partial(
        pyx_range_precision_recall,
        recall_alpha=0.0
    )
):
    """
    A cython version of `havel_anomaly_scoring.range_precision_recall.range_precision_recall_curve`,
    that does not implement `singleton_ranges` option.

    See the Python function docs for further info.
    """

    true_ranges = pyx_labels_to_ranges(true_labels)

    thresholds = thresholds if thresholds is not None else np.unique(pred_scores)
    precisions = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))

    for ix, score in enumerate(thresholds):
        pred_ranges = pyx_labels_to_ranges(pred_scores >= score)
        precisions[ix], recalls[ix], _, _ = precision_recall_fn(true_ranges, pred_ranges)

    # Start from the latest threshold where the full recall is attained.
    perfect_recall_ixs = np.where(recalls == 1.0)[0]
    first_ind = perfect_recall_ixs[-1] if len(perfect_recall_ixs) > 0 else 0
    return np.r_[precisions[first_ind:], 1], np.r_[recalls[first_ind:], 0], thresholds[first_ind:]
