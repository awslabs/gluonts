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

import numpy as np
import numpy.typing as npt
import scipy.stats as st


def nrmse(
    y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Computes the normalized root mean squared error of the predictions.

    Args:
        y_pred: Array of shape [N, D] with the predictions (N: number of elements, D: number of
            metrics).
        y_true: Array of shape [N, D] with the true metrics.

    Returns:
        Array of shape [D] with the average NRMSE for each metric.
    """
    rmse = np.sqrt((y_pred - y_true) ** 2).mean(0)
    return rmse / np.abs(y_true).mean(0)


def smape(
    y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Computes the symmetric mean absolute percentage error of the predictions.

    Args:
        y_pred: Array of shape [N, D] with the predictions (N: number of elements, D: number of
            metrics).
        y_true: Array of shape [N, D] with the true metrics.

    Returns:
        Array of shape [D] with the average sMAPE for each metric.
    """
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_pred) + np.abs(y_true)) / 2
    return 100 * (num / denom).mean(0)


def mrr(
    y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Computes the mean reciprocal rank of the predictions.

    Args:
        y_pred: Array of shape [N, D] with the predictions (N: number of elements, D: number of
            metrics).
        y_true: Array of shape [N, D] with the true metrics.

    Returns:
        Array of shape [D] with the average MRR for each metric.
    """
    minimum_indices = y_pred.argmin(0)  # [D]
    true_ranks = st.rankdata(y_true, method="min", axis=0)  # [N, D]
    ranks = np.take_along_axis(
        true_ranks, minimum_indices[None, :], axis=0
    )  # [N, D]
    result = 1 / ranks
    return result.mean(0)


def precision_k(
    k: int, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Computes the precision@k of the predictions.

    Args:
        k: The number of items that are looked at for computing the precision.
        y_pred: Array of shape [N, D] with the predictions (N: number of elements, D: number of
            metrics).
        y_true: Array of shape [N, D] with the true metrics.

    Returns:
        Array of shape [D] with the precisions@k for each metric.
    """
    pred_ranks = st.rankdata(y_pred, method="ordinal", axis=0) - 1  # [N, D]
    true_ranks = st.rankdata(y_true, method="ordinal", axis=0) - 1  # [N, D]

    pred_relevance = (pred_ranks < k) / k  # [N, D]
    true_relevance = true_ranks < k  # [N, D]

    return (pred_relevance * true_relevance).sum(0)


def ndcg(
    y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Computes the normalized discounted cumulative gain of the predictions.

    Args:
        y_pred: Array of shape [N, D] with the predictions (N: number of elements, D: number of
            metrics).
        y_true: Array of shape [N, D] with the true metrics.

    Returns:
        Array of shape [D] with the nDCG for each metric.
    """
    n = y_pred.shape[0]

    # First, get the relevance
    true_argsort = y_true.argsort(0)  # [N, D]
    true_ranks = true_argsort.argsort(0)  # [N, D]
    relevance = 1 / (true_ranks + 1)  # [N, D]
    relevance[true_ranks >= 10] = 0

    # Then, compute the iDCG
    num = np.take_along_axis(relevance, true_argsort, axis=0)  # [N, D]
    denom = np.log2(np.arange(n) + 2)[:, None]  # [N, D]
    idcg = (num / denom).sum(0)

    # And compute the DCG
    pred_argsort = y_pred.argsort(0)
    num = np.take_along_axis(relevance, pred_argsort, axis=0)
    denom = np.log2(np.arange(n) + 2)[:, None]
    dcg = (num / denom).sum(0)

    return dcg / idcg
