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

# pylint: disable=missing-function-docstring
import math
import numpy as np
import numpy.ma as ma
from typing import List, Dict

from meta.datasets.splits import EvaluationDataset


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def abs_error_sum(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.abs(y_pred - y_true).sum()


def abs_target_sum(y_true: np.ndarray) -> float:
    return np.sum(np.abs(y_true))


def abs_target_mean(y_true: np.ndarray) -> float:
    return np.mean(np.abs(y_true))


def naive_error(y_past: ma.masked_array, seasonality: int) -> np.ndarray:
    error = np.abs(y_past[:, seasonality:] - y_past[:, :-seasonality]).mean(1)
    return ma.getdata(error)


def mase(
    y_pred: np.ndarray, y_true: np.ndarray, error: np.ndarray
) -> np.ndarray:
    mase_values = np.abs(y_pred - y_true).mean(1) / error
    return mase_values.mean()


def smape(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    median = y_pred
    num = np.abs(y_true - median)
    denom = (np.abs(y_true) + np.abs(median)) / 2
    # If the denominator is 0, we set it to float('inf') such that any division yields 0 (this
    # might not be fully mathematically correct, but at least we don't get NaNs)
    denom[denom == 0] = math.inf
    return np.mean(num / denom, axis=1).mean()


def mean_weighted_quantile_loss(
    y_pred: np.ndarray, y_true: np.ndarray, quantiles: List[str]
) -> float:
    y_true_rep = y_true[:, None].repeat(len(quantiles), axis=1)
    quantiles = np.array([float(q) for q in quantiles])
    quantile_losses = 2 * np.sum(
        np.abs(
            (y_pred - y_true_rep)
            * ((y_true_rep <= y_pred) - quantiles[:, None])
        ),
        axis=-1,
    )  # shape [num_time_series, num_quantiles]
    denom = np.sum(np.abs(y_true))  # shape [1]
    weighted_losses = quantile_losses.sum(0) / denom  # shape [num_quantiles]
    return weighted_losses.mean()


def compute_metrics(
    forecasts: np.ndarray,
    dataset: EvaluationDataset,
    quantiles: List[str],
    seasonality: int,
) -> Dict[str, float]:
    """
    Evaluates the forecasts on the provided dataset and returns summary metrics.

    Parameters
    ----------
    forecasts: np.ndarray, shape: [n_time_series, prediction_length, n_quantiles]
        The per time-series forecasts. The forecasts *must* align with the time series of the
        given dataset. Otherwise, behavior is undefined.
    dataset: EvaluationDataset
        The dataset for which to evaluate the metrics.
    quantiles: List[str]
        A list containing the quantiles predicted in the forecast
    seasonality: int
        The seasonality of the data.

    Returns
    -------
    Dict
        The evaluation of the forecasts.
    """
    assert len(forecasts) > 0, "At least one forecast must be given."
    assert len(forecasts) == len(
        dataset.future
    ), "The number of forecasts does not match the number of time series in the dataset."
    median = forecasts[:, :, forecasts.shape[-1] // 2]
    # Compute seasonal error for MASE computation
    seasonal_error = naive_error(dataset.past, seasonality)

    # Compute all the metrics
    results = {
        # normalized deviation (see e.g. DeepAR paper)
        "nd": abs_error_sum(median, dataset.future)
        / abs_target_sum(dataset.future),
        "nrmse": rmse(median, dataset.future)
        / abs_target_mean(dataset.future),
        "mase": mase(median, dataset.future, seasonal_error),
        "smape": smape(median, dataset.future),
        "mean_weighted_quantile_loss": mean_weighted_quantile_loss(
            forecasts.transpose((0, 2, 1)),
            dataset.future,
            quantiles=quantiles,
        ),
    }
    return results
