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

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from tsbench.config.dataset import EvaluationDataset
from tsbench.evaluations.metrics import Metric, Performance
from .metrics import (
    abs_error_sum,
    abs_target_mean,
    abs_target_sum,
    mase,
    naive_error,
    ncrps,
    rmse,
    smape,
)
from .quantile import QuantileForecasts


def evaluate_forecasts(
    forecasts: QuantileForecasts, data: EvaluationDataset
) -> Evaluation:
    """
    Evaluates the forecasts on the provided dataset and returns the metrics
    averaged over all time series.

    Args:
        forecasts: The per time series forecasts. The forecasts *must* align with the time series
            of the given dataset. Otherwise, behavior is undefined.
        dataset: The dataset for which to evaluate the metrics.

    Returns:
        The evaluation of the forecasts.
    """
    assert len(forecasts) > 0, "At least one forecast must be given."
    assert len(forecasts) == len(data.future), (
        "The number of forecasts does not match the number of time series in"
        " the dataset."
    )

    # Compute seasonal error for MASE computation
    seasonal_error = naive_error(data.past, forecasts.seasonality)

    # Compute all the metrics
    results = {
        "nd": abs_error_sum(forecasts.median, data.future)
        / abs_target_sum(data.future),
        "nrmse": rmse(forecasts.median, data.future)
        / abs_target_mean(data.future),
        "mase": mase(forecasts.median, data.future, seasonal_error),
        "smape": smape(forecasts.median, data.future),
        "ncrps": ncrps(forecasts, data.future),
    }

    return Evaluation(results)


# -------------------------------------------------------------------------------------------------


@dataclass
class Evaluation:
    """
    An evaluation instance provides metrics per time series as well as overall
    metrics.
    """

    summary: dict[str, float]
    """
    The metrics summarizing the overall performance of the model.
    """

    @classmethod
    def performance(cls, evaluations: list[Evaluation]) -> Performance:
        """
        Aggregates the provided evaluations into a single performance object.

        Args:
            evaluations: The evaluations.

        Returns:
            The performance object. Since this is not part of the evaluation, it has the
                `num_model_parameters` attribute unset (set to zero).
        """
        metrics = [e.summary for e in evaluations]
        kwargs = {
            m: (
                Metric(0, 0)
                if m == "num_model_parameters"
                else Metric(
                    np.mean(
                        [
                            metric[m] if m in metric else np.nan
                            for metric in metrics
                        ]
                    ),
                    np.std(
                        [
                            metric[m] if m in metric else np.nan
                            for metric in metrics
                        ]
                    ),
                )
            )
            for m in Performance.metrics()
        }
        return Performance(**kwargs)  # type: ignore
