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

import math
from typing import List, Literal, Optional
import numpy as np
import scipy.special as sps
from .quantile import QuantileForecasts

EnsembleWeighting = Literal["relative", "softmax", "uniform"]
"""
The strategy for weighing ensembles:

- `relative`: Models are weighed according to their relative mean weighted quantile loss (i.e. if
    a model's loss is twice as high, it is weighed half as much).
- `softmax`: Models are weighed according to the rank of their mean weighted quantile loss (i.e.
    a lower loss receives the highest weight).
- `uniform`: Models are weighed uniformly.
"""


def ensemble_forecasts(
    forecasts: List[QuantileForecasts],
    weighting: EnsembleWeighting = "uniform",
    ncrps: Optional[List[float]] = None,
) -> QuantileForecasts:
    """
    Ensembles the provided forecasts by computing a weighted average across
    quantile and time steps.

    Args:
        forecasts: The forecasts to ensemble. Each list item should contain the forecasts of a
            single ensemble member.
        weighting: The kind of weighting to apply for computing the weighted average.
        ncrps: The average nCRPS values of the forecasts of the individual ensemble members. Must
            be provided if `weighting` is not set to "uniform".

    Returns:
        The averaged forecasts.
    """
    # First, compute the weights via the provided quantile losses
    if weighting == "relative":
        losses = np.array(ncrps)
        factors = losses.max() / losses
        weights = factors / factors.sum()
        weights = weights.tolist()
    elif weighting == "softmax":
        losses = np.array(ncrps)
        ranks = losses.argsort().argsort()
        weights = sps.softmax(-ranks).tolist()
    else:
        n = len(forecasts)
        weights = [1 / n] * n

    # Some assertions
    ref = forecasts[0]
    assert math.isclose(
        sum(weights), 1, abs_tol=1e-7
    ), "The ensembling weights do not sum to 1."
    assert all(
        len(forecast) == len(ref) for forecast in forecasts
    ), "The different forecasts do not provide equally many values."

    # Then, compute the weighted average
    weighted_average = np.stack(
        [forecasts[i].values * weights[i] for i in range(len(forecasts))],
        axis=0,
    ).sum(0)

    # And return the quantile forecast
    return QuantileForecasts(
        values=weighted_average,
        start_dates=ref.start_dates,
        item_ids=ref.item_ids,
        freq=ref.freq,
        quantiles=ref.quantiles,
    )
