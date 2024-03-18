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

from typing import List, Optional, Collection, Union

import pandas as pd
from matplotlib import pyplot as plt

from gluonts.model import Forecast
from gluonts.exceptions import GluonTSUserError


def get_percentiles(prediction_intervals):
    percentiles_list = [50.0] + [
        50.0 + sign * percentile / 2.0
        for percentile in prediction_intervals
        for sign in [-1.0, +1.0]
    ]
    return sorted(set(percentiles_list))


def plot_forecast(
    forecast: Forecast,
    axis: plt.axis,
    prediction_intervals: Collection[float],
    plot_mean: bool,
    color: str,
    marker: Optional[str] = None,
    label_prefix: str = "",
    label_suffix: str = "",
):
    """Helper function for plotting a single forecast

    Parameters
    ----------
    axis
        plt.axis to plot on
    forecast
        Forecast to plot
    ...
    """

    interval_count = len(prediction_intervals)

    percentiles = get_percentiles(prediction_intervals)
    predictions = [forecast.quantile(p / 100.0) for p in percentiles]

    if plot_mean:
        axis.plot(
            forecast.index.to_timestamp(),
            forecast.mean,
            color=color,
            ls=":",
            label=f"{label_prefix}mean prediction{label_suffix}",
            marker=marker,
        )

    # median prediction
    p50_data = predictions[interval_count]
    p50_series = pd.Series(data=p50_data, index=forecast.index.to_timestamp())
    label = f"{label_prefix}median prediction"
    axis.plot(
        p50_series, label=label, linestyle="--", color=color, marker=marker
    )

    # percentile prediction intervals
    alphas_lower_half = [(p / 100.0) ** 0.3 for p in percentiles]
    alphas = alphas_lower_half + alphas_lower_half[::-1]
    for interval_idx in range(interval_count):
        p = 100 - percentiles[interval_idx] * 2
        label = f"{label_prefix}{p}% prediction interval"

        # plot lower and upper half of median individually to keep colors true
        area_info = [
            {"label": label, "idx": interval_idx},  # give label only once
            {"label": None, "idx": interval_count * 2 - interval_idx - 1},
        ]
        for info in area_info:
            axis.fill_between(
                forecast.index.to_timestamp(),
                predictions[info["idx"]],
                predictions[info["idx"] + 1],
                facecolor=color,
                alpha=alphas[interval_idx],
                interpolate=True,
                label=info["label"],
            )


def read_input_for_marker_or_color(
    value: Union[str, List[str]], entry_count: int, entity_name: str
):
    """normalize input for marker/color into list of length `entry_count`"""
    result = value
    if isinstance(result, str):
        result = [result]
    if len(result) == 0:
        raise GluonTSUserError(f"'{entity_name}' can't be empty list")

    # repeat if necesarry to match entry_count
    result_idx = 0
    while len(result) < entry_count:
        result.append(result[result_idx])
        result_idx += 1

    return result
