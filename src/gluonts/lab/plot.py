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

from typing import Optional, Collection, Tuple

import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from gluonts.model.forecast import SampleForecast

from gluonts.exceptions import GluonTSUserError

from gluonts.model import Forecast
from matplotlib import pyplot as plt


def get_percentiles(prediction_intervals):
    percentiles_list = [50.0] + [
        50.0 + sign * percentile / 2.0
        for percentile in prediction_intervals
        for sign in [-1.0, +1.0]
    ]
    return sorted(set(percentiles_list))


def _plot_forecast(
    forecast: Forecast,
    prediction_intervals: Collection[float],
    ax: plt.axis,
    show_mean: bool,
    color: str,
    prefix: str,
    *args,
    **kwargs,
):
    interval_count = len(prediction_intervals)

    percentiles = get_percentiles(prediction_intervals)
    predictions = [forecast.quantile(p / 100.0) for p in percentiles]

    if show_mean:
        if not isinstance(forecast, SampleForecast):
            raise GluonTSUserError(
                "Plotting the mean only works only with SampleForecast, "
                f"got {type(forecast)} instead"
            )

        mean_data = np.mean(forecast.samples, axis=0)
        ax.plot(
            forecast.index.to_timestamp(),
            mean_data,
            color=color,
            ls=":",
            label=f"{prefix}mean prediction",
            *args,
            **kwargs,
        )

    # median prediction
    p50_data = predictions[interval_count]
    p50_series = pd.Series(data=p50_data, index=forecast.index.to_timestamp())
    p50_series.plot(color=color, ls="-", label=f"{prefix}median prediction")

    # percentile prediction intervals
    alphas_lower_half = [(p / 100.0) ** 0.3 for p in percentiles]
    alphas = alphas_lower_half + alphas_lower_half[::-1]
    for i in range(interval_count):
        # plot lower and upper half of median individually to keep colors true
        label = f"{prefix}{100 - percentiles[i] * 2}% prediction interval"
        area_info = [
            {"label": label, "idx": i},  # give label only once
            {"label": None, "idx": interval_count * 2 - i - 1},
        ]
        for info in area_info:
            plt.fill_between(
                forecast.index.to_timestamp(),
                predictions[info["idx"]],
                predictions[info["idx"] + 1],
                facecolor=color,
                alpha=alphas[i],
                interpolate=True,
                label=info["label"],
                *args,
                **kwargs,
            )


def plot(
    forecast: Optional[Forecast],
    timeseries: Optional[pd.Series] = None,
    prediction_intervals: Collection[float] = (50.0, 90.0),
    show_mean: bool = False,
    color: str = "g",
    label: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    output_file: Optional[str] = None,
    show_plot: bool = True,
    show_grid: bool = True,
    figsize: Tuple[int] = (10, 10),
    legend_location: str = "upper left",
    date_format: Optional[str] = None,
    train_test_separator=None,
    *args,
    **kwargs,
):
    for c in prediction_intervals:
        if not 0.0 <= c <= 100.0:
            raise GluonTSUserError(
                f"Prediction interval {c} is not between 0 and 100"
            )

    prefix = "" if label is None else label + "-"

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if timeseries is not None:
        ax.plot(
            timeseries.to_timestamp().index,
            timeseries.values,
            label=f"{prefix}target",
        )

    if forecast is not None:
        _plot_forecast(
            forecast=forecast,
            prediction_intervals=prediction_intervals,
            ax=ax,
            show_mean=show_mean,
            color=color,
            prefix=prefix,
            *args,
            **kwargs,
        )

    if train_test_separator is not None:
        ax.axvline(train_test_separator, color="r")

    ax.legend(loc=legend_location)
    if xlabel is not None:
        ax.xlabel(xlabel)
    if ylabel is not None:
        ax.ylabel(ylabel)

    if show_grid:
        ax.grid(which="both")
    if date_format is not None:
        ax.xaxis.set_major_formatter(DateFormatter(date_format))

    if output_file:
        fig.savefig(output_file)

    if show_plot:
        fig.show()
