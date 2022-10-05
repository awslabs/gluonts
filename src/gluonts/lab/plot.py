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

from os import PathLike
from typing import List, Optional, Collection, Tuple, Union
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from gluonts.model import Forecast
from gluonts.model.forecast import SampleForecast
from gluonts.exceptions import GluonTSUserError


def get_percentiles(prediction_intervals):
    percentiles_list = [50.0] + [
        50.0 + sign * percentile / 2.0
        for percentile in prediction_intervals
        for sign in [-1.0, +1.0]
    ]
    return sorted(set(percentiles_list))


def dim_suffix(dim: int):
    return f" (dim {dim})"


def _plot_forecast(
    forecast: Forecast,
    prediction_intervals: Collection[float],
    ax: plt.axis,
    show_mean: bool,
    color: str,
    label_prefix: str,
    dim: Optional[int] = None,
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

        label = f"{label_prefix}mean prediction"
        if dim is not None:
            label += dim_suffix(dim)

        mean_data = np.mean(forecast.samples, axis=0)
        ax.plot(
            forecast.index.to_timestamp(),
            mean_data,
            color=color,
            ls=":",
            label=label,
            *args,
            **kwargs,
        )

    # median prediction
    p50_data = predictions[interval_count]
    p50_series = pd.Series(data=p50_data, index=forecast.index.to_timestamp())
    label = f"{label_prefix}median prediction"
    if dim is not None:
        label += dim_suffix(dim)
    ax.plot(p50_series, color=color, linestyle="--", label=label)

    # percentile prediction intervals
    alphas_lower_half = [(p / 100.0) ** 0.3 for p in percentiles]
    alphas = alphas_lower_half + alphas_lower_half[::-1]
    for i in range(interval_count):
        # plot lower and upper half of median individually to keep colors true
        label = (
            f"{label_prefix}{100 - percentiles[i] * 2}% prediction interval"
        )
        if dim is not None:
            label += dim_suffix(dim)

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
    forecast: Forecast,
    timeseries: Optional[pd.Series] = None,
    prediction_intervals: Collection[float] = (50.0, 90.0),
    show_mean: bool = False,
    color: Union[str, List[str]] = "g",
    xlabel: str = "time",
    ylabel: str = "value",
    show_plot: bool = True,
    show_grid: bool = True,
    figsize: Tuple[int] = (10, 10),
    legend_location: str = "upper left",
    label_prefix: Optional[str] = None,
    output_file: Optional[Union[str, bytes, PathLike]] = None,
    train_test_separator=None,
    variates_to_plot: Optional[List[int]] = None,
    *args,
    **kwargs,
):
    for c in prediction_intervals:
        if not 0.0 <= c <= 100.0:
            raise GluonTSUserError(
                f"Prediction interval {c} is not between 0 and 100"
            )

    dim_count = forecast.dim()
    if variates_to_plot is None:
        variates_to_plot = list(range(dim_count))
    variates_to_plot = sorted(set(variates_to_plot))

    if isinstance(color, str):
        color = [color]
    if len(color) != dim_count:
        raise GluonTSUserError(
            f"Forecast has dimensionality of {dim_count} but "
            f"{len(color)} color(s) were provided"
        )

    if not all(0 <= dim < dim_count for dim in variates_to_plot):
        raise GluonTSUserError(
            "Each dim in variates_to_plot must be in range "
            "0 <= dim < forecast.dim()"
        )

    label_prefix = "" if label_prefix is None else label_prefix + " - "

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    use_dim_in_legend = len(variates_to_plot) > 1

    if timeseries is not None:
        for dim in variates_to_plot:
            label = f"{label_prefix}target"
            if use_dim_in_legend:
                label += dim_suffix(dim)

            ax.plot(
                timeseries.to_timestamp().index,
                timeseries.values[:, dim],
                label=label,
                color=color[dim],
            )

    if forecast is not None:
        for dim in variates_to_plot:
            _plot_forecast(
                forecast=forecast.copy_dim(dim),
                prediction_intervals=prediction_intervals,
                ax=ax,
                show_mean=show_mean,
                color=color[dim],
                label_prefix=label_prefix,
                dim=dim if use_dim_in_legend else None,
                *args,
                **kwargs,
            )

    if train_test_separator is not None:
        ax.axvline(train_test_separator, color="r")

    ax.legend(loc=legend_location)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_grid:
        ax.grid(which="both")

    if output_file:
        fig.savefig(output_file)

    if show_plot:
        fig.show()

    return fig, ax
