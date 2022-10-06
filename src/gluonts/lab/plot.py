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


def read_input(value: Union[str, List[str]], dim_count: int, entity_name: str):
    """normalize input for 'maker' or 'color' into list of length dim_count"""
    result = value
    if isinstance(result, str):
        result = [result]
    if len(result) == 0:
        raise GluonTSUserError(f"'{entity_name}' can't be empty list")

    # repeat if necesarry to match dim_count
    result_idx = 0
    while len(result) < dim_count:
        result.append(result[result_idx])
        result_idx += 1

    return result


def plot_forecast(
    ax: plt.axis,
    forecast: Forecast,
    prediction_intervals: Collection[float],
    plot_mean: bool,
    label_prefix: str,
    color: str,
    dim: Optional[int] = None,
    marker: Optional[str] = None,
    *args,
    **kwargs,
):
    interval_count = len(prediction_intervals)

    percentiles = get_percentiles(prediction_intervals)
    predictions = [forecast.quantile(p / 100.0) for p in percentiles]

    if plot_mean:
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
            marker=marker,
            *args,
            **kwargs,
        )

    # median prediction
    p50_data = predictions[interval_count]
    p50_series = pd.Series(data=p50_data, index=forecast.index.to_timestamp())
    label = f"{label_prefix}median prediction"
    if dim is not None:
        label += dim_suffix(dim)
    ax.plot(
        p50_series, label=label, linestyle="--", color=color, marker=marker
    )

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
    timeseries: Optional[Union[pd.DataFrame, pd.Series]] = None,
    prediction_intervals: Collection[float] = (50.0, 90.0),
    variates_to_plot: Optional[List[int]] = None,
    plot_mean: bool = False,
    train_test_separator: Optional[pd.Timestamp] = None,
    figsize: Tuple[int] = (10, 10),
    xlabel: str = "time",
    ylabel: str = "value",
    label_prefix: Optional[str] = None,
    legend_location: str = "upper left",
    plot_grid: bool = True,
    color: Union[str, List[str]] = "g",
    marker: Union[str, List[str]] = "o",
    plot_markers: bool = False,
    show_plot: bool = True,
    save_path: Optional[Union[str, bytes, PathLike]] = None,
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

    if not all(0 <= dim < dim_count for dim in variates_to_plot):
        raise GluonTSUserError(
            "Each dim in variates_to_plot must be in range "
            "0 <= dim < forecast.dim()"
        )

    color = read_input(color, dim_count, "color")
    marker = read_input(marker, dim_count, "marker")

    label_prefix = "" if label_prefix is None else label_prefix + " - "
    plot_multiple_variates = len(variates_to_plot) > 1

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if timeseries is not None:
        if isinstance(timeseries, pd.Series):
            timeseries = timeseries.to_frame()

        for dim in variates_to_plot:
            label = f"{label_prefix}target"
            if plot_multiple_variates:
                label += dim_suffix(dim)

            ax.plot(
                timeseries.to_timestamp().index,
                timeseries.values[:, dim],
                label=label,
                color=color[dim] if plot_multiple_variates else "black",
                marker=marker[dim] if plot_markers else None,
            )

    if forecast is not None:
        for dim in variates_to_plot:
            plot_forecast(
                forecast=forecast.copy_dim(dim),
                prediction_intervals=prediction_intervals,
                ax=ax,
                plot_mean=plot_mean,
                color=color[dim],
                label_prefix=label_prefix,
                dim=dim if plot_multiple_variates else None,
                marker=marker[dim] if plot_markers else None,
                *args,
                **kwargs,
            )

    if train_test_separator is not None:
        ax.axvline(train_test_separator, color="r")

    ax.legend(loc=legend_location)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if plot_grid:
        ax.grid(which="both")

    if save_path:
        fig.savefig(save_path)

    if show_plot:
        fig.show()

    return fig, ax
