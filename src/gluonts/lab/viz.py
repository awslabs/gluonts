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

import pandas as pd
from matplotlib import pyplot as plt
from .helpers import plot_forecast, read_input_for_marker_or_color

from gluonts.model import Forecast
from gluonts.exceptions import GluonTSUserError


def plot_time_series(
    time_series: Union[pd.Series, pd.DataFrame],
    train_test_separator: Optional[pd.Timestamp] = None,
    label: str = "target",
    color="black",
    plt_context: Optional[Tuple[plt.figure, plt.axis]] = None,
    save_path: Optional[Union[str, bytes, PathLike]] = None,
    show_plot: bool = True,
):
    """Plots a univariate or multivariate time series

    Parameters
    ----------
    time_series
        data points to plot, in Pandas format
    train_test_separator
        if provided a time stamp, draws a vertical line at this position
    label
        label to use in call to plot
    color
        line color, defaults to "black"
    plt_context
        tuple (fig, ax) to use; if None, new plot will be created
    save_path
        specifies where to save the plot
    show_plot
        wether or not to show the plot using matplolibs `show`, defaults to True
    """
    if plt_context is None:
        plt_context = plt.subplots()
    fig, axis = plt_context

    if isinstance(time_series, pd.Series):
        time_series = time_series.to_frame()

    axis.plot(
        time_series.to_timestamp().index,
        time_series.values,
        label=label,
        color=color,
    )

    if train_test_separator is not None:
        axis.axvline(train_test_separator, color="r")

    if save_path:
        fig.savefig(save_path)
    if show_plot:
        fig.show()

    return fig, axis


def plot_univariate_forecast(
    forecast: Optional[Forecast],
    time_series: Optional[Union[pd.DataFrame, pd.Series]] = None,
    prediction_intervals: Collection[float] = (50.0, 90.0),
    plot_mean: bool = False,
    figsize: Tuple[int] = (10, 10),
    xlabel: str = "Time",
    ylabel: str = "Value",
    label_prefix: str = "",
    label_suffix: str = "",
    color: str = "g",
    marker: Optional[str] = None,
    legend_location: str = "upper left",
    plot_grid: bool = True,
    show_plot: bool = True,
    save_path: Optional[Union[str, bytes, PathLike]] = None,
) -> Tuple[plt.figure, plt.axis]:
    """Plots prediction intervals of a single probabilistic forecast

    Parameters
    ----------
    forecast
        Forecast to plot
    time_series
        ground truth for comparison, can start at earlier time than forecast
    prediction_intervals
        a collection of numbers between 0 and 100 specifying what prediction
        intervals to plot - the larger a value, the fainter the color
    plot_mean
        wether or not to plot the forecast mean, defaults to False
    ...
    """

    for c in prediction_intervals:
        if not 0.0 <= c <= 100.0:
            raise GluonTSUserError(
                f"Prediction interval {c} is not between 0 and 100"
            )

    dim_count = forecast.dim()
    assert (
        dim_count == 1
    ), f"Expected univariate forecast but got {dim_count} dimensions(s)"

    fig, axis = plt.subplots(1, 1, figsize=figsize)

    if time_series is not None:
        plot_time_series(
            time_series=time_series,
            label=f"{label_prefix}target{label_suffix}",
            plt_context=(fig, axis),
        )

    plot_forecast(
        forecast=forecast,
        prediction_intervals=prediction_intervals,
        axis=axis,
        plot_mean=plot_mean,
        color=color,
        marker=marker,
        label_prefix=label_prefix,
    )

    axis.legend(loc=legend_location)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if plot_grid:
        axis.grid(which="both")

    if save_path:
        fig.savefig(save_path)
    if show_plot:
        fig.show()

    return fig, axis


def plot_multivariate_forecast(
    forecast: Forecast,
    time_series: Optional[Union[pd.DataFrame, pd.Series]] = None,
    prediction_intervals: Collection[float] = (50.0, 90.0),
    variates_to_plot: Optional[List[int]] = None,
    plot_mean: bool = False,
    figsize: Tuple[int] = (10, 10),
    xlabel: str = "Time",
    ylabel: str = "Value",
    label_prefix: Optional[str] = None,
    legend_location: str = "upper left",
    plot_grid: bool = True,
    color: Union[str, List[str]] = "g",
    marker: Union[str, List[str]] = "o",
    use_subplots: bool = True,
    show_plot: bool = True,
    save_path: Optional[Union[str, bytes, PathLike]] = None,
):
    def dim_suffix(dim: int) -> str:
        return f" (dim {dim})"

    for c in prediction_intervals:
        if not 0.0 <= c <= 100.0:
            raise GluonTSUserError(
                f"Prediction interval {c} is not between 0 and 100"
            )

    dim_count = forecast.dim()
    assert (
        dim_count > 1
    ), f"Expected multivariate forecast but got {dim_count} dimension(s)"

    dim_count = forecast.dim()
    if variates_to_plot is None:
        variates_to_plot = list(range(dim_count))
    variates_to_plot = sorted(set(variates_to_plot))

    if not all(0 <= dim < dim_count for dim in variates_to_plot):
        raise GluonTSUserError(
            "Each dim in variates_to_plot must be in range "
            "0 <= dim < forecast.dim()"
        )

    color = read_input_for_marker_or_color(color, dim_count, "color")
    plot_markers = marker is not None
    if plot_markers:
        marker = read_input_for_marker_or_color(marker, dim_count, "marker")

    label_prefix = "" if label_prefix is None else label_prefix

    subplot_count = len(variates_to_plot)
    if use_subplots:
        fig, axes = plt.subplots(
            subplot_count, 1, figsize=figsize, sharex=True, sharey=True
        )
    else:
        fig, axis = plt.subplots(1, 1, figsize=figsize)
        axes = [axis] * subplot_count  # always use the same axis to draw on

    if time_series is not None:
        for axis, dim in zip(axes, variates_to_plot):
            plot_time_series(
                time_series=time_series[dim],
                label=f"{label_prefix}target{dim_suffix(dim)}",
                plt_context=(fig, axis),
            )

    for axis, dim in zip(axes, variates_to_plot):
        label_suffix = dim_suffix(dim)
        plot_forecast(
            forecast=forecast.copy_dim(dim),
            prediction_intervals=prediction_intervals,
            axis=axis,
            plot_mean=plot_mean,
            color=color[dim],
            label_prefix=label_prefix,
            label_suffix=label_suffix,
            marker=marker[dim] if plot_markers else None,
        )

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, labels, loc=legend_location)
        if plot_grid:
            axis.grid(which="both")

    if save_path:
        fig.savefig(save_path)
    if show_plot:
        fig.show()

    return fig, axes


def plot_forecast_comparison(
    forecasts: Collection[Forecast],
    time_series: Optional[Union[pd.DataFrame, pd.Series]] = None,
    prediction_intervals: Collection[float] = (50.0, 90.0),
    plot_mean: bool = False,
    figsize: Tuple[int] = (10, 10),
    xlabel: str = "Time",
    ylabel: str = "Value",
    label_prefix: Optional[str] = None,
    legend_location: str = "upper left",
    plot_grid: bool = True,
    color: Union[str, List[str]] = "g",
    marker: Optional[List[str]] = None,
    use_subplots: bool = True,
    show_plot: bool = True,
    save_path: Optional[Union[str, bytes, PathLike]] = None,
):
    for c in prediction_intervals:
        if not 0.0 <= c <= 100.0:
            raise GluonTSUserError(
                f"Prediction interval {c} is not between 0 and 100"
            )

    forecast_count = len(forecasts)

    color = read_input_for_marker_or_color(color, forecast_count, "color")

    plot_markers = marker is not None
    if plot_markers:
        marker = read_input_for_marker_or_color(
            marker, forecast_count, "marker"
        )

    label_prefix = "" if label_prefix is None else label_prefix

    if use_subplots:
        fig, axes = plt.subplots(
            forecast_count, 1, figsize=figsize, sharex=True, sharey=True
        )
    else:
        fig, axis = plt.subplots(1, 1, figsize=figsize)
        axes = [axis] * forecast_count  # always use the same axis to draw on

    if time_series is not None:
        for axis, forecast_id in zip(axes, range(forecast_count)):
            plot_time_series(
                time_series=time_series,
                label=f"{label_prefix}target",
                plt_context=(fig, axis),
            )

    if forecasts is not None:
        for axis, forecast_id in zip(axes, range(forecast_count)):
            plot_forecast(
                forecast=forecasts[forecast_id],
                prediction_intervals=prediction_intervals,
                axis=axis,
                plot_mean=plot_mean,
                color=color[forecast_id],
                label_prefix=label_prefix,
                label_suffix=str(forecast_id),
                marker=marker[forecast_id] if plot_markers else None,
            )

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, labels, loc=legend_location)
        if plot_grid:
            axis.grid(which="both")

    if save_path:
        fig.savefig(save_path)
    if show_plot:
        fig.show()

    return fig, axes
