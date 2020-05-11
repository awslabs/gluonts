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
import pandas as pd
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt


# Plot posterior densities, from gluonts
def plot_prob_density(
    x_grid,
    densities_eval,
    prediction_intervals=(50.0, 90.0),
    show_mean=False,
    color="b",
):
    for c in prediction_intervals:
        assert 0.0 <= c <= 100.0

    ps = [50.0] + [
        50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
    ]
    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    ps_data = np.percentile(
        densities_eval, q=percentiles_sorted, axis=0, interpolation="lower",
    )
    i_p50 = len(percentiles_sorted) // 2

    p50_data = ps_data[i_p50]
    p50_series = pd.Series(data=p50_data, index=x_grid)
    p50_series.plot(color=color, ls="-", label="median")

    if show_mean:
        mean_data = np.mean(densities_eval, axis=0)
        pd.Series(data=mean_data, index=x_grid).plot(
            color=color, ls=":", label="mean"
        )

    for i in range(len(percentiles_sorted) // 2):
        ptile = percentiles_sorted[i]
        alpha = alpha_for_percentile(ptile)
        plt.fill_between(
            x_grid,
            ps_data[i],
            ps_data[-i - 1],
            facecolor=color,
            alpha=alpha,
            interpolate=True,
        )
        # Hack to create labels for the error intervals.
        # Doesn't actually plot anything, because we only pass a single data point
        pd.Series(data=p50_data[:1], index=x_grid[:1]).plot(
            color=color,
            alpha=alpha,
            linewidth=10,
            label="%d" % (100 - ptile * 2),
        )
        plt.legend()


def plot_1D_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [
        f"{k}% prediction interval" for k in prediction_intervals
    ][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color="g")
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


def plot_probabilistic_forecast(
    dates, observations, prediction_matrix, quantiles, title
):
    mean_pred = np.mean(prediction_matrix, axis=0)
    qnts = mquantiles(
        prediction_matrix, quantiles, axis=0, alphap=0.5, betap=0.5
    )
    prediction_length = len(mean_pred)
    plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.plot(dates, observations, color="r")
    plt.plot(dates[-prediction_length:], mean_pred, color="b")
    plt.fill_between(
        dates[-prediction_length:], qnts[0], qnts[3], alpha=0.3, color="b"
    )
    plt.fill_between(
        dates[-prediction_length:], qnts[1], qnts[2], alpha=0.5, color="b"
    )
    plt.show()
