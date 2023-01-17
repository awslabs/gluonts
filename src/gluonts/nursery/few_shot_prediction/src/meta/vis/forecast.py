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

from typing import List
import numpy as np
import matplotlib.pyplot as plt


def plot_forecast_supportset_attention(
    query_past: List[np.ndarray],
    query_future: List[np.ndarray],
    pred: List[np.ndarray],
    supps: List[np.ndarray],
    attention: List[np.ndarray],
    quantiles: List[str],
) -> plt.Figure:
    """
    Plots the provided forecasts for each sample with confidence intervals using all provided quantiles.
    Furthermore, plots the time series in the support set of this sample aligned with their attention scores.

    Parameters
    ----------
    query_past: List[np.ndarray] of length n_samples, each array has shape [context length]
        The past query which the model uses to make a prediction.
    query_future: List[np.ndarray] of length n_samples, each array has shape [prediction horizon length]
        The ground truth for the forecast.
    pred: List[np.ndarray], of length n_samples, each array has shape  [prediction horizon length, n_quantiles]
        The prediction of the model.
    supps: List[np.ndarray], length is n_samples, each item has shape[supps_size, support ts length]
        The support sets for each query.
    attention: List[np.ndarray], length is n_samples, each item has shape[supps_size, support ts length]
        The attention scores for each support set times series.
    quantiles: List[float], shape[n_quantiles,]

    Returns
    -------
    plt.figure.Figure
        A plot containing one subplot for each sample. Each subplot displays the ground truth, predicted confidence
        intervals, the predicted median. Furthermore, the time series in the support set aligned with their attention
        scores.
    """
    supps_size = len(supps[0])
    n_samples = len(query_past)
    plots_per_sample = supps_size + 1
    n_plots = plots_per_sample * n_samples
    fig, subs = plt.subplots(n_plots, figsize=(10, 4 * n_plots))
    for i in range(n_samples):
        # plot prediction
        subs[plots_per_sample * i].set_title(f"query {i}")
        _plot_quantile_forecast(
            subs[plots_per_sample * i],
            query_past[i],
            query_future[i],
            pred[i],
            quantiles,
        )
        # plot support ts and attention weights
        for j in range(1, supps_size + 1):
            subs[plots_per_sample * i + j].set_title(
                f"support ts {j} for query {i}"
            )
            subs[plots_per_sample * i + j].plot(supps[i][j - 1].squeeze())

            subs[plots_per_sample * i + j].sharex(subs[plots_per_sample * i])
            subs[plots_per_sample * i + j].sharey(subs[plots_per_sample * i])
            # twin object for two different y-axis on the sample plot
            ax2 = subs[plots_per_sample * i + j].twinx()
            for head in range(attention[0][0].shape[1]):
                n_att = len(attention[i][j - 1][:, head])
                n_supps = len(supps[i][j - 1])
                ax2.fill_between(
                    np.linspace(start=0, stop=n_supps, num=n_att),
                    attention[i][j - 1][:, head],
                    alpha=0.25,
                    # label=f"head {head}",
                    label="accumulated attention",
                )
            if j == 1:
                ax_base = ax2
            else:
                ax2.sharey(ax_base)
            ax2.set_ylabel("attention scores", color="red")
            ax2.legend()
    return fig


def plot_quantile_forecast(
    query_past: List[np.ndarray],
    query_future: List[np.ndarray],
    pred: List[np.ndarray],
    quantiles: List[str],
) -> plt.Figure:
    """
    Plots the provided forecasts for each sample with confidence intervals using all provided quantiles.

    Parameters
    ----------
    query_past: List[np.ndarray] of length n_samples, each array has shape [context length]
        The past query which the model uses to make a prediction.
    query_future: List[np.ndarray] of length n_samples, each array has shape [prediction horizon length]
        The ground truth for the forecast.
    pred: List[np.ndarray], of length n_samples, each array has shape  [prediction horizon length, n_quantiles]
        The prediction of the model.
    quantiles: List[float], shape[n_quantiles,]

    Returns
    -------
    plt.figure.Figure
        A plot containing one subplot for each sample. Each subplot displays the ground truth, predicted confidence
        intervals and the predicted median.
    """
    n_samples = len(query_past)
    fig, subs = plt.subplots(n_samples, figsize=(10, 4 * n_samples))
    if n_samples > 1:
        for i in range(n_samples):
            subs[i].set_title(f"sample {i}")
            _plot_quantile_forecast(
                subs[i], query_past[i], query_future[i], pred[i], quantiles
            )
    else:
        i = 0
        subs.set_title(f"sample {i}")
        _plot_quantile_forecast(
            subs, query_past[i], query_future[i], pred[i], quantiles
        )
    return fig


def _plot_quantile_forecast(
    sub: plt.Axes,
    query_past: np.ndarray,
    query_future: np.ndarray,
    pred: np.ndarray,
    quantiles: List[str],
):
    """
    Plots the provided forecast with confidence intervals using all provided quantiles.

    Parameters
    ----------
    sub: plt.Axis
        The axis object to plot on.
    query_past: np.ndarray, shape[context length]
        The past query which the model uses to make a prediction.
    query_future: np.ndarray, shape[prediction horizon length]
        The ground truth for the forecast.
    pred: np.ndarray, shape[prediction horizon length, n_quantiles]
        The prediction of the model.
    quantiles: List[str], shape[n_quantiles,]
    """
    assert len(pred) == len(
        query_future
    ), f"len pred is {len(pred)}, len query_future is {len(query_future)}"
    query = np.concatenate([query_past, query_future])
    sub.plot(query, label="gt")
    sub.axvline(len(query_past) - 1, color="r")  # end of train dataset
    pred_time = np.arange(len(query_past), len(query_past) + len(pred))
    cmap = plt.get_cmap("Oranges")
    # Plot the shapes for the confidence intervals, starting with the outermost
    num_intervals = pred.shape[-1] // 2
    for i in range(num_intervals):
        lower = pred[..., i]
        upper = pred[..., -(i + 1)]
        ci = 100 - 2 * int(float(quantiles[i]) * 100)
        cmap_index = int((cmap.N / (num_intervals + 1)) * (i + 1))
        if len(pred) > 1:
            sub.fill_between(
                pred_time,
                lower,
                upper,
                color=cmap(cmap_index),
                label=f"{ci}% CI",
            )
        else:
            sub.fill_between(
                np.append(pred_time, pred_time[0] + 0.5),
                np.repeat(lower, 2),
                np.repeat(upper, 2),
                color=cmap(cmap_index),
                label=f"{ci}% CI",
            )
    if len(pred) > 1:
        sub.plot(
            pred_time,
            pred[..., num_intervals],
            color="black",
            label="pred median",
        )
    else:
        sub.plot(
            np.append(pred_time, pred_time[0] + 0.5),
            np.repeat(pred[..., num_intervals], 2),
            color="black",
            label="pred median",
        )
    sub.legend()


def plot_point_forecast(sub, query_past, query_future, pred):
    query = np.concatenate([query_past, query_future])
    sub.plot(query, label="gt")
    sub.axvline(len(query_past) - 1, color="r")  # end of train dataset
    pred_time = np.arange(len(query_past), len(query_past) + len(pred))
    sub.plot(pred_time, pred, label="pred")
