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
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils import parse_dec_spec_list


def plot_kl_cum_sum(kl_cum_sum, dataset):
    layer_indices = list(range(kl_cum_sum.shape[1]))

    kl_cum_sum_mean = torch.mean(kl_cum_sum, dim=0)
    kl_cum_sum_std = torch.std(kl_cum_sum, dim=0)
    kl_cum_sum_lower = kl_cum_sum_mean - 2 * kl_cum_sum_std
    kl_cum_sum_upper = kl_cum_sum_mean + 2 * kl_cum_sum_std

    # plotting
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # implicitly converts to numpy
    ax.plot(layer_indices, kl_cum_sum_mean, color="black", linewidth=2)  # line plot
    ax.plot(layer_indices, kl_cum_sum_lower, color="black", linewidth=0.5)  # line plot
    ax.plot(layer_indices, kl_cum_sum_upper, color="black", linewidth=0.5)  # line plot

    ax.set_xlabel("Layer index")
    ylabel = "Cumulative, batch-averaged KLs"
    if dataset == "mnist":
        ylabel += " [nats per dim]"
    else:
        ylabel += " [bits per dim]"
    ax.set_ylabel(ylabel)

    # horizontal line at 0.0
    ax.hlines(
        y=0.0,
        xmin=layer_indices[0],
        xmax=layer_indices[-1],
        color="black",
        linestyle="--",
    )

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])

    return fig


def plot_state_norm(state_norm, enc_or_dec, res_to_n_layers):
    # print("l2 norm of (input) in top-down block diagramme, block idx %d, res %d: "%(self.idx, x.shape[2]), torch.mean(torch.linalg.norm(torch.flatten(x, start_dim=1), dim=1)).item())
    norm_mean = np.mean(state_norm, axis=0).flatten()
    norm_std = np.std(state_norm, axis=0).flatten()

    lower = norm_mean - 2 * norm_std
    upper = norm_mean + 2 * norm_std
    n_blocks = sum([n_layers for n_layers in res_to_n_layers.values()])
    layer_idxs = np.arange(n_blocks)

    # horizontal lines indicating the resolution jumps
    res_to_reps_tuples = [(key, val) for key, val in res_to_n_layers.items()]
    if enc_or_dec == "enc":
        reverse = True
    else:
        reverse = False
    res_to_reps_tuples = sorted(
        res_to_reps_tuples, key=lambda tup: tup[0], reverse=reverse
    )
    reps = [tup[1] for tup in res_to_reps_tuples]
    resolutions = [int(tup[0]) for tup in res_to_reps_tuples]
    jumps = (np.cumsum(reps) - 0.5).tolist()[:-1]

    # plotting
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(layer_idxs, norm_mean, color="black", linewidth=2)
    plt.plot(layer_idxs, lower, color="black", linewidth=0.5)
    plt.plot(layer_idxs, upper, color="black", linewidth=0.5)

    for jump in jumps:
        plt.axvline(x=jump, color="g", linestyle="--")
    if enc_or_dec == "enc":
        x_label = "Forward (encoder) block $i$"
        y_label_insert = "forward"
    else:
        x_label = "Backward (decoder) block $i$"
        y_label_insert = "backward"
    plt.xlabel(x_label)
    plt.ylabel(
        "$\|\| y_i \|\|_2$ ($y_i$ is output of " + y_label_insert + " block $i$)"
    )
    plt.xlim(0, np.max(layer_idxs))
    plt.ylim(0, np.max(upper))

    y_text_pos = 0.9 * np.max(upper)
    for i, jump in enumerate([0] + jumps):
        label = str(resolutions[i])
        shift = 0.2 + 0.5 * (len(jumps) / 100)
        plt.text(
            jump + shift,
            y_text_pos,
            label,
            fontsize=6,
            color="green",
            rotation="vertical",
        )

    # plt.show()

    return fig


def plot_inputs_and_recons(
    x_list,
    x_hat_list,
    recon_n_rows,
    recon_n_cols,
    do_samples=False,
    x_context_list=None,
):
    # find dimensions of one input
    # ts_dims = x_list[0].size()

    assert len(x_list) >= recon_n_rows * recon_n_cols
    assert len(x_hat_list) >= recon_n_rows * recon_n_cols
    assert len(x_context_list) >= recon_n_rows * recon_n_cols

    # find out scneario
    if x_context_list is None:
        conditional = False
    else:
        conditional = True

    # find out dimensions of context and forecast window, create indices
    if conditional:
        context_length = x_context_list[0].flatten().shape[0]
        context_indices = np.arange(context_length)
    forecast_length = x_list[0].flatten().shape[0]
    forecast_indices = (
        np.arange(context_length, forecast_length + context_length)
        if conditional
        else np.arange(forecast_length)
    )

    # make grid of plots
    # expects list of time series, each of shape (C x T)
    # plotting
    # Note: Nice examples on how to share x and/or y axis, see examples in: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    plt.clf()
    fig = plt.figure(figsize=(recon_n_cols * 1.5 * 3, recon_n_rows * 3))
    gs = fig.add_gridspec(
        nrows=recon_n_rows, ncols=recon_n_cols, hspace=0
    )  # , hspace=0, wspace=0
    axes = gs.subplots(sharex="col")

    ind = 0
    for r in range(recon_n_rows):
        for c in range(recon_n_cols):
            # TODO flatten assumes 1d time series --> later extend to multivariate
            if do_samples:
                x_hat_q50 = torch.quantile(x_hat_list[ind], dim=0, q=0.5).flatten()
                x_hat_q05 = torch.quantile(x_hat_list[ind], dim=0, q=0.05).flatten()
                x_hat_q95 = torch.quantile(x_hat_list[ind], dim=0, q=0.95).flatten()
                axes[r, c].plot(
                    forecast_indices,
                    x_hat_q50,
                    color="blue",
                    label="forecast (median)",
                    linewidth=0.5,
                )
                axes[r, c].plot(
                    forecast_indices,
                    x_hat_q05,
                    color="blue",
                    label="forecast (0.05-quantile)",
                    linewidth=0.5,
                    linestyle="--",
                )
                axes[r, c].plot(
                    forecast_indices,
                    x_hat_q95,
                    color="blue",
                    label="forecast (.95-quantile)",
                    linewidth=0.5,
                    linestyle="--",
                )
                axes[r, c].plot(
                    forecast_indices,
                    x_list[ind].flatten(),
                    color="black",
                    label="ground truth",
                    linewidth=0.5,
                )
                if conditional:
                    axes[r, c].plot(
                        context_indices,
                        x_context_list[ind].flatten(),
                        color="green",
                        label="context window",
                        linewidth=0.5,
                    )
            else:
                axes[r, c].plot(
                    forecast_indices,
                    x_list[ind].flatten(),
                    color="black",
                    label="ground truth",
                    linewidth=0.8,
                )
                axes[r, c].plot(
                    forecast_indices,
                    x_hat_list[ind].flatten(),
                    color="blue",
                    label="forecast (mean)",
                    linewidth=0.5,
                )
                if conditional:
                    axes[r, c].plot(
                        context_indices,
                        x_context_list[ind].flatten(),
                        color="green",
                        label="context window",
                        linestyle="--",
                        linewidth=0.5,
                    )

            ind += 1

    # fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])

    # add shared legend (taking the one from the first element)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3 if conditional else 2)

    # for ax in axes.flat:
    #     ax.label_outer()

    # ax.set_xticks([])
    # ax.set_yticks([])

    # fig.text(0.5, 0.99, 'Pairs of inputs and reconstructions', ha='center')  # common X label

    # if do_samples:
    #     print("hello")

    # plt.show()

    return fig


def plot_p_sample(ts_list, p_sample_n_rows, p_sample_n_cols, x_context_list=None):
    # find dimensions of one input
    # ts_dims = ts_list[0].size()

    # find out scneario
    if x_context_list is None:
        conditional = False
    else:
        conditional = True

    # find out dimensions of context and forecast window, create indices
    if conditional:
        context_length = x_context_list[0].flatten().shape[0]
        context_indices = np.arange(context_length)
    ts_length = ts_list[0].flatten().shape[0]
    ts_indices = (
        np.arange(context_length, ts_length + context_length)
        if conditional
        else np.arange(ts_length)
    )

    # make grid of plots
    # expects list of time series, each of shape (C x T)
    # plotting
    # Note: Nice examples on how to share x and/or y axis, see examples in: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    plt.clf()
    fig = plt.figure(figsize=(p_sample_n_cols * 3 * 1.5, p_sample_n_rows * 3))
    gs = fig.add_gridspec(
        nrows=p_sample_n_rows, ncols=p_sample_n_cols, hspace=0
    )  # , hspace=0, wspace=0
    axes = gs.subplots(sharex="col")

    ind = 0
    for r in range(p_sample_n_rows):
        for c in range(p_sample_n_cols):
            # TODO flatten assumes 1d time series --> later extend to multivariate
            axes[r, c].plot(
                ts_indices,
                ts_list[ind].flatten(),
                color="blue",
                label="sample",
                linewidth=0.5,
            )
            if conditional:
                axes[r, c].plot(
                    context_indices,
                    x_context_list[ind].flatten(),
                    color="green",
                    label="context window (conditioned)",
                    linestyle="--",
                    linewidth=0.5,
                )

            ind += 1

    # fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])

    # add shared legend (taking the one from the first element)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3 if conditional else 2)

    return fig
