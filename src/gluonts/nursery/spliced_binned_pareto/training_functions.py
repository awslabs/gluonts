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

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch.nn
import torch.optim

from .distr_tcn import DistributionalTCN


def train_step_from_batch(
    ts_chunks: torch.Tensor,
    targets: torch.Tensor,
    distr_tcn: DistributionalTCN,
    optimizer: torch.optim.Adam,
):
    """
    Arguments
    ----------
    ts_chunks: Mini-batch chunked from the time series
    targets: Corresponding chunk of target values
    distr_tcn: DistributionalTCN
    otimizer: Optimizer containing parameters, learning rate, etc
    """
    distr_outputs = distr_tcn(ts_chunks.float())
    loss = -distr_outputs.log_prob(targets.float())
    loss = loss.mean()
    loss_value = loss.cpu().detach().numpy()
    loss.backward()
    optimizer.step()
    return loss_value


def eval_on_series(
    distr_tcn: DistributionalTCN,
    optimizer: torch.optim.Adam,
    series_tensor: torch.Tensor,
    ts_len: int,
    context_length: int,
    is_train: bool = False,
    return_predictions: bool = False,
    lead_time: int = 1,
):
    """
    Arguments
    ----------
    distr_tcn: DistributionalTCN
    otimizer: Optimizer containing parameters, learning rate, etc
    series_tensor: Time series
    ts_len: Length of time series
    context_length: Number of time steps to input
    is_train: True if time series is training set
    return_predictions: True if to return (loss, predictions), False if to return loss only
    lead_time: Number of time steps to predict ahead
    """
    loss_log = []

    # Parallelising the training:
    trying_mini_batches = True
    if is_train:
        mini_batch_size = 64
        stride = 1

        window_length = context_length + lead_time

        unfold_layer = torch.nn.Unfold(
            kernel_size=(1, window_length), stride=stride
        )

        ts_windows = (
            unfold_layer(series_tensor.unsqueeze(2))
            .transpose(1, 2)
            .transpose(0, 1)
        )

        numb_mini_batches = ts_windows.shape[0] // mini_batch_size

        if trying_mini_batches:
            batch_indices = np.arange(ts_len - window_length - 1)
            for i in range(numb_mini_batches):
                idx = np.random.choice(batch_indices, mini_batch_size)
                batch_indices = np.setdiff1d(batch_indices, idx)

                ts_chunks = ts_windows[idx, :, :-lead_time]
                targets = ts_windows[idx, :, -1]

                loss_log.append(
                    train_step_from_batch(
                        ts_chunks, targets, distr_tcn, optimizer
                    )
                )

        else:
            ts_chunks = ts_windows[:, :, :-lead_time]
            targets = ts_windows[:, :, -1]

            loss_log.append(train_step_from_batch(ts_chunks, targets))

        return loss_log

    if return_predictions:
        predictions = {
            "low_lower": [],
            "lower": [],
            "median": [],
            "upper": [],
            "up_upper": [],
        }

    for i in range(ts_len - context_length - lead_time - 1):
        ts_chunk = series_tensor[:, :, i : i + context_length]
        target = series_tensor[:, :, i + context_length + lead_time - 1]

        distr_output = distr_tcn(ts_chunk.float())

        if return_predictions:
            predictions["lower"].append(distr_output.icdf(torch.tensor(0.05)))
            predictions["median"].append(distr_output.icdf(torch.tensor(0.5)))
            predictions["upper"].append(distr_output.icdf(torch.tensor(0.95)))
            predictions["low_lower"].append(
                distr_output.icdf(torch.tensor(0.01))
            )
            predictions["up_upper"].append(
                distr_output.icdf(torch.tensor(0.99))
            )

        loss = -distr_output.log_prob(target.float())
        loss_value = loss.cpu().detach().numpy()[0]
        loss_log.append(loss_value)

        if is_train:
            loss.backward()
            optimizer.step()

    if return_predictions:
        return loss_log, predictions
    return loss_log


def plot_prediction(
    val_ts_tensor: torch.Tensor,
    predictions: torch.Tensor,
    context_length: int,
    lead_time: int = 1,
    start: int = 0,
    end: int = 500,
    fig: Optional[matplotlib.figure.Figure] = None,
):
    """
    Arguments
    ----------
    val_ts_tensor: Time series
    predictions: Prediction series
    context_length: Number of time steps to input
    lead_time: Number of time steps to predict ahead
    start: Index of time series at which to start plotting
    end: Index of time series at which to end plotting
    """
    end = np.minimum(end, len(predictions["lower"]))

    for key in predictions.keys():
        try:
            predictions[key] = np.array(
                list(map(lambda x: x.detach().cpu().numpy(), predictions[key]))
            ).ravel()
        except Exception:
            # TODO: handle error
            pass

    if fig is None:
        fig = plt.figure(figsize=(16, 7))
    plt.plot(
        val_ts_tensor.cpu().flatten()[context_length + lead_time - 1 :][
            start:end
        ],
        "ko",
        markersize=2,
        label=r"$x_t$",
    )
    plt.plot(
        predictions["up_upper"][start:end],
        "r-",
        alpha=0.3,
        label=r"$z_{0.01}(t)|x_{t-1}$",
    )
    plt.plot(
        predictions["upper"][start:end],
        "y-",
        alpha=0.3,
        label=r"$z_{0.05}(t)|x_{t-1} \rightarrow \tau_t^{\rm{upper}}$",
    )
    plt.fill_between(
        np.arange(start, end),
        predictions["lower"][start:end],
        predictions["upper"][start:end],
        facecolor="y",
        alpha=0.3,
        label=None,
    )
    plt.plot(
        predictions["median"][start:end], "g", label=r"$z_{0.5\,}(t)|x_{t-1}$"
    )
    plt.plot(
        predictions["lower"][start:end],
        "y-",
        alpha=0.3,
        label=r"$z_{0.95}(t)|x_{t-1} \rightarrow \tau_t^{\rm{lower}}$",
    )
    plt.plot(
        predictions["low_lower"][start:end],
        "r-",
        alpha=0.3,
        label=r"$z_{0.99}(t)|x_{t-1}$",
    )

    plt.legend(loc="upper right")

    median = np.array(predictions["median"][start:end])
    true_val = np.array(
        val_ts_tensor.cpu().flatten()[context_length + lead_time - 1 :][
            start:end
        ]
    )
    MAE = np.mean(np.abs(median - true_val))
    plt.title(str(MAE))

    return fig


def highlight_min(data, color="lightgreen"):
    """
    Highlights the minimum in a Series or DataFrame.
    """
    attr = f"background-color: {color}"
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else "" for v in is_min]
    else:  # from .apply(axis=None)
        return pd.DataFrame(
            np.where(is_min, attr, ""), index=data.index, columns=data.columns
        )
