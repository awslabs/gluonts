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

import json
import pickle
import random

import torch
import numpy as np

import pandas as pd
from tqdm.auto import tqdm

from typing import Iterator
from gluonts.dataset.common import DataEntry, Dataset, ListDataset


PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target_cdf",
    "past_observed_values",
    "future_time_feat",
    "past_is_pad",
]


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path: str = None):
        if json_path is not None:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by params.dict['learning_rate']"""
        return self.__dict__


class AttackResults:
    def __init__(self, batch, perturbation, true_future_target, attack_idx):
        self.batch = batch
        self.perturbation = perturbation
        self.true_future_target = true_future_target
        self.attack_idx = attack_idx


class Metrics:
    def __init__(
        self,
        mse,
        mape,
        ql,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ):
        self.mse = mse
        self.mape = mape
        self.ql = ql
        self.quantiles = quantiles


def smoothed_inference(
    batch,
    past_target,
    net,
    sigma,
    device,
    num_noised_samples: int = 100,
    intermediate_noise: float = None,
    retain_positivity: bool = True,
):
    outputs = []

    for _ in tqdm(range(num_noised_samples), leave=False):
        noised_past_target = change_device(past_target, device)
        noised_past_target += torch.normal(
            mean=torch.zeros(noised_past_target.shape, device=device),
            std=sigma * torch.abs(noised_past_target),
        )
        if retain_positivity:
            noised_past_target = torch.clamp(noised_past_target, min=0)

        noised_inputs = {key: batch[key] for key in PREDICTION_INPUT_NAMES}
        noised_inputs["past_target_cdf"] = noised_past_target

        sample = net(**noised_inputs)

        outputs.append(sample.detach().cpu().numpy())

        del noised_past_target, noised_inputs, sample
        torch.cuda.empty_cache()

    return np.concatenate(outputs, axis=1)


def requires_grad_(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def get_augmented_dataset(dataset, num_noises: int = 100, sigma: float = 0.1):
    train_data_list = list(iter(dataset.train_ds))
    train_length = len(train_data_list)

    for _ in range(num_noises):
        for idx in range(train_length):
            target = train_data_list[idx]["target"]
            data = {
                "start": train_data_list[idx]["start"],
                "target": target
                + np.random.normal(
                    loc=np.zeros_like(target), scale=sigma * target
                ),
                "feat_static_cat": train_data_list[idx][
                    "feat_static_cat"
                ].copy(),
                "item_id": None,
                "source": None,
            }
            train_data_list.append(data)

    random.shuffle(train_data_list)
    return ListDataset(
        data_iter=train_data_list, freq=dataset.freq, one_dim_target=False
    )


def add_ts_dataframe(
    data_iterator: Iterator[DataEntry], freq
) -> Iterator[DataEntry]:
    for data_entry in data_iterator:
        data = data_entry.copy()

        index = pd.date_range(
            start=data["start"],
            freq=freq,
            periods=data["target"].shape[-1],
        )
        data["ts"] = pd.DataFrame(index=index, data=data["target"].transpose())
        yield data


def ts_iter(dataset: Dataset, freq) -> pd.DataFrame:
    for data_entry in add_ts_dataframe(iter(dataset), freq):
        yield data_entry["ts"]


def load_pickle(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def change_device(var, device):
    if isinstance(var, torch.Tensor):
        if var.device == "cpu":
            var.to(device)
            return var
        elif var.device != device:
            return var.cpu().to(device)
        else:
            return var
    if isinstance(var, dict):
        for key in var.keys():
            if isinstance(var[key], torch.Tensor):
                var[key] = var[key].to(device)
        return var

    return torch.from_numpy(var).float().to(device)


def quantile_loss(true, pred, quantile):
    denom = np.abs(true).sum(1).sum(1)  # batch x 1
    if true.ndim < 3:
        true = true.reshape(true.shape + tuple([1] * (3 - true.ndim)))
    if pred.ndim < 3:
        pred = pred.reshape(pred.shape + tuple([1] * (3 - pred.ndim)))
    batch, time, dim = true.shape
    ql = np.zeros(batch)
    idx = 0
    for y_hat, y in zip(pred, true):
        num = 0
        for t in range(time):
            for j in range(dim):
                num += (
                    (1 - quantile) * abs(y_hat[t, j] - y[t, j])
                    if y_hat[t, j] > y[t, j]
                    else quantile * abs(y_hat[t, j] - y[t, j])
                )
        ql[idx] = num
        idx += 1
    denom[denom == 0] = 1
    if (denom != 0).any():
        return (2 * ql / denom).reshape(batch, 1, 1)
    else:
        return None


def calc_loss(
    attack_data,
    forecasts,
    attack_idx,
    target_items,
    quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
):
    testset_size = sum(
        [
            attack_data[i].true_future_target.shape[0]
            for i in range(len(attack_data))
        ]
    )
    mse = {
        key: np.zeros((testset_size, len(attack_idx), len(target_items)))
        for key in forecasts.keys()
    }
    mape = {
        key: np.zeros((testset_size, len(attack_idx), len(target_items)))
        for key in forecasts.keys()
    }
    ql = {
        key: np.zeros((len(quantiles), testset_size, 1, len(target_items)))
        for key in forecasts.keys()
    }
    testset_idx = 0

    for i in tqdm(range(len(attack_data))):
        true_future_target = attack_data[i].true_future_target
        batch_size = true_future_target.shape[0]

        for attack_type in forecasts.keys():
            if (
                true_future_target[:, attack_idx][..., target_items] != 0
            ).prod() == 0:
                mape[attack_type][
                    testset_idx : testset_idx + batch_size
                ] = np.abs(
                    forecasts[attack_type][i][:, :, attack_idx][
                        ..., target_items
                    ].mean(1)
                    - true_future_target[:, attack_idx][..., target_items]
                )
                mse[attack_type][testset_idx : testset_idx + batch_size] = (
                    forecasts[attack_type][i][:, :, attack_idx][
                        ..., target_items
                    ].mean(1)
                    - true_future_target[:, attack_idx][..., target_items]
                ) ** 2
                for j, quantile in enumerate(quantiles):
                    pred = np.quantile(
                        a=forecasts[attack_type][i][
                            :, :, attack_idx, target_items
                        ],
                        q=quantile,
                        axis=1,
                    )
                    true = true_future_target[:, attack_idx][..., target_items]
                    ql[attack_type][
                        j, testset_idx : testset_idx + batch_size
                    ] = quantile_loss(true, pred, quantile)
            else:
                mape[attack_type][
                    testset_idx : testset_idx + batch_size
                ] = np.abs(
                    forecasts[attack_type][i][:, :, attack_idx][
                        ..., target_items
                    ].mean(1)
                    / true_future_target[:, attack_idx][..., target_items]
                    - 1
                )
                mse[attack_type][testset_idx : testset_idx + batch_size] = (
                    mape[attack_type][testset_idx : testset_idx + batch_size]
                    ** 2
                )
                for j, quantile in enumerate(quantiles):
                    pred = np.quantile(
                        a=forecasts[attack_type][i][:, :, attack_idx][
                            ..., target_items
                        ],
                        q=quantile,
                        axis=1,
                    )
                    true = true_future_target[:, attack_idx][..., target_items]
                    ql[attack_type][
                        j, testset_idx : testset_idx + batch_size
                    ] = quantile_loss(true, pred, quantile)
        testset_idx += batch_size

    return mse, mape, ql
