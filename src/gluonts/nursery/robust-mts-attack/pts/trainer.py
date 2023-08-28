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

import time
from typing import List, Optional, Union

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gluonts.core.component import validated


def requires_grad_(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


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


class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-2,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        optimizer = Adam(
            net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1

            # training loop
            with tqdm(train_iter, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    inputs = []
                    for v in data_entry.values():
                        if v.ndim >= 4:
                            v = v.reshape(v.shape[:3])
                        inputs.append(v.to(self.device))
                    # inputs = [v.to(self.device) for v in data_entry.values()]

                    output = net(*inputs)

                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                        },
                        refresh=False,
                    )

                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(
                            net.parameters(), self.clip_gradient
                        )

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break
                it.close()

            # validation loop
            if validation_iter is not None:
                cumm_epoch_loss_val = 0.0
                with tqdm(validation_iter, total=total, colour="green") as it:
                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [
                            v.to(self.device) for v in data_entry.values()
                        ]
                        with torch.no_grad():
                            output = net(*inputs)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        cumm_epoch_loss_val += loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                            },
                            refresh=False,
                        )

                        if self.num_batches_per_epoch == batch_no:
                            break

                it.close()

            # mark epoch end time and log time cost of current epoch
            toc = time.time()


class Trainer_adv(Trainer):
    @validated()
    def __init__(
        self,
        sparse_net,
        noise_sd=0.1,
        clamp=False,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-2,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device
        self.sparse_net = sparse_net
        self.noise_sd = noise_sd
        self.clamp = clamp

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        optimizer = Adam(
            net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        optimizer_adv = Adam(
            self.sparse_net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )
        loss_function = nn.MSELoss()

        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1

            # training loop
            with tqdm(train_iter, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    # train sparse net
                    self.sparse_net.zero_grad()
                    inputs = {
                        key: data_entry[key].to(self.device)
                        for key in data_entry.keys()
                    }
                    shapes = data_entry["past_target_cdf"].shape
                    delta = self.sparse_net(
                        data_entry["past_target_cdf"].to(self.device),
                        n_sample=100,
                    ).view(shapes)
                    if self.clamp:
                        delta = torch.clamp(
                            delta,
                            max=self.sparse_net.max_norm
                            * data_entry["past_target_cdf"].abs().max(),
                        )
                    inputs["past_target_cdf"] += delta
                    output = net(**inputs)
                    mu, target = output[2], output[-1]
                    loss_sparse = -loss_function(mu, target)
                    loss_sparse.backward()
                    optimizer_adv.step()

                    # train forecasting model
                    net.zero_grad()
                    perturbed_inputs = {
                        key: data_entry[key].to(self.device)
                        for key in data_entry.keys()
                    }
                    shapes = data_entry["past_target_cdf"].shape
                    delta = self.sparse_net(
                        data_entry["past_target_cdf"].to(self.device),
                        n_sample=100,
                    ).view(shapes)
                    perturbed_inputs["past_target_cdf"] += delta
                    output = net(**perturbed_inputs)

                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output
                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                        },
                        refresh=False,
                    )

                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(
                            net.parameters(), self.clip_gradient
                        )

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break

                it.close()

            # validation loop
            if validation_iter is not None:
                cumm_epoch_loss_val = 0.0
                with tqdm(validation_iter, total=total, colour="green") as it:
                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [
                            v.to(self.device) for v in data_entry.values()
                        ]
                        with torch.no_grad():
                            output = net(*inputs)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        cumm_epoch_loss_val += loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                            },
                            refresh=False,
                        )

                        if self.num_batches_per_epoch == batch_no:
                            break

                it.close()

            # mark epoch end time and log time cost of current epoch
            toc = time.time()
            # print('Epoch: {}\tTime: {:.1f}'.format(epoch_no, toc - tic))
