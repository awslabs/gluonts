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
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import copy
import random
import math
from pts.trainers.timer import Timer


class SAdam:
    def __init__(
        self,
        freq: int = 1,
        task_name: str = "none",
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        num_workers: int = 1,
        pin_memory: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        num_strata: int = 8,
        nesterov: bool = False,
        decreasing_step_size: bool = False,
        weighted_batch: bool = True,
        validation_freq: int = 5,
        eval_model: bool = False,
        gamma: int = 0.7,
        tensorboard_path: str = "./runs/SAdam",
        device: Optional[torch.device] = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_strata = num_strata
        self.task_name = task_name
        self.freq = freq
        self.decreasing_step_size = decreasing_step_size
        self.weighted_batch = weighted_batch
        self.validation_freq = validation_freq
        self.eval_model = eval_model
        self.tensorboard_path = tensorboard_path
        self.gamma = gamma

    def inference(self, model, inputs, weighted_batch=False, group_ratio=None):
        output = model(*inputs)
        if weighted_batch:
            y = torch.ones(output.shape).to(output.get_device())
            for i in range(self.batch_size):
                for j in range(len(group_ratio)):
                    y[i * len(group_ratio) + j] *= group_ratio[j]
            output = (output * y).mean() * self.num_strata
        else:
            output = output.mean()
        if isinstance(output, (list, tuple)):
            loss = output[0]
        else:
            loss = output
        return loss

    def __call__(
        self, net: nn.Module, input_names: List[str], data_loaders
    ) -> None:
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        writer = SummaryWriter(self.tensorboard_path)

        timer = Timer()

        training_iter = iter(data_loaders["training_data_loader"])
        anchor_iter = iter(data_loaders["anchor_data_loader"])

        group_ratio = (
            data_loaders["group_ratio"] if self.weighted_batch else None
        )
        avg_epoch_grad = 0.0
        v_0_norm = 0.0
        v_t_norm = 0.0
        for epoch_no in range(self.epochs):
            if self.decreasing_step_size:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 1 / math.sqrt(epoch_no + 1)
            for batch_no in range(self.num_batches_per_epoch):
                iter_n = epoch_no * self.num_batches_per_epoch + batch_no
                if (
                    iter_n == 0
                    or v_t_norm <= self.gamma * v_0_norm
                    or iter_n % self.freq == 0
                ):
                    anchor_model = copy.deepcopy(net)
                    sg_model = copy.deepcopy(net)
                    anchor_model.zero_grad()
                    with timer("gradient oracle"):
                        data_entry = next(anchor_iter)
                        inputs = [
                            data_entry[k].to(self.device) for k in input_names
                        ]
                        loss = self.inference(
                            anchor_model,
                            inputs,
                            weighted_batch=self.weighted_batch,
                            group_ratio=group_ratio,
                        )
                        loss.backward()
                    for p in anchor_model.parameters():
                        if p.grad is None:
                            continue
                        v_0_norm += torch.norm(p.grad.data) ** 2

                v_t_norm = 0.0
                data_entry = next(training_iter)
                optimizer.zero_grad()
                with timer("gradient oracle"):
                    inputs = [
                        data_entry[k].to(self.device) for k in input_names
                    ]
                    inputs_ = copy.deepcopy(inputs)
                    net.zero_grad()
                    sg_model.zero_grad()
                    loss = self.inference(sg_model, inputs)
                    loss.backward()

                loss = self.inference(net, inputs_)
                loss.backward()
                with timer("gradient oracle"):
                    for p1, p2, p3 in zip(
                        net.parameters(),
                        sg_model.parameters(),
                        anchor_model.parameters(),
                    ):
                        if (
                            p1.grad is None
                            or p2.grad is None
                            or p3.grad is None
                        ):
                            continue
                        v_t = torch.zeros_like(p1.grad.data, device=p1.device)
                        v_t.add_(p1.grad.data - p2.grad.data + p3.grad.data)
                        p1.grad.data.zero_().add_(v_t)
                        v_t_norm += torch.norm(v_t) ** 2
                    optimizer.step()

            # compute the gradient norm and loss over training set
            avg_epoch_loss = 0.0
            full_batch_iter = iter(data_loaders["full_batch_loader"])
            net.zero_grad()
            for i, data_entry in enumerate(full_batch_iter):
                inputs = [data_entry[k].to(self.device) for k in input_names]
                loss = self.inference(net, inputs)
                loss.backward()
                avg_epoch_loss += loss.item()
            avg_epoch_loss /= i + 1
            epoch_grad = 0.0
            for p in net.parameters():
                if p.grad is None:
                    continue
                epoch_grad += torch.norm(p.grad.data / (i + 1))
            net.zero_grad()

            # compute the validation loss
            if self.eval_model and epoch_no % self.validation_freq == 0:
                net_validate = copy.deepcopy(net)
                validation_iter = iter(data_loaders["validation_data_loader"])
                validation_loss = 0.0
                with torch.no_grad():
                    for i, data_entry in enumerate(validation_iter):
                        net_validate.zero_grad()
                        inputs = [
                            data_entry[k].to(self.device) for k in input_names
                        ]
                        loss = self.inference(net_validate, inputs)
                        validation_loss += loss.item()
                validation_loss /= i + 1

            num_iters = (
                self.num_batches_per_epoch
                * (epoch_no + 1)
                * 2
                * self.batch_size
                + self.num_batches_per_epoch
                / self.freq
                * self.num_strata
                * (epoch_no + 1)
                * self.batch_size
            )
            avg_epoch_grad = (avg_epoch_grad * epoch_no + epoch_grad) / (
                epoch_no + 1
            )
            time_in_ms = timer.totals["gradient oracle"] * 1000
            writer.add_scalar(
                "gradnorm/iters",
                avg_epoch_grad,
                (epoch_no + 1) * self.num_batches_per_epoch,
            )
            writer.add_scalar("gradnorm/grads", avg_epoch_grad, num_iters)
            writer.add_scalar("gradnorm/time", avg_epoch_grad, time_in_ms)
            writer.add_scalar(
                "train_loss/iters",
                avg_epoch_loss,
                (epoch_no + 1) * self.num_batches_per_epoch,
            )
            writer.add_scalar("train_loss/grads", avg_epoch_loss, num_iters)
            writer.add_scalar("train_loss/time", avg_epoch_loss, time_in_ms)
            if self.eval_model and epoch_no % self.validation_freq == 0:
                writer.add_scalar(
                    "val_loss/iters",
                    validation_loss,
                    (epoch_no + 1) * self.num_batches_per_epoch,
                )
                writer.add_scalar("val_loss/grads", validation_loss, num_iters)
                writer.add_scalar("val_loss/time", validation_loss, time_in_ms)
                print(
                    "\nTraining Loss: {:.4f}, Test Loss: {:.4f}\n".format(
                        avg_epoch_loss, validation_loss
                    )
                )
            else:
                print(f"\nTraining Loss: {avg_epoch_loss:.4f} \n")
            print("Epoch ", epoch_no, " is done!")

        writer.close()
        print(
            "task: "
            + self.task_name
            + " on SAdam with lr="
            + str(self.learning_rate)
            + " is done!"
        )
