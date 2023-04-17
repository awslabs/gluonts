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
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from typing import List
from sparse_layer import SparseNet


def forward_model(model, inputs):
    samples = model(**inputs)
    sample_means = torch.mean(samples, axis=1)
    return samples, sample_means


def generate_adv_target(mode, future_target, factor=1.5):
    if mode == "over":
        adv_target = factor * future_target
    elif mode == "under":
        adv_target = factor * future_target
    elif mode == "zero":
        adv_target = torch.zeros_like(future_target)
    else:
        raise Exception("No such mode")
    return adv_target


class SparseLayerAttack:
    def __init__(self, model, params, input_names):
        self.model = model
        self.params = params
        self.input_names = input_names
        self.model.dropout_rate = 0
        self.model.rnn.dropout = 0
        self.model.train()

    def attack_batch(self, batch, true_future_target):
        res = {}
        batch_size, context_length, dim = batch["past_target_cdf"].shape
        max_norm = self.params.max_norm

        with torch.no_grad():
            inputs = {key: batch[key] for key in self.input_names}
            _, sample_mu = forward_model(self.model, inputs)
            future_target = sample_mu
            loss_function = nn.MSELoss(reduction="none")
            best_loss = [0] * len(self.params.sparsity)

        for mode in self.params.modes:
            print("Mode ", mode)

            # generate target
            adv_target = generate_adv_target(
                mode, future_target, self.params.factor
            )

            # traverse over each sparsity
            for i, s in enumerate(self.params.sparsity):
                # Generate sparse attack using sparse layer
                sparse_layer = SparseNet(
                    context_length=context_length,
                    target_dim=dim,
                    target_item=self.params.target_items[0],
                    hidden_dim=40,
                    m=s,
                    max_norm=max_norm,
                ).cuda()
                optimizer = optim.Adam(
                    sparse_layer.parameters(), lr=self.params.learning_rate
                )

                # iteration steps
                for _ in tqdm(range(self.params.n_iterations)):
                    optimizer.zero_grad()
                    sparse_delta = sparse_layer(
                        batch["past_target_cdf"], n_sample=100
                    ).view(batch_size, context_length, dim)
                    perturbed_inputs = dict(
                        [(key, batch[key]) for key in self.input_names]
                    )
                    perturbed_inputs["past_target_cdf"] = batch[
                        "past_target_cdf"
                    ] * (1 + sparse_delta)
                    _, perturbed_mu = forward_model(
                        self.model, perturbed_inputs
                    )
                    loss = loss_function(perturbed_mu, adv_target)[
                        :, self.params.attack_idx, self.params.target_items
                    ].mean()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    # construct perturbed inputs
                    perturbation = sparse_layer(
                        batch["past_target_cdf"], n_sample=100
                    ).view(batch_size, context_length, dim)
                    perturbed_inputs = {
                        key: batch[key] for key in self.input_names
                    }
                    perturbed_inputs["past_target_cdf"] = perturbed_inputs[
                        "past_target_cdf"
                    ] * (1 + perturbation)

                    # get model outputs
                    _, perturbed_mu = forward_model(
                        self.model, perturbed_inputs
                    )
                    # compute loss
                    loss = loss_function(
                        perturbed_mu[
                            :, self.params.attack_idx, self.params.target_items
                        ],
                        true_future_target[
                            :, self.params.attack_idx, self.params.target_items
                        ],
                    ).mean()
                    if loss >= best_loss[i]:
                        best_loss[i] = loss
                        res[s] = perturbation.data

        return res


class AttackLoss(nn.Module):
    def __init__(
        self,
        c: float,
        attack_idx: List[int],
        target_items: List[int],
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(AttackLoss, self).__init__()
        self.c = c
        self.attack_idx = attack_idx
        self.device = device
        self.target_items = target_items

    def forward(self, perturbation, mu, adv_target):
        # mu: batch x prediction_length x dim
        mu = mu[:, self.attack_idx, :]  # batch x 1 x dim

        adv_target = adv_target[:, self.attack_idx, :]  # batch x 1 x dim

        loss_function = nn.MSELoss(reduction="none")

        distance_tensor = loss_function(adv_target, mu)[
            ..., self.target_items
        ].sum(
            -1
        )  # batch x len(attack_idx)
        distance_per_sample = distance_tensor.sum(axis=1)  # batch
        distance = distance_per_sample.sum()  # scalar

        zero = torch.zeros(perturbation.shape).to(self.device)
        norm_tensor = loss_function(perturbation, zero).sum(
            -1
        )  # batch x context_length
        norm_per_sample = norm_tensor.sum(axis=1)  # batch
        norm = norm_per_sample.sum()  # scalar

        loss_per_sample = distance_per_sample
        loss = loss_per_sample.sum()

        return (
            norm_per_sample,
            distance_per_sample,
            loss_per_sample,
            norm,
            distance,
            loss,
        )


class AttackModule(nn.Module):
    def __init__(self, model, params, c, batch, input_names):
        super(AttackModule, self).__init__()

        self.model = model
        self.params = params
        self.c = c
        self.batch = batch
        self.input_names = input_names

        self.attack_loss = AttackLoss(
            c,
            attack_idx=self.params.attack_idx,
            device=self.params.device,
            target_items=self.params.target_items,
        )

        # Initialize perturbation: batch x context length x target dim
        self.perturbation = nn.Parameter(
            torch.zeros_like(batch["past_target_cdf"]), requires_grad=True
        )
        self.perturbation.to(self.params.device)

    # Returns predicted mean and scale
    def forward(self, num_parallel_samples=100):
        perturbed_inputs = dict(
            [(key, self.batch[key]) for key in self.input_names]
        )
        perturbed_inputs["past_target_cdf"] = self.batch["past_target_cdf"] * (
            1 + self.perturbation
        )
        _, samples = forward_model(self.model, perturbed_inputs)
        return samples

    def get_perturbation(self):
        return self.perturbation

    def get_grad(self):
        return self.perturbation.grad


class SparseAttackModule(nn.Module):
    def __init__(self, model, params, c, batch, input_names, attack_item):
        super(SparseAttackModule, self).__init__()

        self.model = model
        self.params = params
        self.c = c
        self.batch = batch
        self.input_names = input_names
        self.attack_item = attack_item

        self.attack_loss = AttackLoss(
            c,
            attack_idx=self.params.attack_idx,
            device=self.params.device,
            target_items=self.params.target_items,
        )

        # Initialize perturbation: batch x context length x target dim
        self.perturbation = nn.Parameter(
            torch.zeros_like(batch["past_target_cdf"]), requires_grad=True
        )
        self.perturbation.to(self.params.device)
        self.mask = torch.zeros_like(
            self.perturbation, device=self.params.device
        )
        self.mask[..., attack_item] = 1

    # Returns predicted mean and scale
    def forward(self, num_parallel_samples=100):
        perturbed_inputs = dict(
            [(key, self.batch[key]) for key in self.input_names]
        )
        perturbed_inputs["past_target_cdf"] = self.batch["past_target_cdf"] * (
            1 + self.perturbation * self.mask
        )
        _, samples = forward_model(self.model, perturbed_inputs)
        return samples

    def get_perturbation(self):
        return self.perturbation * self.mask

    def get_grad(self):
        return self.perturbation.grad * self.mask


class Attack:
    def __init__(self, model, params, input_names):
        self.model = model
        self.params = params
        self.input_names = input_names
        self.model.dropout_rate = 0
        self.model.rnn.dropout = 0
        self.model.train()
        self.max_norm = params.max_norm

    def attack_step(
        self, attack_module, optimizer, adv_target, ground_truth=None
    ):
        optimizer.zero_grad()

        perturbed_mu = attack_module()
        batch_size = perturbed_mu.shape[0]
        max_norm = self.max_norm

        # targeted loss
        _, _, _, _, _, loss = attack_module.attack_loss(
            attack_module.get_perturbation(), perturbed_mu, adv_target
        )

        # backward pass
        loss.backward()

        # renorm grad
        grad_norm = (
            attack_module.get_grad().view(batch_size, -1).norm(p=2, dim=1)
        )
        attack_module.perturbation.grad.div_(grad_norm.view(batch_size, 1, 1))

        # back-propogation
        optimizer.step()
        attack_module.perturbation.data.clamp_(min=-max_norm, max=max_norm)

    def attack_batch(self, batch, true_future_target):
        res = {}

        # generate adv target
        with torch.no_grad():
            inputs = {key: batch[key] for key in self.input_names}
            _, sample_mu = forward_model(self.model, inputs)
            future_target = sample_mu
            attack_idx = self.params.attack_idx
            target_items = self.params.target_items
            loss_function = nn.MSELoss(reduction="none")
            best_dense = 0

        for mode in self.params.modes:
            print("Mode: ", mode)

            # Generate dense attack
            attack_module = AttackModule(
                model=self.model,
                params=self.params,
                c=0,
                batch=batch,
                input_names=self.input_names,
            )
            adv_target = generate_adv_target(
                future_target=future_target,
                mode=mode,
                factor=self.params.factor,
            )

            optimizer = optim.Adam(
                [attack_module.perturbation], lr=self.params.learning_rate
            )

            for _ in tqdm(range(self.params.n_iterations)):
                self.attack_step(attack_module, optimizer, adv_target)

            # evaluate attack
            with torch.no_grad():
                perturbed_mu = attack_module()
                loss = loss_function(
                    true_future_target[:, attack_idx, target_items],
                    perturbed_mu[:, attack_idx, target_items],
                ).mean()
                if loss >= best_dense:
                    best_dense = loss
                    res["dense"] = attack_module.get_perturbation().data

        return res
