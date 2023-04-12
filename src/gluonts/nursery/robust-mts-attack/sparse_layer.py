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
import torch.nn.functional as F
import torch.distributions
from torch.distributions import Normal
from abc import ABC
import numpy as np
import warnings

warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os, pickle

hidden_dim = 100
sparsity = 100
sampling_size = 20


def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output)


def load_object(filename):
    with open(filename, "rb") as output:
        return pickle.load(output)


def check(FOLDER):
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)


def seed(np_seed=11041987, torch_seed=20051987):
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(torch_seed)


class SparseNet(
    nn.Module, ABC
):  # accept x, return w for which E[|| w ||_0] <= m -- our sparse layer
    def __init__(
        self,
        context_length,
        target_dim,
        target_item,
        hidden_dim,
        m,
        max_norm,
        norm=True,
    ):
        super(SparseNet, self).__init__()
        self.input_dim = context_length * target_dim
        self.context_length = context_length
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.m = m
        self.norm = norm
        self.gamma = nn.Parameter(
            -1.0 * torch.ones((1, self.target_dim)), requires_grad=True
        ).to(device)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim).to(device)
        self.fc21 = nn.Linear(hidden_dim, self.input_dim).to(device)
        self.fc22 = nn.Linear(hidden_dim, self.input_dim).to(device)
        self.bn = nn.BatchNorm1d(hidden_dim).to(device)
        self.dist = Normal(
            torch.FloatTensor([0.0]).to(device),
            torch.FloatTensor([1.0]).to(device),
        )
        self.target_item = target_item
        self.max_norm = max_norm
        # self.use_mean = use_mean

    def _r(self):  # output is of size 1 by input_dim
        return (
            (self.m / np.sqrt(self.target_dim))
            * torch.exp(0.5 * self.gamma)
            / torch.sqrt(torch.sum(torch.exp(self.gamma)))
        )

    def forward(self, x, n_sample=100):  # x is batch_size by input_dim
        x = x.view(x.shape[0], self.input_dim)
        x = self.fc1(x)
        x_reshape = x.view(-1, self.fc1.out_features)
        if (x_reshape.shape[0] > 1) and (self.norm is True):
            shape = x.shape
            x = x.view(-1, self.fc1.out_features)
            x = F.relu(self.bn(x)).view(shape)
        else:
            x = F.relu(x)
        mu, log_var = self.fc21(x), self.fc22(x)
        std = torch.exp(0.5 * log_var)
        eps = (
            torch.empty(n_sample, mu.shape[0], self.input_dim)
            .normal_(0, 1)
            .to(device)
        )
        w = mu + eps * std
        w = w.mean(0)
        u = torch.empty(mu.shape[0], self.target_dim).normal_(0, 1).to(device)
        r = self._r()
        r = r.repeat(mu.shape[0], 1)
        mask = (u <= self.dist.icdf(r)).to(device)  # batch x target_dim
        mask_target_item = torch.ones(mu.shape[0], self.target_dim).to(device)
        if self.target_item is not None:
            mask_target_item[:, self.target_item] = 0

        # renorm output
        output = w.view(-1, self.context_length, self.target_dim) * (
            mask.float() * mask_target_item
        ).reshape(-1, 1, self.target_dim).repeat(
            1, self.context_length, 1
        )  # batch x context x target dim
        output.clamp_(min=-self.max_norm, max=self.max_norm)
        return output.view(
            -1, self.context_length, self.target_dim
        )  # output size = batch_size * input_dim
