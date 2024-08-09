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

from typing import Tuple

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.model.simple_feedforward import make_linear_layer
from gluonts.torch.util import weighted_average


class BimModel(nn.Module):
    """
    Module implementing Bim model extended for probabilistic
    forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    hidden_dimension
        Size of last hidden layers in the feed-forward network.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dimension: int,
        mem_num: int,
        ep_mem_num: int,
        ep_topk: int,
        gamma: float = 1.0,
        distr_output=StudentTOutput(),
        scaling: str = "mean",
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimension = hidden_dimension
        self.mem_dim = hidden_dimension * prediction_length
        self.mem_num = mem_num
        self.ep_mem_num = ep_mem_num
        self.gamma = gamma
        self.ep_topk = ep_topk
        self.point = 0
        self.first = 1
        self.substitution = 1
        self.num_hard_example = 2

        self.distr_output = distr_output
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        # concat loc and scale to the context window
        self.linear_backbone = nn.Linear(context_length + 2, self.mem_dim)

        self.end_conv = nn.Conv1d(
            self.mem_dim * 3, self.mem_dim, kernel_size=1
        )
        self.end_conv2 = nn.Conv1d(
            self.mem_dim * 2, self.mem_dim, kernel_size=1
        )

        # memory
        self.memory = self.construct_memory()
        self.ep_frequency = self.construct_episodic_memory()

        # emission head
        self.args_proj = self.distr_output.get_args_proj(hidden_dimension)

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict["Memory"] = nn.Parameter(
            torch.randn(self.mem_num, self.mem_dim), requires_grad=True
        )
        memory_dict["Wq"] = nn.Parameter(
            torch.randn(self.mem_dim, self.mem_dim), requires_grad=True
        )
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def construct_episodic_memory(self):
        self.register_buffer(
            "episodic_memory", torch.zeros(self.mem_num, self.mem_dim)
        )
        ep_frequency = dict(zip(range(self.ep_mem_num), [1] * self.ep_mem_num))
        return ep_frequency

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
            },
            torch.zeros,
        )

    def combine_features(self, h, h_memory, h_ep):
        h_m_sum = torch.cat((h_memory * 1, h), dim=-1)
        h_all = torch.cat((h_m_sum, h_ep * self.gamma), dim=-1)
        y = self.end_conv(h_all.unsqueeze(-1)).squeeze(-1)
        return y

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )
        loc_scale = torch.cat(
            [loc.sign() * loc.abs().log1p(), scale.log()], dim=-1
        )
        h = self.linear_backbone(
            torch.cat([past_target_scaled, loc_scale], dim=-1)
        )
        h_memory, _, _, _ = self.query_memory(h)
        h_ep = self.query_episodic_memory(h)
        y = self.combine_features(h, h_memory, h_ep)

        distr_args = self.args_proj(
            y.reshape(-1, self.prediction_length, self.hidden_dimension)
        )

        if self.training:
            return (distr_args, loc, scale, h)
        else:
            return (distr_args, loc, scale)

    def query_episodic_memory(self, h_t: torch.Tensor):
        query = h_t
        top_k = self.ep_topk
        matched_idx, k_score = self.get_nearest_key(
            query, key_dict=self.episodic_memory, k=top_k
        )
        k_score = torch.softmax(k_score, dim=-1).unsqueeze(-1)
        mem = self.episodic_memory[matched_idx].detach()
        value = torch.sum(
            mem * k_score, dim=1
        )  # Changed from dim=2 to dim=1 due to reduced dimensionality

        frequency = torch.bincount(matched_idx.reshape(-1)).tolist()
        if self.first and torch.sum(self.episodic_memory) == 0:
            self.first = 0
            pass
        else:
            for f in range(len(frequency)):
                self.ep_frequency[f] += frequency[f]
        if self.point == 0:
            self.ep_frequency = dict(
                sorted(self.ep_frequency.items(), key=lambda x: x[1])
            )  # slow
            self.id = list(self.ep_frequency.keys())
        return value

    # We also need to implement the get_nearest_key method:
    def get_nearest_key(self, query, key_dict, k=3, sim="cosine"):
        if sim == "cosine":
            sim_func = nn.CosineSimilarity(dim=-1, eps=1e-6)
        else:
            raise NotImplementedError
        with torch.no_grad():
            key_dict = key_dict.unsqueeze(0)  # (1, K, d)
            similarity = sim_func(key_dict, query.unsqueeze(1))  # (B, K)
            topk = torch.topk(similarity, k)
            k_score = topk.values
            k_index = topk.indices

        return k_index, k_score

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory["Wq"])  # (B, d)
        att_score = torch.softmax(
            torch.matmul(
                query, self.memory["Memory"].t() / (self.mem_dim**0.5)
            ),
            dim=-1,
        )  # alpha: (B, M)
        value = torch.matmul(att_score, self.memory["Memory"])  # (B, d)
        _, matched_idx = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory["Memory"][matched_idx[:, 0]]  # B, d
        neg = self.memory["Memory"][matched_idx[:, 1]]  # B, d
        return value, query, pos, neg

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            distr_args, loc, scale, h = self(
                past_target=past_target,
                past_observed_values=past_observed_values,
            )
            loss = self.distr_output.loss(
                target=future_target,
                distr_args=distr_args,
                loc=loc,
                scale=scale,
            )

            return weighted_average(
                loss, weights=future_observed_values, dim=-1
            ), h
        else:
            distr_args, loc, scale = self(
                past_target=past_target,
                past_observed_values=past_observed_values,
            )
            loss = self.distr_output.loss(
                target=future_target,
                distr_args=distr_args,
                loc=loc,
                scale=scale,
            )

            return weighted_average(
                loss, weights=future_observed_values, dim=-1
            )

    def update_episodic_memory(self, h, nll):
        # update the episodic memory with the most challenging targets
        _, index = torch.topk(nll, k=self.num_hard_example, largest=True)
        hard_examples = h[index]

        # Update episodic memory
        for i, hard_example in enumerate(hard_examples):
            memory_index = self.id[self.point]
            self.episodic_memory[memory_index, :] = hard_example.data
            self.ep_frequency[memory_index] = 0
            self.point += 1
            self.point %= self.num_hard_example * self.substitution

            # Break if we've updated all available memory slots
            if i >= self.ep_mem_num - 1:
                break

        # If self.point has wrapped around, update self.id
        if self.point == 0:
            self.ep_frequency = dict(
                sorted(self.ep_frequency.items(), key=lambda x: x[1])
            )
            self.id = list(self.ep_frequency.keys())
