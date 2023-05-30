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

from typing import List, Optional

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.torch.modules.feature import FeatureEmbedder


class LookupValues(nn.Module):
    @validated()
    def __init__(self, bin_values: torch.Tensor):
        super().__init__()
        self.register_buffer("bin_values", bin_values)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.clamp(indices, 0, self.bin_values.shape[0] - 1)
        return torch.index_select(
            self.bin_values, 0, indices.reshape(-1)
        ).view_as(indices)


class CausalDilatedResidualLayer(nn.Module):
    @validated()
    def __init__(
        self,
        n_residual_channels: int,
        n_skip_channels: int,
        dilation: int,
        kernel_size: int,
        return_dense_output: bool,
    ):
        super().__init__()
        self.n_residual_channels = n_residual_channels
        self.n_skip_channels = n_skip_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.return_dense_output = return_dense_output

        # Modules
        self.conv_sigmoid = nn.Sequential(
            nn.Conv1d(
                in_channels=n_residual_channels,
                out_channels=n_residual_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.Sigmoid(),
        )
        self.conv_tanh = nn.Sequential(
            nn.Conv1d(
                in_channels=n_residual_channels,
                out_channels=n_residual_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.Tanh(),
        )
        self.conv_skip = nn.Conv1d(
            in_channels=n_residual_channels,
            out_channels=n_skip_channels,
            kernel_size=1,
        )

        if self.return_dense_output:
            self.conv_residual = nn.Conv1d(
                in_channels=n_residual_channels,
                out_channels=n_residual_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.conv_sigmoid(x) * self.conv_tanh(x)
        s = self.conv_skip(u)
        if not self.return_dense_output:
            return s, torch.zeros_like(u)

        out = self.conv_residual(u)
        out = out + x[..., (self.kernel_size - 1) * self.dilation :]

        return s, out


class WaveNet(nn.Module):
    @validated()
    def __init__(
        self,
        pred_length: int,
        bin_values: List[float],
        n_residual_channels: int,
        n_skip_channels: int,
        dilation_depth: int,
        n_stacks: int,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        n_parallel_samples: int = 100,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.dilation_depth = dilation_depth
        self.prediction_length = pred_length
        self.n_parallel_samples = n_parallel_samples
        self.temperature = temperature
        self.num_features = (
            embedding_dimension * len(cardinality)
            + num_feat_dynamic_real
            + num_feat_static_real
            + 1  # the log(scale)
        )

        self.n_bins = len(bin_values) + 1
        self.dilations = self._get_dilations(dilation_depth, n_stacks)
        self.receptive_field = self.get_receptive_field(
            dilation_depth, n_stacks
        )
        self.trim_lengths = [
            sum(self.dilations) - sum(self.dilations[: i + 1])
            for i, _ in enumerate(self.dilations)
        ]

        # Modules
        self.feature_embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=[embedding_dimension] * len(cardinality),
        )
        self.target_embedder = nn.Embedding(
            num_embeddings=self.n_bins, embedding_dim=n_residual_channels
        )
        self.residuals = nn.ModuleList()
        for i, d in enumerate(self.dilations):
            self.residuals.add_module(
                f"residual_layer_{i}",
                CausalDilatedResidualLayer(
                    n_residual_channels=n_residual_channels,
                    n_skip_channels=n_skip_channels,
                    dilation=d,
                    kernel_size=2,
                    return_dense_output=i + 1 < len(self.dilations),
                ),
            )

        self.conv_project = nn.Conv1d(
            in_channels=n_residual_channels + self.num_features,
            out_channels=n_residual_channels,
            kernel_size=1,
            bias=True,
        )
        with torch.no_grad():
            self.conv_project.bias.zero_()

        self.conv1 = nn.Conv1d(
            in_channels=n_skip_channels,
            out_channels=n_skip_channels,
            kernel_size=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_skip_channels,
            out_channels=self.n_bins,
            kernel_size=1,
        )
        self.output_act = nn.ELU()
        self.lookup_values = LookupValues(
            torch.tensor(bin_values, dtype=torch.float32)
        )
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    @staticmethod
    def _get_dilations(dilation_depth, n_stacks):
        return [2**i for i in range(dilation_depth)] * n_stacks

    @staticmethod
    def get_receptive_field(dilation_depth, n_stacks):
        dilations = WaveNet._get_dilations(
            dilation_depth=dilation_depth, n_stacks=n_stacks
        )
        return sum(dilations) + 1

    def get_full_features(
        self,
        feat_static_cat: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_observed_values: Optional[torch.Tensor],
        scale: torch.Tensor,
    ):
        embedded_cat = self.feature_embedder(feat_static_cat.long())
        static_feat = torch.cat([embedded_cat, torch.log(scale + 1.0)], dim=1)
        repeated_static_feat = torch.repeat_interleave(
            static_feat[..., None],
            self.prediction_length + self.receptive_field,
            dim=-1,
        )

        if future_observed_values is None:
            future_observed_values = torch.ones_like(future_time_feat[:, 0])

        full_observed = torch.cat(
            [past_observed_values, future_observed_values], dim=-1
        ).unsqueeze(dim=1)

        full_time_features = torch.cat(
            [past_time_feat, future_time_feat], dim=-1
        )
        full_features = torch.cat(
            [full_time_features, full_observed, repeated_static_feat], dim=1
        )
        return full_features

    def target_feature_embedding(
        self, target: torch.Tensor, features: torch.Tensor
    ):
        out = self.target_embedder(target)
        out = torch.transpose(out, 1, 2)
        out = torch.cat([out, features], dim=1)
        out = self.conv_project(out)
        return out

    def base_net(
        self,
        inputs: torch.Tensor,
        prediction_mode: bool = False,
        queues: List[torch.Tensor] = None,
    ):
        if prediction_mode:
            assert (
                queues is not None
            ), "Queues cannot be empty in prediction mode!"

        skip_outs = []
        queues_next = []
        out = inputs
        for i, layer in enumerate(self.residuals):
            skip, out = layer(out)
            if prediction_mode:
                trimmed_skip = skip
                if i + 1 < len(self.residuals):
                    out = torch.cat([queues[i], out], dim=-1)
                    queues_next.append(out[..., 1:])
            else:
                trimmed_skip = skip[..., self.trim_lengths[i] :]
            skip_outs.append(trimmed_skip)

        y = torch.stack(skip_outs).sum(dim=0)
        y = self.output_act(y)
        y = self.conv1(y)
        y = self.output_act(y)
        y = self.conv2(y)
        logits = y.transpose(1, 2)
        return logits, queues_next

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        scale: torch.Tensor,
    ):
        full_target = torch.cat([past_target, future_target], dim=-1).long()
        full_features = self.get_full_features(
            feat_static_cat=feat_static_cat,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_observed_values=future_observed_values,
            scale=scale,
        )
        input_embedding = self.target_feature_embedding(
            target=full_target[..., :-1], features=full_features[..., 1:]
        )
        logits, _ = self.base_net(input_embedding, prediction_mode=False)
        # logits = (batch, pred_length, n_bins)
        labels = full_target[..., self.receptive_field :]
        # label = (batch, pred_length)
        loss_weight = torch.cat(
            [past_observed_values, future_observed_values], dim=-1
        )[..., self.receptive_field :]
        # loss_weight = (batch, pred_length)
        assert labels.size() == loss_weight.size()
        loss = self.criterion(
            logits.reshape(-1, self.n_bins), labels.reshape(-1)
        ) * loss_weight.reshape(-1)
        loss = loss.view_as(labels)
        return loss

    def _initialize_conv_queues(
        self, past_target: torch.Tensor, features: torch.Tensor
    ):
        out = self.target_feature_embedding(past_target, features)
        queues = []
        for i, (d, layer) in enumerate(zip(self.dilations, self.residuals)):
            sz = 1 if d == 2 ** (self.dilation_depth - 1) else d * 2
            _, out = layer(out)

            if i + 1 < len(self.dilations):
                out_chunk = out[..., -sz - 1 : -1]
            else:
                out_chunk = out
            queues.append(out_chunk)
        return queues

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        scale: torch.Tensor,
    ):
        past_target = past_target.long()
        full_features = self.get_full_features(
            feat_static_cat=feat_static_cat,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_observed_values=None,
            scale=scale,
        )

        # To compute queues for the first step, we need features from
        # -self.pred_length - self.receptive_field + 1 to -self.pred_length + 1
        features_start_idx = -self.prediction_length - self.receptive_field + 1
        features_end_idx = (
            -self.prediction_length + 1 if self.prediction_length > 1 else None
        )
        queues = self._initialize_conv_queues(
            past_target=past_target[..., -self.receptive_field :],
            features=full_features[
                ...,
                features_start_idx:features_end_idx,
            ],
        )

        queues = [
            torch.repeat_interleave(q, self.n_parallel_samples, dim=0)
            for q in queues
        ]

        res = torch.repeat_interleave(
            past_target[..., -2:], self.n_parallel_samples, dim=0
        )

        for t in range(self.prediction_length):
            current_target = res[..., -2:]
            current_features = full_features[
                ...,
                self.receptive_field + t - 1 : self.receptive_field + t + 1,
            ]
            input_embedding = self.target_feature_embedding(
                current_target,
                torch.repeat_interleave(
                    current_features, self.n_parallel_samples, dim=0
                ),
            )
            logits, queues = self.base_net(
                input_embedding, prediction_mode=True, queues=queues
            )

            if self.temperature > 0.0:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                y = torch.multinomial(probs.view(-1, self.n_bins), 1).view(
                    logits.size()[:-1]
                )
            else:
                y = torch.argmax(logits, dim=-1)
            y = y.long()
            res = torch.cat([res, y], dim=-1)

        samples = res[..., -self.prediction_length :]
        samples = samples.view(
            -1, self.n_parallel_samples, self.prediction_length
        )
        samples = self.lookup_values(samples)
        samples = samples * scale[:, None]
        return samples
