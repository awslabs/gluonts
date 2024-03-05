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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.lookup_table import LookupValues


class CausalDilatedResidualLayer(nn.Module):
    @validated()
    def __init__(
        self,
        num_residual_channels: int,
        num_skip_channels: int,
        dilation: int,
        kernel_size: int,
        return_dense_output: bool,
    ):
        super().__init__()
        self.num_residual_channels = num_residual_channels
        self.num_skip_channels = num_skip_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.return_dense_output = return_dense_output

        # Modules
        self.conv_sigmoid = nn.Sequential(
            nn.Conv1d(
                in_channels=num_residual_channels,
                out_channels=num_residual_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.Sigmoid(),
        )
        self.conv_tanh = nn.Sequential(
            nn.Conv1d(
                in_channels=num_residual_channels,
                out_channels=num_residual_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.Tanh(),
        )
        self.conv_skip = nn.Conv1d(
            in_channels=num_residual_channels,
            out_channels=num_skip_channels,
            kernel_size=1,
        )

        if self.return_dense_output:
            self.conv_residual = nn.Conv1d(
                in_channels=num_residual_channels,
                out_channels=num_residual_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.conv_sigmoid(x) * self.conv_tanh(x)
        s = self.conv_skip(u)
        if not self.return_dense_output:
            return s, torch.zeros_like(u)

        out = self.conv_residual(u)
        out = out + x[..., (self.kernel_size - 1) * self.dilation :]

        return s, out


class WaveNet(nn.Module):
    """
    The WaveNet model.

    Parameters
    ----------
    pred_length
        Prediction length.
    bin_values
        List of bin values.
    num_residual_channels
        Number of residual channels.
    num_skip_channels
        Number of skip channels.
    dilation_depth
        The depth of the dilated convolution.
    num_stacks
        The number of dilation stacks.
    num_feat_dynamic_real, optional
        The number of dynamic real features, by default 1
    num_feat_static_real, optional
        The number of static real features, by default 1
    cardinality, optional
        The cardinalities of static categorical features, by default [1]
    embedding_dimension, optional
        The dimension of the embeddings for categorical features, by default 5
    num_parallel_samples, optional
        The number of parallel samples to generate during inference.
        This parameter is only used in inference mode, by default 100
    temperature, optional
        Temparature used for sampling from the output softmax distribution,
        by default 1.0
    """

    @validated()
    def __init__(
        self,
        pred_length: int,
        bin_values: List[float],
        num_residual_channels: int,
        num_skip_channels: int,
        dilation_depth: int,
        num_stacks: int,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        num_parallel_samples: int = 100,
        temperature: float = 1.0,
        use_log_scale_feature: bool = True,
    ):
        super().__init__()

        self.dilation_depth = dilation_depth
        self.prediction_length = pred_length
        self.num_parallel_samples = num_parallel_samples
        self.temperature = temperature
        self.num_features = (
            embedding_dimension * len(cardinality)
            + num_feat_dynamic_real
            + num_feat_static_real
            + int(use_log_scale_feature)  # the log(scale)
            + 1  # for observed value indicator
        )
        self.use_log_scale_feature = use_log_scale_feature

        # 1 extra bin to accounts for extreme values
        self.n_bins = len(bin_values) + 1
        self.dilations = self._get_dilations(dilation_depth, num_stacks)
        self.receptive_field = self.get_receptive_field(
            dilation_depth, num_stacks
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
            num_embeddings=self.n_bins, embedding_dim=num_residual_channels
        )
        self.residuals = nn.ModuleList()
        for i, d in enumerate(self.dilations):
            self.residuals.add_module(
                f"residual_layer_{i}",
                CausalDilatedResidualLayer(
                    num_residual_channels=num_residual_channels,
                    num_skip_channels=num_skip_channels,
                    dilation=d,
                    kernel_size=2,
                    return_dense_output=i + 1 < len(self.dilations),
                ),
            )

        self.conv_project = nn.Conv1d(
            in_channels=num_residual_channels + self.num_features,
            out_channels=num_residual_channels,
            kernel_size=1,
            bias=True,
        )
        with torch.no_grad():
            assert self.conv_project.bias is not None
            self.conv_project.bias.zero_()

        self.conv1 = nn.Conv1d(
            in_channels=num_skip_channels,
            out_channels=num_skip_channels,
            kernel_size=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_skip_channels,
            out_channels=self.n_bins,
            kernel_size=1,
        )
        self.output_act = nn.ELU()
        self.lookup_values = LookupValues(
            torch.tensor(bin_values, dtype=torch.float32)
        )
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    @staticmethod
    def _get_dilations(dilation_depth: int, num_stacks: int) -> List[int]:
        return [2**i for i in range(dilation_depth)] * num_stacks

    @staticmethod
    def get_receptive_field(dilation_depth: int, num_stacks: int) -> int:
        dilations = WaveNet._get_dilations(
            dilation_depth=dilation_depth, num_stacks=num_stacks
        )
        return sum(dilations) + 1

    def get_full_features(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_observed_values: Optional[torch.Tensor],
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepares the inputs for the network by repeating static feature and
        concatenating it with time features and observed value indicator.

        Parameters
        ----------
        feat_static_cat
            Static categorical features: (batch_size, num_cat_features)
        feat_static_real
            Static real-valued features: (batch_size, num_feat_static_real)
        past_observed_values
            Observed value indicator for the past target: (batch_size,
            receptive_field)
        past_time_feat
            Past time features: (batch_size, num_time_features,
            receptive_field)
        future_time_feat
            Future time features: (batch_size, num_time_features, pred_length)
        future_observed_values
            Observed value indicator for the future target:
            (batch_size, pred_length). This will be set to all ones, if not
            provided (e.g., during inference)
        scale
            scale of the time series: (batch_size, 1)

        Returns
        -------
            A tensor containing all the features ready to be passed through the
            network.
            Shape: (batch_size, num_features, receptive_field + pred_length)
        """
        static_feat = self.feature_embedder(feat_static_cat.long())
        if self.use_log_scale_feature:
            static_feat = torch.cat(
                [static_feat, torch.log(scale + 1.0)], dim=1
            )
        static_feat = torch.cat([static_feat, feat_static_real], dim=1)
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
    ) -> torch.Tensor:
        """
        Provides a joint embedding for the target and features.

        Parameters
        ----------
        target
            Full target of shape (batch_size, sequence_length)
        features
            Full features of shape (batch_size, num_features, sequence_length)

        Returns
        -------
            A tensor containing a joint embedding of target and features.
            Shape: (batch_size, n_residue, sequence_length)
        """
        out = self.target_embedder(target)
        out = torch.transpose(out, 1, 2)
        out = torch.cat([out, features], dim=1)
        out = self.conv_project(out)
        return out

    def base_net(
        self,
        inputs: torch.Tensor,
        queues: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the WaveNet.

        Parameters
        ----------
        inputs
            A tensor of inputs
            Shape: (batch_size, num_residual_channels, sequence_length)
        queues, optional
            Convolutional queues containing past computations.
            This speeds up predictions and must be provided
            during prediction mode. See [Paine et al., 2016] for details,
            by default None

        [Paine et al., 2016] "Fast wavenet generation algorithm."
           arXiv preprint arXiv:1611.09482 (2016).

        Returns
        -------
            A tensor containing the unnormalized outputs of the network of
            shape (batch_size, pred_length, num_bins) and a list containing the
            convolutional queues for each layer. The queue corresponding to
            layer `l` has shape: (batch_size, num_residual_channels, 2^l).
        """
        skip_outs = []
        queues_next = []
        out = inputs
        for i, layer in enumerate(self.residuals):
            skip, out = layer(out)
            if queues is not None:
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
        feat_static_real: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the training loss for the wavenet model.

        Parameters
        ----------
        feat_static_cat
            Static categorical features: (batch_size, num_cat_features)
        feat_static_real
            Static real-valued features: (batch_size, num_feat_static_real)
        past_target
            Past target: (batch_size, receptive_field)
        past_observed_values
            Observed value indicator for the past target: (batch_size,
            receptive_field)
        past_time_feat
            Past time features: (batch_size, num_time_features,
            receptive_field)
        future_time_feat
            Future time features: (batch_size, num_time_features, pred_length)
        future_target
            Target on which the loss is computed: (batch_size, pred_length)
        future_observed_values
            Observed value indicator for the future target:
            (batch_size, pred_length). This will be set to all ones, if not
            provided (e.g., during inference)
        scale
            Scale of the time series: (batch_size, 1)

        Returns
        -------
            Loss tensor with shape (batch_size, pred_length)
        """
        full_target = torch.cat([past_target, future_target], dim=-1).long()
        full_features = self.get_full_features(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_observed_values=future_observed_values,
            scale=scale,
        )
        input_embedding = self.target_feature_embedding(
            target=full_target[..., :-1], features=full_features[..., 1:]
        )
        logits, _ = self.base_net(input_embedding)
        labels = full_target[..., self.receptive_field :]
        loss_weight = torch.cat(
            [past_observed_values, future_observed_values], dim=-1
        )[..., self.receptive_field :]

        assert labels.size() == loss_weight.size()
        loss = self.criterion(
            logits.reshape(-1, self.n_bins), labels.reshape(-1)
        ) * loss_weight.reshape(-1)
        loss = loss.view_as(labels)
        return loss

    def _initialize_conv_queues(
        self, past_target: torch.Tensor, features: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Initialize the convolutional queues to speed up predictions.

        Parameters
        ----------
        past_target
            Past target: (batch_size, receptive_field)
        features
            Tensor of features: (batch_size, num_features, receptive_field)

        Returns
        -------
            A list containing the convolutional queues for each layer.
            The queue corresponding to layer `l` has shape:
            (batch_size, n_residue, 2^l).
        """
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
        feat_static_real: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        scale: torch.Tensor,
        prediction_length: Optional[int] = None,
        num_parallel_samples: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate predictions from the WaveNet model.

        Parameters
        ----------
        feat_static_cat
            Static categorical features: (batch_size, num_cat_features)
        feat_static_real
            Static real-valued features: (batch_size, num_feat_static_real)
        past_target
            Past target: (batch_size, receptive_field)
        past_observed_values
            Observed value indicator for the past target: (batch_size,
            receptive_field)
        past_time_feat
            Past time features: (batch_size, num_time_features,
            receptive_field)
        future_time_feat
            Future time features: (batch_size, num_time_features, pred_length)
        scale
            Scale of the time series: (batch_size, 1)
        prediction_length
            Time length of the samples to generate. If not provided, use
            ``self.prediction_length``.
        num_parallel_samples
            Number of samples to generate. If not provided, use
            ``self.num_parallel_samples``.
        temperature
            Temperature to use in generating samples. If not provided, use
            ``self.temperature``.

        Returns
        -------
            Predictions with shape (batch_size, num_parallel_samples, pred_length)
        """
        if prediction_length is None:
            prediction_length = self.prediction_length
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples
        if temperature is None:
            temperature = self.temperature

        past_target = past_target.long()
        full_features = self.get_full_features(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_observed_values=None,
            scale=scale,
        )

        # To compute queues for the first step, we need features from
        # -self.pred_length - self.receptive_field + 1 to -self.pred_length + 1
        features_start_idx = -prediction_length - self.receptive_field + 1
        features_end_idx = (
            -prediction_length + 1 if prediction_length > 1 else None
        )
        queues = self._initialize_conv_queues(
            past_target=past_target[..., -self.receptive_field :],
            features=full_features[
                ...,
                features_start_idx:features_end_idx,
            ],
        )

        queues = [
            torch.repeat_interleave(q, num_parallel_samples, dim=0)
            for q in queues
        ]

        res = torch.repeat_interleave(
            past_target[..., -2:], num_parallel_samples, dim=0
        )

        for t in range(prediction_length):
            current_target = res[..., -2:]
            current_features = full_features[
                ...,
                self.receptive_field + t - 1 : self.receptive_field + t + 1,
            ]
            input_embedding = self.target_feature_embedding(
                current_target,
                torch.repeat_interleave(
                    current_features, num_parallel_samples, dim=0
                ),
            )
            logits, queues = self.base_net(input_embedding, queues=queues)

            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                y = torch.multinomial(probs.view(-1, self.n_bins), 1).view(
                    logits.size()[:-1]
                )
            else:
                y = torch.argmax(logits, dim=-1)
            y = y.long()
            res = torch.cat([res, y], dim=-1)

        samples = res[..., -prediction_length:]
        samples = samples.view(-1, num_parallel_samples, prediction_length)
        samples = self.lookup_values(samples)
        samples = samples * scale[:, None]
        return samples
