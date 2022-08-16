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

from functools import partial
from typing import Optional, Callable, List, Union

import torch
from torch import nn
from torch.distributions import (
    Categorical,
    MixtureSameFamily,
    Normal,
)
from gluonts.core.component import validated

from gluonts.torch.distributions.discrete_distribution import (
    DiscreteDistribution,
)
from .scaling import (
    min_max_scaling,
    standard_normal_scaling,
)

INPUT_SCALING_MAP = {
    "min_max_scaling": partial(min_max_scaling, dim=1, keepdim=True),
    "standard_normal_scaling": partial(
        standard_normal_scaling, dim=1, keepdim=True
    ),
}


def init_weights(module: nn.Module, scale: float = 1.0):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, -scale, scale)
        nn.init.zeros_(module.bias)


class FeatureEmbedder(nn.Module):
    """Creates a feature embedding for the static categorical features."""

    @validated()
    def __init__(
        self,
        cardinalities: List[int],
        embedding_dimensions: List[int],
    ):
        super().__init__()

        assert (
            len(cardinalities) > 0
        ), "Length of `cardinalities` list must be greater than zero"
        assert len(cardinalities) == len(
            embedding_dimensions
        ), "Length of `embedding_dims` and `embedding_dims` should match"
        assert all(
            [c > 0 for c in cardinalities]
        ), "Elements of `cardinalities` should be > 0"
        assert all(
            [d > 0 for d in embedding_dimensions]
        ), "Elements of `embedding_dims` should be > 0"

        self.embedders = [
            torch.nn.Embedding(num_embeddings=card, embedding_dim=dim)
            for card, dim in zip(cardinalities, embedding_dimensions)
        ]
        for embedder in self.embedders:
            embedder.apply(init_weights)

    def forward(self, features: torch.Tensor):
        """
        Parameters
        ----------
        features
            Input features to the model, shape: (-1, num_features).

        Returns
        -------
        torch.Tensor
            Embedding, shape: (-1, sum(self.embedding_dimensions)).
        """
        embedded_features = torch.cat(
            [
                embedder(features[:, i].long())
                for i, embedder in enumerate(self.embedders)
            ],
            dim=-1,
        )
        return embedded_features


class DeepNPTSNetwork(nn.Module):
    """Base class implementing a simple feed-forward neural network that takes
    in static and dynamic features and produces `num_hidden_nodes` independent
    outptus. These outputs are then used by derived classes to construct the
    forecast distribution for a single time step.

    Note that the dynamic features are just treated as independent features
    without considering their temporal nature.
    """

    @validated()
    def __init__(
        self,
        context_length: int,
        num_hidden_nodes: List[int],
        cardinality: List[int],
        embedding_dimension: List[int],
        num_time_features: int,
        batch_norm: bool = False,
        input_scaling: Optional[Union[Callable, str]] = None,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()

        self.context_length = context_length
        self.num_hidden_nodes = num_hidden_nodes
        self.batch_norm = batch_norm
        self.input_scaling = (
            INPUT_SCALING_MAP[input_scaling]
            if isinstance(input_scaling, str)
            else input_scaling
        )
        self.dropout_rate = dropout_rate

        # Embedding for categorical features
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality, embedding_dimensions=embedding_dimension
        )
        total_embedding_dim = sum(embedding_dimension)

        # We have two target related features: past_target and observed value
        # indicator each of length `context_length`.
        # Also, +1 for the static real feature.
        dimensions = [
            context_length * (num_time_features + 2) + total_embedding_dim + 1
        ] + num_hidden_nodes
        modules: List[nn.Module] = []
        for in_features, out_features in zip(dimensions[:-1], dimensions[1:]):
            modules += [nn.Linear(in_features, out_features), nn.ReLU()]
            if self.batch_norm:
                modules.append(nn.BatchNorm1d(out_features))
            if self.dropout_rate:
                modules.append(nn.Dropout(self.dropout_rate))

        self.model = nn.Sequential(*modules)
        self.model.apply(partial(init_weights, scale=0.07))

    # TODO: Handle missing values using the observed value indicator.
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
    ):
        """
        Parameters
        ----------
        feat_static_cat
            Shape (-1, num_features).
        feat_static_real
            Shape (-1, num_features).
        past_target
            Shape (-1, context_length).
        past_observed_values
            Shape (-1, context_length).
        past_time_feat
            Shape (-1, context_length, self.num_time_features).
        """
        x = past_target
        if self.input_scaling:
            loc, scale = self.input_scaling(x)
            x_scaled = (x - loc) / scale
        else:
            x_scaled = x

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, torch.tensor(feat_static_real)),
            dim=1,
        )
        time_features = torch.cat(
            [
                x_scaled.unsqueeze(dim=-1),
                past_observed_values.unsqueeze(dim=-1),
                past_time_feat,
            ],
            dim=-1,
        )

        features = torch.cat(
            [
                time_features.reshape(time_features.shape[0], -1),
                static_feat,
            ],
            dim=-1,
        )
        return self.model(features)


class DeepNPTSNetworkDiscrete(DeepNPTSNetwork):
    """Extends `DeepNTPSNetwork` by implementing the output layer which
    converts the ouptuts from the base network into probabilities of length
    `context_length`. These probabilities together with the past values in the
    context window constitute the one-step-ahead forecast distribution.
    Specifically, the forecast is always one of the values observed in the
    context window with the corresponding predicted probability.

    Parameters ---------- *args     Arguments to ``DeepNPTSNetwork``.
    use_softmax     Flag indicating whether to use softmax or normalization for
    converting the outputs of the base network     to probabilities. kwargs
    Keyword arguments to ``DeepNPTSNetwork``.
    """

    @validated()
    def __init__(self, *args, use_softmax: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_softmax = use_softmax
        modules: List[nn.Module] = (
            []
            if self.dropout_rate is None
            else [nn.Dropout(self.dropout_rate)]
        )
        modules.append(
            nn.Linear(self.num_hidden_nodes[-1], self.context_length)
        )
        self.output_layer = nn.Sequential(*modules)
        self.output_layer.apply(init_weights)

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
    ) -> DiscreteDistribution:
        h = super().forward(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
        )
        outputs = self.output_layer(h)
        probs = (
            nn.functional.softmax(outputs, dim=1)
            if self.use_softmax
            else nn.functional.normalize(
                nn.functional.softplus(outputs), p=1, dim=1
            )
        )
        return DiscreteDistribution(values=past_target, probs=probs)


class DeepNPTSNetworkSmooth(DeepNPTSNetwork):
    """Extends `DeepNTPSNetwork` by implementing the output layer which
    converts the ouptuts from the base network into a smoothed mixture
    distribution. The components of the mixture are Gaussians centered around
    the observations in the context window. The mixing probabilities as well as
    the width of the Gaussians are predicted by the network.

    This mixture distribution represents the one-step-ahead forecast
    distribution. Note that the forecast can contain values not observed in the
    context window.
    """

    @validated()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        modules = (
            []
            if self.dropout_rate is None
            else [nn.Dropout(self.dropout_rate)]
        )
        modules += [
            nn.Linear(self.num_hidden_nodes[-1], self.context_length + 1),
            nn.Softplus(),
        ]
        self.output_layer = nn.Sequential(*modules)
        self.output_layer.apply(init_weights)

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
    ) -> MixtureSameFamily:
        h = super().forward(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
        )
        outputs = self.output_layer(h)
        probs = outputs[:, :-1]
        kernel_width = outputs[:, -1:]
        mix = Categorical(probs)
        components = Normal(loc=past_target, scale=kernel_width)
        return MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )


class DeepNPTSMultiStepPredictor(nn.Module):
    """Implements multi-step prediction given a trained `DeepNPTSNewtork` model
    that outputs one-step-ahead forecast distribution."""

    @validated()
    def __init__(
        self,
        net: DeepNPTSNetwork,
        prediction_length: int,
        num_parallel_samples: int = 100,
    ):
        super().__init__()
        self.net = net
        self.prediction_length = prediction_length
        self.num_parallel_samples = num_parallel_samples

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
    ):
        """Generates samples from the forecast distribution.

        Parameters ---------- feat_static_cat     Shape (-1, num_features).
        feat_static_real     Shape (-1, num_features). past_target     Shape
        (-1, context_length). past_observed_values     Shape (-1,
        context_length). past_time_feat     Shape (-1, context_length,
        self.num_time_features). future_time_feat     Shape (-1,
        prediction_length, self.num_time_features).  Returns -------
        torch.Tensor     Tensor containing samples from the predicted
        distribution.     Shape is (-1, self.num_parallel_samples,
        self.prediction_length).
        """
        # Blow up the initial `x` by the number of parallel samples required.
        # (batch_size * num_parallel_samples, context_length)
        past_target = past_target.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        # Note that gluonts returns empty future_observed_values.
        future_observed_values = torch.ones(
            (past_observed_values.shape[0], self.prediction_length)
        )
        observed_values = torch.cat(
            [past_observed_values, future_observed_values], dim=1
        )
        observed_values = observed_values.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        time_feat = torch.cat([past_time_feat, future_time_feat], dim=1)
        time_feat = time_feat.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        feat_static_cat = feat_static_cat.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        feat_static_real = feat_static_real.repeat_interleave(
            self.num_parallel_samples, dim=0
        )

        future_samples = []
        for t in range(self.prediction_length):
            distr = self.net(
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_target=past_target,
                past_observed_values=observed_values[
                    :, t : -self.prediction_length + t
                ],
                past_time_feat=time_feat[
                    :, t : -self.prediction_length + t, :
                ],
            )
            samples = distr.sample()
            if past_target.dim() != samples.dim():
                samples = samples.unsqueeze(dim=-1)

            future_samples.append(samples)
            past_target = torch.cat([past_target[:, 1:], samples], dim=1)

        # (batch_size * num_parallel_samples, prediction_length)
        samples_out = torch.stack(future_samples, dim=1)

        # (batch_size, num_parallel_samples, prediction_length)
        return samples_out.reshape(
            -1, self.num_parallel_samples, self.prediction_length
        )
