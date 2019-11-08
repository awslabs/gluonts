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

import math

from mxnet.gluon import HybridBlock
from mxnet.gluon import nn

# First-party imports
from gluonts.block.feature import FeatureEmbedder
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class DeepFactorNetworkBase(HybridBlock):
    def __init__(
        self,
        global_model: HybridBlock,
        local_model: HybridBlock,
        embedder: FeatureEmbedder,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.global_model = global_model
        self.local_model = local_model
        self.embedder = embedder
        with self.name_scope():
            self.loading = nn.Dense(
                units=global_model.num_output, use_bias=False
            )

    def assemble_features(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, 1)
        time_feat: Tensor,  # (batch_size, history_length, num_features)
    ) -> Tensor:  # (batch_size, history_length, num_features)
        # todo: this is shared by more than one places, and should be a general routine

        embedded_cat = self.embedder(
            feat_static_cat
        )  # (batch_size, num_features * embedding_size)

        # a workaround when you wish to repeat without knowing the number
        # of repeats
        helper_ones = F.ones_like(
            F.slice_axis(time_feat, axis=2, begin=-1, end=None)
        )
        # (batch_size, history_length, num_features * embedding_size)
        repeated_cat = F.batch_dot(
            helper_ones, F.expand_dims(embedded_cat, axis=1)
        )

        # putting together all the features
        input_feat = F.concat(repeated_cat, time_feat, dim=2)
        return embedded_cat, input_feat

    def compute_global_local(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, 1)
        time_feat: Tensor,  # (batch_size, history_length, num_features)
    ) -> (Tensor, Tensor):  # both of size (batch_size, history_length, 1)

        cat, local_input = self.assemble_features(
            F, feat_static_cat, time_feat
        )
        loadings = self.loading(cat)  # (batch_size, num_factors)
        global_factors = self.global_model(
            time_feat
        )  # (batch_size, history_length, num_factors)

        fixed_effect = F.batch_dot(
            global_factors, loadings.expand_dims(axis=2)
        )  # (batch_size, history_length, 1)
        random_effect = F.log(
            F.exp(self.local_model(local_input)) + 1.0
        )  # (batch_size, history_length, 1)
        return F.exp(fixed_effect), random_effect

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError

    def negative_normal_likelihood(self, F, y, mu, sigma):
        return (
            F.log(sigma)
            + 0.5 * math.log(2 * math.pi)
            + 0.5 * F.square((y - mu) / sigma)
        )


class DeepFactorTrainingNetwork(DeepFactorNetworkBase):
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, 1)
        past_time_feat: Tensor,
        # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length)
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space
        feat_static_cat
            Shape: (batch_size, 1)
        past_time_feat
            Shape: (batch_size, history_length, num_features)
        past_target
            Shape: (batch_size, history_length)

        Returns
        -------
        Tensor
            A batch of negative log likelihoods.
        """

        fixed_effect, random_effect = self.compute_global_local(
            F, feat_static_cat, past_time_feat
        )

        loss = self.negative_normal_likelihood(
            F, past_target.expand_dims(axis=2), fixed_effect, random_effect
        )
        return loss


class DeepFactorPredictionNetwork(DeepFactorNetworkBase):
    @validated()
    def __init__(
        self, prediction_len: int, num_parallel_samples: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_len = prediction_len
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        past_target: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space
        feat_static_cat
            Shape: (batch_size, 1)
        past_time_feat
            Shape: (batch_size, history_length, num_features)
        future_time_feat
            Shape: (batch_size, prediction_length, num_features)
        past_target
            Shape: (batch_size, history_length)

        Returns
        -------
        Tensor
            Samples of shape (batch_size, num_samples, prediction_length).
        """
        time_feat = F.concat(past_time_feat, future_time_feat, dim=1)
        fixed_effect, random_effect = self.compute_global_local(
            F, feat_static_cat, time_feat
        )

        samples = F.concat(
            *[
                F.sample_normal(fixed_effect, random_effect)
                for _ in range(self.num_parallel_samples)
            ],
            dim=2,
        )  # (batch_size, train_len + prediction_len, num_samples)
        pred_samples = F.slice_axis(
            samples, axis=1, begin=-self.prediction_len, end=None
        )  # (batch_size, prediction_len, num_samples)

        return pred_samples.swapaxes(1, 2)
