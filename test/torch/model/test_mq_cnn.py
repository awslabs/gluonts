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

from typing import List

import torch
import pytest

from gluonts.torch.distributions import QuantileOutput
from gluonts.torch.model.mq_cnn import MQCNNLightningModule


@pytest.mark.parametrize(
    "past_feat_dynamic_real_dim, feat_dynamic_real_dim, cardinality_dynamic, feat_static_real_dim, cardinality_static, quantiles",
    [
        (3, 2, [5, 5, 5], 3, [5, 2], [0.1, 0.5, 0.9]),
        (2, 0, [4, 2], 2, [4, 2, 2], [0.05, 0.25]),
    ],
)
def test_mq_cnn_modules(
    past_feat_dynamic_real_dim: int,
    feat_dynamic_real_dim: int,
    cardinality_dynamic: List[int],
    feat_static_real_dim: int,
    cardinality_static: List[int],
    quantiles: List[float],
):
    batch_size = 4
    prediction_length = 6
    context_length = 12
    num_forking = 8

    enc_cnn_init_dim = (
        2 + past_feat_dynamic_real_dim + sum(cardinality_dynamic)
    )  # with target, observed
    enc_mlp_init_dim = (
        1 + feat_static_real_dim + sum(cardinality_static)
    )  # with scaler
    dec_future_init_dim = feat_dynamic_real_dim + sum(cardinality_dynamic)
    joint_embedding_dim = 30
    encoder_mlp_dim_seq = [30]
    decoder_mlp_dim_seq = [30]
    decoder_hidden_dim = 64
    decoder_future_embedding_dim = 50
    channels_seq = [30, 30, 30]
    dilation_seq = [1, 3, 9]
    kernel_size_seq = [7, 3, 3]

    lightning_module = MQCNNLightningModule(
        {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_forking": num_forking,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "feat_dynamic_real_dim": feat_dynamic_real_dim,
            "cardinality_dynamic": cardinality_dynamic,
            "embedding_dimension_dynamic": cardinality_dynamic,
            "feat_static_real_dim": feat_static_real_dim,
            "cardinality_static": cardinality_static,
            "embedding_dimension_static": cardinality_static,
            "scaling": False,
            "scaling_decoder_dynamic_feature": False,
            "encoder_cnn_init_dim": enc_cnn_init_dim,
            "dilation_seq": dilation_seq,
            "kernel_size_seq": kernel_size_seq,
            "channels_seq": channels_seq,
            "joint_embedding_dimension": joint_embedding_dim,
            "encoder_mlp_init_dim": enc_mlp_init_dim,
            "encoder_mlp_dim_seq": encoder_mlp_dim_seq,
            "use_residual": True,
            "decoder_mlp_dim_seq": decoder_mlp_dim_seq,
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_future_init_dim": dec_future_init_dim,
            "decoder_future_embedding_dim": decoder_future_embedding_dim,
            "distr_output": QuantileOutput(quantiles),
        }
    )
    model = lightning_module.model

    feat_static_cat = torch.zeros(
        batch_size, len(cardinality_static), dtype=torch.long
    )
    feat_static_real = torch.ones(batch_size, feat_static_real_dim)
    future_feat_dynamic_cat = torch.zeros(
        batch_size,
        num_forking,
        prediction_length,
        len(cardinality_dynamic),
        dtype=torch.long,
    )
    past_feat_dynamic_cat = torch.zeros(
        batch_size,
        context_length,
        len(cardinality_dynamic),
        dtype=torch.long,
    )
    future_feat_dynamic = torch.ones(
        batch_size,
        num_forking,
        prediction_length,
        feat_dynamic_real_dim,
    )
    past_feat_dynamic = torch.ones(
        batch_size, context_length, past_feat_dynamic_real_dim
    )
    past_target = torch.ones(batch_size, context_length, 1)
    past_observed_values = torch.ones(batch_size, context_length, 1)
    future_target = torch.ones(batch_size, num_forking, prediction_length)
    future_observed_values = torch.ones(
        batch_size, num_forking, prediction_length
    )
    output, loc, scale = model(
        past_target=past_target,
        past_feat_dynamic=past_feat_dynamic,
        future_feat_dynamic=future_feat_dynamic,
        feat_static_real=feat_static_real,
        feat_static_cat=feat_static_cat,
        past_observed_values=past_observed_values,
        past_feat_dynamic_cat=past_feat_dynamic_cat,
        future_feat_dynamic_cat=future_feat_dynamic_cat,
    )

    assert output[0].shape == (
        batch_size,
        num_forking,
        prediction_length,
        len(quantiles),
    )
    assert loc.shape == scale.shape == (batch_size, 1)

    batch = dict(
        past_target=past_target,
        future_target=future_target,
        past_feat_dynamic=past_feat_dynamic,
        future_feat_dynamic=future_feat_dynamic,
        feat_static_real=feat_static_real,
        feat_static_cat=feat_static_cat,
        past_observed_values=past_observed_values,
        future_observed_values=future_observed_values,
        past_feat_dynamic_cat=past_feat_dynamic_cat,
        future_feat_dynamic_cat=future_feat_dynamic_cat,
    )

    assert lightning_module.training_step(batch, batch_idx=0).shape == ()
    assert lightning_module.validation_step(batch, batch_idx=0).shape == ()
