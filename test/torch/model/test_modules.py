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

import pytest
import torch

from gluonts.torch.distributions import QuantileOutput
from gluonts.torch.model.deepar import DeepARModel
from gluonts.torch.model.mqf2 import MQF2MultiHorizonModel
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardModel
from gluonts.torch.model.tft import TemporalFusionTransformerModel
from gluonts.torch.model.mq_cnn import MQCNNModel


def assert_shapes_and_dtypes(tensors, shapes, dtypes):
    if isinstance(tensors, torch.Tensor):
        assert tensors.shape == shapes
        assert tensors.dtype == dtypes
    else:
        for tensor, shape, dtype in zip(tensors, shapes, dtypes):
            assert_shapes_and_dtypes(tensor, shape, dtype)


@pytest.mark.parametrize(
    "module, batch_size, expected_shapes, expected_dtypes",
    [
        (
            DeepARModel(
                freq="1H",
                context_length=24,
                prediction_length=12,
                num_feat_dynamic_real=1,
                num_feat_static_real=1,
                num_feat_static_cat=1,
                cardinality=[1],
            ),
            4,
            (4, 100, 12),
            torch.float,
        ),
        (
            MQF2MultiHorizonModel(
                freq="1H",
                context_length=24,
                prediction_length=12,
                num_feat_dynamic_real=1,
                num_feat_static_real=1,
                num_feat_static_cat=1,
                cardinality=[1],
            ),
            4,
            (4, 100, 12),
            torch.float,
        ),
        (
            SimpleFeedForwardModel(
                context_length=24,
                prediction_length=12,
            ),
            4,
            [[(4, 12), (4, 12), (4, 12)], (4, 1), (4, 1)],
            [
                [torch.float, torch.float, torch.float],
                torch.float,
                torch.float,
            ],
        ),
        (
            TemporalFusionTransformerModel(
                context_length=24,
                prediction_length=12,
                distr_output=QuantileOutput([0.2, 0.25, 0.5, 0.9, 0.95]),
                d_past_feat_dynamic_real=[1],
                d_feat_dynamic_real=[2, 5],
                d_feat_static_real=[3, 1, 1],
                c_past_feat_dynamic_cat=[2, 2, 2],
                c_feat_dynamic_cat=[2],
                c_feat_static_cat=[2, 2],
            ),
            4,
            [[(4, 12, 5)], (4, 1), (4, 1)],
            [[torch.float], torch.float, torch.float],
        ),
        (
            MQCNNModel(
                context_length=24,
                prediction_length=12,
                num_forking=8,
                distr_output=QuantileOutput([0.2, 0.25, 0.5, 0.9, 0.95]),
                past_feat_dynamic_real_dim=4,
                feat_dynamic_real_dim=2,
                feat_static_real_dim=2,
                cardinality_dynamic=[2],
                cardinality_static=[2, 2],
                scaling=False,
                scaling_decoder_dynamic_feature=False,
                embedding_dimension_dynamic=[2],
                embedding_dimension_static=[2, 2],
                encoder_cnn_init_dim=8,
                dilation_seq=[1, 3, 9],
                kernel_size_seq=[7, 3, 3],
                channels_seq=[30, 30, 30],
                joint_embedding_dimension=30,
                encoder_mlp_init_dim=7,
                encoder_mlp_dim_seq=[30],
                use_residual=True,
                decoder_mlp_dim_seq=[30],
                decoder_hidden_dim=60,
                decoder_future_init_dim=4,
                decoder_future_embedding_dim=50,
            ),
            4,
            [[(4, 8, 12, 5)], (4, 1), (4, 1)],
            [[torch.float], torch.float, torch.float],
        ),
    ],
)
def test_module_smoke(module, batch_size, expected_shapes, expected_dtypes):
    batch = module.describe_inputs(batch_size).zeros()
    outputs = module(**batch)
    assert_shapes_and_dtypes(outputs, expected_shapes, expected_dtypes)
