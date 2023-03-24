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

from gluonts.torch.model.deepar import DeepARModel
from gluonts.torch.model.mqf2 import MQF2MultiHorizonModel
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardModel
from gluonts.torch.model.tft import TemporalFusionTransformerModel


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
                quantiles=[0.2, 0.25, 0.5, 0.9, 0.95],
                d_past_feat_dynamic_real=[1],
                d_feat_dynamic_real=[2, 5],
                d_feat_static_real=[3, 1, 1],
                c_past_feat_dynamic_cat=[2, 2, 2],
                c_feat_dynamic_cat=[2],
                c_feat_static_cat=[2, 2],
            ),
            4,
            (4, 5, 12),  # (batch_size, len(quantiles), prediction_length)
            torch.float,
        ),
    ],
)
def test_module_smoke(module, batch_size, expected_shapes, expected_dtypes):
    batch = module.describe_inputs(batch_size).zeros()
    outputs = module(**batch)
    assert_shapes_and_dtypes(outputs, expected_shapes, expected_dtypes)
