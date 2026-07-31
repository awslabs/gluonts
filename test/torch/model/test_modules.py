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
        pytest.param(
            None,
            4,
            (4, 100, 12),
            torch.float,
            id="mqf2",
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
    ],
)
def test_module_smoke(module, batch_size, expected_shapes, expected_dtypes):
    if module is None:
        from gluonts.torch.model.mqf2 import MQF2MultiHorizonModel

        module = MQF2MultiHorizonModel(
            freq="1H",
            context_length=24,
            prediction_length=12,
            num_feat_dynamic_real=1,
            num_feat_static_real=1,
            num_feat_static_cat=1,
            cardinality=[1],
        )
    batch = module.describe_inputs(batch_size).zeros()
    outputs = module(**batch)
    assert_shapes_and_dtypes(outputs, expected_shapes, expected_dtypes)


@pytest.mark.parametrize("num_feat_dynamic_real", [1, 3])
def test_PatchTSTModel_time_features_shape(num_feat_dynamic_real):
    """
    Regression test for #3167 (umbrella #3296 item 6).

    PR #3167 added ``num_feat_dynamic_real`` support to PatchTST,
    routing extra real-valued time features through ``past_time_feat``
    / ``future_time_feat`` of shape
    ``(batch, length, num_feat_dynamic_real)``. The existing
    ``test_module_smoke`` only exercises modules at the default
    ``num_feat_dynamic_real=0``, so the time-feature branch is not
    pinned anywhere in the test suite.

    This test pins both the ``describe_inputs()`` shape contract (so a
    future refactor of the ``Input`` spec doesn't silently change the
    layout) and a forward-pass shape sanity check that the new fields
    actually flow through the encoder.
    """
    from gluonts.torch.model.patch_tst import PatchTSTModel

    batch_size = 2
    context_length = 24
    prediction_length = 12

    module = PatchTSTModel(
        prediction_length=prediction_length,
        context_length=context_length,
        patch_len=16,
        stride=8,
        padding_patch="end",
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        activation="relu",
        norm_first=False,
        num_encoder_layers=1,
        scaling="mean",
        num_feat_dynamic_real=num_feat_dynamic_real,
    )

    spec = module.describe_inputs(batch_size)

    # Shape contract: time-feature inputs are present and have the
    # expected layout. Order matters for downstream code that flattens
    # the spec into positional kwargs.
    assert "past_time_feat" in spec
    assert "future_time_feat" in spec
    assert spec.shapes["past_time_feat"] == (
        batch_size,
        context_length,
        num_feat_dynamic_real,
    )
    assert spec.shapes["future_time_feat"] == (
        batch_size,
        prediction_length,
        num_feat_dynamic_real,
    )

    # Forward-pass sanity: feeding the spec'd zeros must produce
    # distribution args matching prediction_length, with no shape
    # mismatch when concatenating time-feature patches into the encoder
    # input. distr_args is a tuple of tensors (per StudentTOutput).
    batch = module.describe_inputs(batch_size).zeros()
    distr_args, loc, scale = module(**batch)
    assert loc.shape == (batch_size, 1)
    assert scale.shape == (batch_size, 1)
    for arg in distr_args:
        assert arg.shape[:2] == (batch_size, prediction_length)
