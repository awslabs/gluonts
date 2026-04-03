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

from gluonts.torch.distributions import QuantileOutput, StudentTOutput
from gluonts.torch.model.crossformer import CrossformerLightningModule


@pytest.mark.parametrize(
    "num_feat_dynamic_real, distr_output, expected_output_shapes",
    [
        (
            0,
            StudentTOutput(),
            [(4, 6, 3), (4, 6, 3), (4, 6, 3)],
        ),
        (
            2,
            QuantileOutput([0.1, 0.5, 0.9]),
            [(4, 6, 3, 3)],
        ),
    ],
)
def test_crossformer_modules(
    num_feat_dynamic_real,
    distr_output,
    expected_output_shapes,
):
    batch_size = 4
    context_length = 12
    prediction_length = 6
    target_dim = 3

    lightning_module = CrossformerLightningModule(
        {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "seg_len": 3,
            "d_model": 12,
            "d_ff": 24,
            "n_heads": 3,
            "num_encoder_layers": 2,
            "num_feat_dynamic_real": num_feat_dynamic_real,
            "distr_output": distr_output,
        }
    )
    model = lightning_module.model

    batch = {
        "past_target": torch.ones(batch_size, context_length, target_dim),
        "past_observed_values": torch.ones(
            batch_size, context_length, target_dim
        ),
        "future_target": torch.ones(
            batch_size, prediction_length, target_dim
        ),
        "future_observed_values": torch.ones(
            batch_size, prediction_length, target_dim
        ),
    }
    if num_feat_dynamic_real > 0:
        batch["past_time_feat"] = torch.ones(
            batch_size, context_length, num_feat_dynamic_real
        )
        batch["future_time_feat"] = torch.ones(
            batch_size, prediction_length, num_feat_dynamic_real
        )

    distr_args, loc, scale = model(
        past_target=batch["past_target"],
        past_observed_values=batch["past_observed_values"],
        past_time_feat=batch.get("past_time_feat"),
        future_time_feat=batch.get("future_time_feat"),
    )

    assert [arg.shape for arg in distr_args] == expected_output_shapes
    assert loc.shape == scale.shape == (batch_size, 1, target_dim)
    pred_output = lightning_module(
        past_target=batch["past_target"],
        past_observed_values=batch["past_observed_values"],
        past_time_feat=batch.get("past_time_feat"),
        future_time_feat=batch.get("future_time_feat"),
    )
    if hasattr(distr_output, "distribution"):
        assert pred_output.shape == (
            batch_size,
            100,
            prediction_length,
            target_dim,
        )
    else:
        pred_distr_args, pred_loc, pred_scale = pred_output
        assert [arg.shape for arg in pred_distr_args] == expected_output_shapes
        assert pred_loc.shape == pred_scale.shape == (
            batch_size,
            1,
            target_dim,
        )

    assert lightning_module.training_step(batch, batch_idx=0).shape == ()
    assert lightning_module.validation_step(batch, batch_idx=0).shape == ()
