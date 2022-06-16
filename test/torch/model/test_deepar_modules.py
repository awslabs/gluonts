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

import pytest
import torch

from gluonts.torch.model.deepar import DeepARLightningModule, DeepARModel
from gluonts.torch.model.deepar.util import lagged_sequence_values


@pytest.mark.parametrize(
    "lag_indices, prior_sequence, sequence",
    [
        (
            [0, 1, 5, 10, 20],
            torch.randn((4, 100)),
            torch.randn((4, 8)),
        ),
        (
            [0, 1, 5, 10, 20],
            torch.randn((4, 100, 1)),
            torch.randn((4, 8, 1)),
        ),
        (
            [0, 1, 5, 10, 20],
            torch.ones((4, 100, 2)),
            torch.ones((4, 8, 2)),
        ),
    ],
)
def test_lagged_sequence_values(
    lag_indices: List[int],
    prior_sequence: torch.Tensor,
    sequence: torch.Tensor,
):
    res = lagged_sequence_values(lag_indices, prior_sequence, sequence, None)
    full_sequence = torch.cat((prior_sequence, sequence), dim=1)
    for t in range(res.shape[1]):
        expected_lags_t = torch.stack(
            [
                full_sequence[:, t + prior_sequence.shape[1] - l]
                for l in lag_indices
            ],
            dim=-1,
        ).reshape(sequence.shape[0], -1)
        assert torch.allclose(expected_lags_t, res[:, t, :])


@pytest.mark.parametrize(
    "num_feat_dynamic_real, num_feat_static_real, num_feat_static_cat, cardinality, scaling",
    [
        (5, 4, 1, [1], True),
        (1, 4, 2, [2, 3], False),
        (5, 1, 3, [4, 5, 6], True),
        (5, 4, 1, [1], False),
    ],
)
def test_deepar_modules(
    num_feat_dynamic_real: int,
    num_feat_static_real: int,
    num_feat_static_cat: int,
    cardinality: Optional[List[int]],
    scaling: bool,
):
    batch_size = 4
    prediction_length = 6
    context_length = 12

    model = DeepARModel(
        freq="1H",
        context_length=context_length,
        prediction_length=prediction_length,
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_feat_static_real=num_feat_static_real,
        num_feat_static_cat=num_feat_static_cat,
        cardinality=cardinality,
        scaling=scaling,
    )

    # TODO uncomment the following
    # torch.jit.script(model)

    feat_static_cat = torch.zeros(
        batch_size, num_feat_static_cat, dtype=torch.long
    )
    feat_static_real = torch.ones(batch_size, num_feat_static_real)
    past_time_feat = torch.ones(
        batch_size, model._past_length, num_feat_dynamic_real
    )
    future_time_feat = torch.ones(
        batch_size, prediction_length, num_feat_dynamic_real
    )
    past_target = torch.ones(batch_size, model._past_length)
    past_observed_values = torch.ones(batch_size, model._past_length)
    future_target = torch.ones(batch_size, prediction_length)
    future_observed_values = torch.ones(batch_size, prediction_length)

    params, scale, _, _, _ = model.unroll_lagged_rnn(
        feat_static_cat,
        feat_static_real,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
        future_target,
    )

    assert scale.shape == (batch_size, 1)
    for p in params:
        assert p.shape == (batch_size, context_length + prediction_length - 1)

    distr = model.output_distribution(params, scale)

    assert distr.batch_shape == (
        batch_size,
        context_length + prediction_length - 1,
    )
    assert distr.event_shape == ()

    sliced_distr = model.output_distribution(params, scale, trailing_n=2)

    assert sliced_distr.batch_shape == (batch_size, 2)
    assert sliced_distr.event_shape == ()

    samples = model(
        feat_static_cat,
        feat_static_real,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
    )

    assert samples.shape == (batch_size, 100, prediction_length)

    batch = dict(
        feat_static_cat=feat_static_cat,
        feat_static_real=feat_static_real,
        past_time_feat=past_time_feat,
        future_time_feat=future_time_feat,
        past_target=past_target,
        past_observed_values=past_observed_values,
        future_target=future_target,
        future_observed_values=future_observed_values,
    )

    lightning_module = DeepARLightningModule(model=model)

    assert lightning_module.training_step(batch, batch_idx=0).shape == ()
    assert lightning_module.validation_step(batch, batch_idx=0).shape == ()
