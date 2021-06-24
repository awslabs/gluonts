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
import pytest

from gluonts.torch.model.deepar import (
    DeepARModel,
    DeepARLightningModule,
)
from gluonts.torch.model.deepar.module import LaggedLSTM


@pytest.mark.parametrize(
    "model, prior_input, input, features, more_input, more_features",
    [
        (
            LaggedLSTM(
                input_size=1, features_size=3, lags_seq=[0, 1, 5, 10, 20]
            ),
            torch.ones((4, 100)),
            torch.ones((4, 8)),
            torch.ones((4, 8, 3)),
            torch.ones((4, 5)),
            torch.ones((4, 5, 3)),
        ),
        (
            LaggedLSTM(
                input_size=1, features_size=3, lags_seq=[0, 1, 5, 10, 20]
            ),
            torch.ones((4, 100, 1)),
            torch.ones((4, 8, 1)),
            torch.ones((4, 8, 3)),
            torch.ones((4, 5, 1)),
            torch.ones((4, 5, 3)),
        ),
        (
            LaggedLSTM(
                input_size=2, features_size=3, lags_seq=[0, 1, 5, 10, 20]
            ),
            torch.ones((4, 100, 2)),
            torch.ones((4, 8, 2)),
            torch.ones((4, 8, 3)),
            torch.ones((4, 5, 2)),
            torch.ones((4, 5, 3)),
        ),
    ],
)
def test_lagged_lstm(
    model: LaggedLSTM,
    prior_input: torch.Tensor,
    input: torch.Tensor,
    features: torch.Tensor,
    more_input: torch.Tensor,
    more_features: torch.Tensor,
):
    torch.jit.script(model)
    output, state = model(prior_input, input, features=features)
    assert output.shape[:2] == input.shape[:2]
    more_output, state = model(
        torch.cat((prior_input, input), dim=1),
        more_input,
        features=more_features,
        state=state,
    )
    assert more_output.shape[:2] == more_input.shape[:2]


@pytest.mark.parametrize(
    "num_feat_dynamic_real, num_feat_static_real, num_feat_static_cat, cardinality",
    [
        (5, 4, 1, [1]),
        (1, 4, 2, [2, 3]),
        (5, 1, 3, [4, 5, 6]),
        (5, 4, 1, [1]),
    ],
)
def test_deepar_modules(
    num_feat_dynamic_real: int,
    num_feat_static_real: int,
    num_feat_static_cat: int,
    cardinality: Optional[List[int]],
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

    params, scale, _, _ = model.unroll_lagged_rnn(
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
