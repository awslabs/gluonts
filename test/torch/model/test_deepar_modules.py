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

    lightning_module = DeepARLightningModule(
        model_kwargs={
            "freq": "1H",
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_feat_dynamic_real": num_feat_dynamic_real,
            "num_feat_static_real": num_feat_static_real,
            "num_feat_static_cat": num_feat_static_cat,
            "cardinality": cardinality,
            "scaling": scaling,
        }
    )
    model = lightning_module.model

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

    log_densities = model.log_prob(
        feat_static_cat,
        feat_static_real,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
        future_target,
    )

    assert log_densities.shape == (batch_size,)

    log_densities = model.log_prob(
        feat_static_cat,
        feat_static_real,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
        samples.transpose(0, 1),  # TODO: ugly!
    )

    assert log_densities.shape == (100, batch_size)

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

    assert lightning_module.training_step(batch, batch_idx=0).shape == ()
    assert lightning_module.validation_step(batch, batch_idx=0).shape == ()


@pytest.mark.parametrize(
    "prediction_length, context_length, lags_seq",
    [
        (1, 5, [1, 10, 15]),
        (1, 5, [3, 6, 9, 10]),
        (2, 5, [1, 2, 7]),
        (2, 5, [2, 3, 4, 6]),
        (4, 5, [1, 5, 9, 11]),
        (4, 5, [7, 8, 13, 14]),
    ],
)
def test_rnn_input(
    prediction_length: int, context_length: int, lags_seq: List[int]
):
    num_samples = 10
    history_length = max(lags_seq) + context_length - 1

    model = DeepARModel(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length,
        lags_seq=lags_seq,
        scaling=False,
        num_parallel_samples=num_samples,
    )

    # Construct test batch of size 1, such that:
    # - target values are increasing integers
    # - there is one dynamic feature that is equal to the target
    #
    # Since scaling=False, this way we can compare lagged target
    # values and dynamic feature and verify that the expected values
    # are given as input to the RNN, at the expected time index.

    batch = {
        "feat_static_cat": torch.tensor([[0]], dtype=torch.int64),
        "feat_static_real": torch.tensor([[0.0]], dtype=torch.float32),
        "past_time_feat": torch.arange(
            history_length, dtype=torch.float32
        ).view(1, history_length, 1),
        "past_target": torch.arange(history_length, dtype=torch.float32).view(
            1, history_length
        ),
        "past_observed_values": torch.arange(
            history_length, dtype=torch.float32
        ).view(1, history_length),
    }

    # test with no future_target (only one step prediction)

    batch["future_time_feat"] = torch.arange(
        history_length,
        history_length + 1,
        dtype=torch.float32,
    ).view(1, 1, 1)

    rnn_input, scale, _ = model.prepare_rnn_input(**batch)

    assert (scale == 1.0).all()

    ref = torch.arange(
        history_length - context_length + 1,
        history_length + 1,
        dtype=torch.float32,
    )

    for idx, lag in enumerate(lags_seq):
        assert torch.equal(ref - lag, rnn_input[0, :, idx])

    # test with all future data

    batch["future_time_feat"] = torch.arange(
        history_length,
        history_length + prediction_length,
        dtype=torch.float32,
    ).view(1, prediction_length, 1)

    batch["future_target"] = torch.arange(
        history_length,
        history_length + prediction_length,
        dtype=torch.float32,
    ).view(1, prediction_length)

    rnn_input, scale, _ = model.prepare_rnn_input(**batch)

    assert (scale == 1.0).all()

    ref = torch.arange(
        history_length - context_length + 1,
        history_length + prediction_length,
        dtype=torch.float32,
    )

    for idx, lag in enumerate(lags_seq):
        assert torch.equal(ref - lag, rnn_input[0, :, idx])
