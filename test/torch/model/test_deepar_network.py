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

from gluonts.torch.model.deepar import DeepARNetwork, DeepARLightningNetwork


@pytest.mark.parametrize(
    "num_feat_dynamic_real, num_feat_static_real, num_feat_static_cat, cardinality",
    [
        (5, 4, 1, [1]),
        (1, 4, 2, [2, 3]),
        (5, 1, 3, [4, 5, 6]),
        (5, 4, 1, [1]),
    ],
)
def test_deepar_network_forward(
    num_feat_dynamic_real: int,
    num_feat_static_real: int,
    num_feat_static_cat: int,
    cardinality: Optional[List[int]],
):
    prediction_length = 6
    context_length = 12

    network = DeepARLightningNetwork(
        freq="1H",
        prediction_length=prediction_length,
        context_length=context_length,
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_feat_static_real=num_feat_static_real,
        num_feat_static_cat=num_feat_static_cat,
        cardinality=cardinality,
        num_parallel_samples=100,
    )

    feat_static_cat = torch.zeros(4, num_feat_static_cat, dtype=torch.long)
    feat_static_real = torch.ones(4, num_feat_static_real)
    past_time_feat = torch.ones(4, network._past_length, num_feat_dynamic_real)
    future_time_feat = torch.ones(4, prediction_length, num_feat_dynamic_real)
    past_target = torch.ones(4, network._past_length)
    past_observed_values = torch.ones(4, network._past_length)
    future_target = torch.ones(4, prediction_length)
    future_observed_values = torch.ones(4, prediction_length)

    network.train()

    distr = network(
        feat_static_cat,
        feat_static_real,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
        future_target,
    )

    assert distr.batch_shape == (4, context_length + prediction_length)
    assert distr.event_shape == ()

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

    assert network.training_step(batch).shape == (4,)
    assert network.validation_step(batch).shape == (4,)

    network.eval()

    samples = network(
        feat_static_cat,
        feat_static_real,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
    )

    assert samples.shape == (4, 100, prediction_length)
