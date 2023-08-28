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

from toolz import valmap

import torch

from gluonts.torch.model.deepar import DeepARModel
from gluonts.torch.model.tft import TemporalFusionTransformerModel
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardModel


def all_equal(obj1, obj2):
    if isinstance(obj1, tuple):
        return all(all_equal(el1, el2) for el1, el2 in zip(obj1, obj2))
    assert isinstance(obj1, torch.Tensor)
    return (obj1 == obj2).all()


@pytest.mark.parametrize(
    "model",
    [
        DeepARModel(
            freq="1H",
            context_length=10,
            prediction_length=3,
            num_feat_dynamic_real=1,
            num_feat_static_real=1,
            num_feat_static_cat=1,
            cardinality=[2],
            scaling=True,
        ),
        TemporalFusionTransformerModel(
            context_length=10,
            prediction_length=3,
            c_feat_dynamic_cat=[2],
            c_feat_static_cat=[2],
            c_past_feat_dynamic_cat=[2],
        ),
        SimpleFeedForwardModel(
            context_length=10,
            prediction_length=3,
        ),
    ],
)
def test_jit_trace(model):
    zeros_input = model.describe_inputs().zeros()
    ones_input = valmap(torch.ones_like, zeros_input)

    script_module = torch.jit.trace(model, tuple(zeros_input.values()))

    torch.manual_seed(0)
    output_1 = model(**ones_input)

    torch.manual_seed(0)
    output_2 = script_module(**ones_input)

    assert all_equal(output_1, output_2)
