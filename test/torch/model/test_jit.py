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
from gluonts.torch.model.tft import TemporalFusionTransformerModel


def get_model_and_input(model):
    return model, tuple(
        [
            torch.zeros(shape, dtype=model.input_types()[name])
            for (name, shape) in model.input_shapes().items()
        ]
    )


@pytest.mark.parametrize(
    "model, input", [
        get_model_and_input(
            DeepARModel(
                freq="1H",
                context_length=10,
                prediction_length=3,
                num_feat_dynamic_real=1,
                num_feat_static_real=1,
                num_feat_static_cat=1,
                cardinality=[2],
                scaling=True,
            )
        ),
        get_model_and_input(
            TemporalFusionTransformerModel(
                context_length=10,
                prediction_length=3,
            )
        ),
    ]
)
def test_jit_trace(model, input):
    torch.manual_seed(0)
    output_1 = model(*input)
    script_module = torch.jit.trace(model, input)

    torch.manual_seed(0)
    output_2 = script_module(*input)

    assert (output_1 == output_2).all()
