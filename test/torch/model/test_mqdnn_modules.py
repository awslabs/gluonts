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

from gluonts.torch.model.mqdnn.module import MQCNNModel, MQRNNModel
from gluonts.torch.model.mqdnn.lightning_module import MQDNNLightningModule


@pytest.mark.parametrize(
    "model, inputs, num_quantiles, prediction_length",
    [
        (
            MQCNNModel(
                prediction_length=2,
                num_output_quantiles=5,
                num_feat_dynamic_real=3,
                num_feat_static_real=4,
                cardinalities=[],
                embedding_dimensions=[],
                decoder_latent_length=6,
                global_hidden_sizes=[10, 10],
                local_hidden_sizes=[5, 5],
                is_iqf=True,
            ),
            dict(
                past_target=torch.rand((4, 50)),
                past_feat_dynamic=torch.rand((4, 50, 3)),
                future_feat_dynamic=torch.rand((4, 2, 3)),
                feat_static_real=torch.rand((4, 4)),
            ),
            5,
            2,
        ),
        (
            MQRNNModel(
                prediction_length=2,
                num_output_quantiles=5,
                num_feat_dynamic_real=3,
                num_feat_static_real=4,
                cardinalities=[],
                embedding_dimensions=[],
                encoder_hidden_size=5,
                decoder_latent_length=6,
                global_hidden_sizes=[10, 10],
                local_hidden_sizes=[5, 5],
                is_iqf=True,
            ),
            dict(
                past_target=torch.rand((4, 50)),
                past_feat_dynamic=torch.rand((4, 50, 3)),
                future_feat_dynamic=torch.rand((4, 2, 3)),
                feat_static_real=torch.rand((4, 4)),
            ),
            5,
            2,
        ),
    ],
)
def test_mdqnn_model(
    model,
    inputs,
    num_quantiles,
    prediction_length,
):
    batch_size = inputs["past_target"].shape[0]

    quantiles = model(**inputs)

    assert quantiles.shape == (batch_size, num_quantiles, prediction_length)
    assert (quantiles.diff(dim=-2) >= 0).all()


@pytest.mark.parametrize(
    "lightning_module, inputs",
    [
        (
            MQDNNLightningModule(
                MQCNNModel(
                    prediction_length=2,
                    num_output_quantiles=5,
                    num_feat_dynamic_real=3,
                    num_feat_static_real=4,
                    cardinalities=[],
                    embedding_dimensions=[],
                    encoder_channels=[5, 5],
                    encoder_dilations=[1, 5],
                    encoder_kernel_sizes=[3, 3],
                    decoder_latent_length=6,
                    global_hidden_sizes=[10, 10],
                    local_hidden_sizes=[5, 5],
                ),
                quantiles=[0.1, 0.2, 0.5, 0.7, 0.95],
            ),
            dict(
                past_target=torch.rand((4, 50)),
                past_feat_dynamic=torch.rand((4, 50, 3)),
                future_feat_dynamic=torch.rand((4, 2, 3)),
                future_target=torch.rand((4, 2)),
                future_observed_values=torch.ones((4, 2)),
                feat_static_real=torch.rand((4, 4)),
            ),
        ),
        (
            MQDNNLightningModule(
                MQRNNModel(
                    prediction_length=2,
                    num_output_quantiles=5,
                    num_feat_dynamic_real=3,
                    num_feat_static_real=4,
                    cardinalities=[],
                    embedding_dimensions=[],
                    encoder_hidden_size=5,
                    decoder_latent_length=6,
                    global_hidden_sizes=[10, 10],
                    local_hidden_sizes=[5, 5],
                ),
                quantiles=[0.1, 0.2, 0.5, 0.7, 0.95],
            ),
            dict(
                past_target=torch.rand((4, 50)),
                past_feat_dynamic=torch.rand((4, 50, 3)),
                future_feat_dynamic=torch.rand((4, 2, 3)),
                future_target=torch.rand((4, 2)),
                future_observed_values=torch.ones((4, 2)),
                feat_static_real=torch.rand((4, 4)),
            ),
        ),
    ],
)
def test_mqdnn_loss(
    lightning_module,
    inputs,
):
    loss = lightning_module.training_step(
        batch=inputs,
        batch_idx=0,
    )

    assert loss.ndim == 0
