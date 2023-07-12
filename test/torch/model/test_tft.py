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

from typing import List

import torch
import pytest

from gluonts.torch.model.tft import TemporalFusionTransformerLightningModule


@pytest.mark.parametrize(
    "d_past_feat_dynamic_real, c_past_feat_dynamic_cat, d_feat_dynamic_real, c_feat_dynamic_cat, d_feat_static_real, c_feat_static_cat, quantiles",
    [
        ([3], [4, 2], [5], [5, 5, 5], [5, 2], [4], [0.1, 0.5, 0.9]),
        ([1], [4, 2], [2, 4], [2], [2], [4, 2, 2], [0.05, 0.25]),
    ],
)
def test_tft_modules(
    d_past_feat_dynamic_real: List[int],
    c_past_feat_dynamic_cat: List[int],
    d_feat_dynamic_real: List[int],
    c_feat_dynamic_cat: List[int],
    d_feat_static_real: List[int],
    c_feat_static_cat: List[int],
    quantiles: List[float],
):
    batch_size = 4
    prediction_length = 6
    context_length = 12

    lightning_module = TemporalFusionTransformerLightningModule(
        {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "d_past_feat_dynamic_real": d_past_feat_dynamic_real,
            "c_past_feat_dynamic_cat": c_past_feat_dynamic_cat,
            "d_feat_dynamic_real": d_feat_dynamic_real,
            "c_feat_dynamic_cat": c_feat_dynamic_cat,
            "d_feat_static_real": d_feat_static_real,
            "c_feat_static_cat": c_feat_static_cat,
            "quantiles": quantiles,
        }
    )
    model = lightning_module.model

    feat_static_cat = torch.zeros(
        batch_size, len(c_feat_static_cat), dtype=torch.long
    )
    feat_static_real = torch.ones(batch_size, sum(d_feat_static_real))
    feat_dynamic_cat = torch.zeros(
        batch_size,
        prediction_length + context_length,
        len(c_feat_dynamic_cat),
        dtype=torch.long,
    )
    feat_dynamic_real = torch.ones(
        batch_size,
        prediction_length + context_length,
        sum(d_feat_dynamic_real),
    )
    past_feat_dynamic_cat = torch.zeros(
        batch_size,
        context_length,
        len(c_past_feat_dynamic_cat),
        dtype=torch.long,
    )
    past_feat_dynamic_real = torch.ones(
        batch_size, context_length, sum(d_past_feat_dynamic_real)
    )
    past_target = torch.ones(batch_size, context_length)
    past_observed_values = torch.ones(batch_size, context_length)
    future_target = torch.ones(batch_size, prediction_length)
    future_observed_values = torch.ones(batch_size, prediction_length)

    (
        past_covariates,
        future_covariates,
        static_covariates,
        loc,
        scale,
    ) = model._preprocess(
        past_target=past_target,
        past_observed_values=past_observed_values,
        feat_static_real=feat_static_real,
        feat_static_cat=feat_static_cat,
        feat_dynamic_real=feat_dynamic_real,
        feat_dynamic_cat=feat_dynamic_cat,
        past_feat_dynamic_real=past_feat_dynamic_real,
        past_feat_dynamic_cat=past_feat_dynamic_cat,
    )
    for x in past_covariates:
        assert x.shape == (batch_size, context_length, model.d_var)
    for x in future_covariates:
        assert x.shape == (batch_size, prediction_length, model.d_var)
    for x in static_covariates:
        assert x.shape == (batch_size, model.d_var)
    assert loc.shape == scale.shape == (batch_size, 1)

    output = model(
        past_target=past_target,
        past_observed_values=past_observed_values,
        feat_static_real=feat_static_real,
        feat_static_cat=feat_static_cat,
        feat_dynamic_real=feat_dynamic_real,
        feat_dynamic_cat=feat_dynamic_cat,
        past_feat_dynamic_real=past_feat_dynamic_real,
        past_feat_dynamic_cat=past_feat_dynamic_cat,
    )

    assert output.shape == (batch_size, len(quantiles), prediction_length)

    batch = dict(
        past_target=past_target,
        past_observed_values=past_observed_values,
        past_feat_dynamic_real=past_feat_dynamic_real,
        past_feat_dynamic_cat=past_feat_dynamic_cat,
        feat_dynamic_real=feat_dynamic_real,
        feat_dynamic_cat=feat_dynamic_cat,
        feat_static_real=feat_static_real,
        feat_static_cat=feat_static_cat,
        future_target=future_target,
        future_observed_values=future_observed_values,
    )

    assert lightning_module.training_step(batch, batch_idx=0).shape == ()
    assert lightning_module.validation_step(batch, batch_idx=0).shape == ()
