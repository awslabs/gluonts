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

from gluonts.torch.model.patch_tst import PatchTSTModel


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_patchtst_model_dtype_consistency(dtype):
    """PatchTST positional embeddings must match the model dtype (#3198)."""
    torch.set_default_dtype(dtype)
    try:
        model = PatchTSTModel(
            prediction_length=12,
            context_length=24,
            patch_len=8,
            stride=4,
            padding_patch="start",
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            num_feat_dynamic_real=0,
            dropout=0.1,
            activation="relu",
            norm_first=False,
            num_encoder_layers=2,
            scaling="mean",
        )
        batch = model.describe_inputs(batch_size=2).zeros()
        # ensure inputs are the correct dtype
        for key in batch:
            batch[key] = batch[key].to(dtype)
        outputs = model(**batch)
        # verify positional embedding weight matches dtype
        pos_embed = model.positional_encoding
        assert pos_embed.weight.dtype == dtype
    finally:
        torch.set_default_dtype(torch.float32)
