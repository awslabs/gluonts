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

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from gluonts.core.component import validated
from pts.model import weighted_average

from .tft_modules import (
    FeatureProjector,
    FeatureEmbedder,
    VariableSelectionNetwork,
    GatedResidualNetwork,
    TemporalFusionEncoder,
    TemporalFusionDecoder,
)
from .tft_output import QuantileOutput


class TemporalFusionTransformerNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        variable_dim: int,
        embed_dim: int,
        num_heads: int,
        num_outputs: int,
        d_past_feat_dynamic_real: List[int],
        c_past_feat_dynamic_cat: List[int],
        d_feat_dynamic_real: List[int],
        c_feat_dynamic_cat: List[int],
        d_feat_static_real: List[int],
        c_feat_static_cat: List[int],
        dropout: float = 0.0,
    ):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.normalize_eps = 1e-5

        self.target_proj = nn.Linear(in_features=1, out_features=variable_dim)

        if d_past_feat_dynamic_real:
            self.past_feat_dynamic_proj = FeatureProjector(
                feature_dims=d_past_feat_dynamic_real,
                embedding_dims=[variable_dim] * len(d_past_feat_dynamic_real),
            )
        else:
            self.past_feat_dynamic_proj = None

        if c_past_feat_dynamic_cat:
            self.past_feat_dynamic_embed = FeatureEmbedder(
                cardinalities=c_past_feat_dynamic_cat,
                embedding_dims=[variable_dim] * len(c_past_feat_dynamic_cat),
            )
        else:
            self.past_feat_dynamic_embed = None

        if d_feat_dynamic_real:
            self.feat_dynamic_proj = FeatureProjector(
                feature_dims=d_feat_dynamic_real,
                embedding_dims=[variable_dim] * len(d_feat_dynamic_real),
            )
        else:
            self.feat_dynamic_proj = None

        if c_feat_dynamic_cat:
            self.feat_dynamic_embed = FeatureEmbedder(
                cardinalities=c_feat_dynamic_cat,
                embedding_dims=[variable_dim] * len(c_feat_dynamic_cat),
            )
        else:
            self.feat_dynamic_embed = None

        if d_feat_static_real:
            self.feat_static_proj = FeatureProjector(
                feature_dims=d_feat_static_real,
                embedding_dims=[variable_dim] * len(d_feat_static_real),
            )
        else:
            self.feat_static_proj = None

        if c_feat_static_cat:
            self.feat_static_embed = FeatureEmbedder(
                cardinalities=c_feat_static_cat,
                embedding_dims=[variable_dim] * len(c_feat_static_cat),
            )
        else:
            self.feat_static_embed = None

        n_feat_static = len(d_feat_static_real) + len(c_feat_static_cat)
        self.static_selector = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=n_feat_static,
            dropout=dropout,
        )

        n_past_feat_dynamic = len(d_past_feat_dynamic_real) + len(
            c_past_feat_dynamic_cat
        )
        n_feat_dynamic = len(d_feat_dynamic_real) + len(c_feat_dynamic_cat)
        self.ctx_selector = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=n_past_feat_dynamic + n_feat_dynamic + 1,
            add_static=True,
            dropout=dropout,
        )

        self.tgt_selector = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=n_feat_dynamic,
            add_static=True,
            dropout=dropout,
        )

        self.selection = GatedResidualNetwork(
            d_hidden=variable_dim,
            dropout=dropout,
        )

        self.enrichment = GatedResidualNetwork(
            d_hidden=variable_dim,
            dropout=dropout,
        )

        self.state_h = GatedResidualNetwork(
            d_hidden=variable_dim,
            d_output=embed_dim,
            dropout=dropout,
        )

        self.state_c = GatedResidualNetwork(
            d_hidden=variable_dim,
            d_output=embed_dim,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalFusionEncoder(
            d_input=variable_dim,
            d_hidden=embed_dim,
        )
        self.temporal_decoder = TemporalFusionDecoder(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_hidden=embed_dim,
            d_var=variable_dim,
            n_head=num_heads,
            dropout=dropout,
        )

        self.quantiles = sum(
            [[i / 10, 1.0 - i / 10] for i in range(1, (num_outputs + 1) // 2)],
            [0.5],
        )
        self.output = QuantileOutput(
            input_size=embed_dim, quantiles=self.quantiles
        )
        self.output_proj = self.output.get_quantile_proj()
        self.loss = self.output.get_loss()

    def _preprocess(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_feat_dynamic_real: torch.Tensor,
        past_feat_dynamic_cat: torch.Tensor,
        feat_dynamic_real: torch.Tensor,
        feat_dynamic_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
    ):
        obs = past_target * past_observed_values
        count = past_observed_values.sum(dim=1, keepdim=True)
        offset = obs.sum(1, keepdim=True) / (count + self.normalize_eps)
        scale = torch.sum(obs**2, 1, keepdim=True) / (
            count + self.normalize_eps
        )
        scale = torch.sqrt(scale - offset**2)

        past_target = (past_target - offset) / (scale + self.normalize_eps)
        past_target = past_target.unsqueeze(-1)

        proj = self.target_proj(past_target)

        past_covariates = []
        future_covariates = []
        static_covariates: List[torch.Tensor] = []

        past_covariates.append(proj)
        if self.past_feat_dynamic_proj is not None:
            projs = self.past_feat_dynamic_proj(past_feat_dynamic_real)
            past_covariates.extend(projs)
        if self.past_feat_dynamic_embed is not None:
            embs = self.past_feat_dynamic_embed(past_feat_dynamic_cat)
            past_covariates.extend(embs)
        if self.feat_dynamic_proj is not None:
            projs = self.feat_dynamic_proj(feat_dynamic_real)
            for proj in projs:
                ctx_proj = proj[:, 0 : self.context_length, ...]
                tgt_proj = proj[:, self.context_length :, ...]
                past_covariates.append(ctx_proj)
                future_covariates.append(tgt_proj)
        if self.feat_dynamic_embed is not None:
            embs = self.feat_dynamic_embed(feat_dynamic_cat)
            for emb in embs:
                ctx_emb = emb[:, 0 : self.context_length, ...]
                tgt_emb = emb[:, self.context_length :, ...]
                past_covariates.append(ctx_emb)
                future_covariates.append(tgt_emb)

        if self.feat_static_proj is not None:
            projs = self.feat_static_proj(feat_static_real)
            static_covariates.extend(projs)
        if self.feat_static_embed is not None:
            embs = self.feat_static_embed(feat_static_cat)
            static_covariates.extend(embs)

        return (
            past_covariates,
            future_covariates,
            static_covariates,
            offset,
            scale,
        )

    def _postprocess(
        self,
        preds: torch.Tensor,
        offset: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        offset = offset.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
        preds = (preds * (scale + self.normalize_eps)) + offset
        return preds

    def forward(
        self,
        past_observed_values: torch.Tensor,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
        static_covariates: torch.Tensor,
    ):
        static_var, _ = self.static_selector(static_covariates)
        c_selection = self.selection(static_var).unsqueeze(1)
        c_enrichment = self.enrichment(static_var).unsqueeze(1)
        c_h = self.state_h(static_var)
        c_c = self.state_c(static_var)

        ctx_input, _ = self.ctx_selector(past_covariates, c_selection)
        tgt_input, _ = self.tgt_selector(future_covariates, c_selection)

        encoding = self.temporal_encoder(
            ctx_input, tgt_input, [c_h.unsqueeze(0), c_c.unsqueeze(0)]
        )
        decoding = self.temporal_decoder(
            encoding, c_enrichment, past_observed_values
        )

        preds = self.output_proj(decoding)

        return preds


class TemporalFusionTransformerTrainingNetwork(
    TemporalFusionTransformerNetwork
):
    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_feat_dynamic_real: torch.Tensor,
        past_feat_dynamic_cat: torch.Tensor,
        feat_dynamic_real: torch.Tensor,
        feat_dynamic_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
    ) -> torch.Tensor:
        (
            past_covariates,
            future_covariates,
            static_covariates,
            offset,
            scale,
        ) = self._preprocess(
            past_target,
            past_observed_values,
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
        )

        preds = super().forward(
            past_observed_values,
            past_covariates,
            future_covariates,
            static_covariates,
        )

        preds = self._postprocess(preds, offset, scale)

        loss = self.loss(future_target, preds)
        loss = weighted_average(loss, future_observed_values)
        return loss.mean()


class TemporalFusionTransformerPredictionNetwork(
    TemporalFusionTransformerNetwork
):
    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_feat_dynamic_real: torch.Tensor,
        past_feat_dynamic_cat: torch.Tensor,
        feat_dynamic_real: torch.Tensor,
        feat_dynamic_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
    ) -> torch.Tensor:
        (
            past_covariates,
            future_covariates,
            static_covariates,
            offset,
            scale,
        ) = self._preprocess(
            past_target,
            past_observed_values,
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
        )

        preds = super().forward(
            past_observed_values,
            past_covariates,
            future_covariates,
            static_covariates,
        )

        preds = self._postprocess(preds, offset, scale)
        return preds.permute(0, 2, 1)
