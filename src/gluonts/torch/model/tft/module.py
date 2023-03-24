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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.modules.quantile_output import QuantileOutput
from gluonts.torch.scaler import StdScaler
from gluonts.torch.util import weighted_average

from .layers import (
    FeatureEmbedder,
    FeatureProjector,
    GatedResidualNetwork,
    TemporalFusionDecoder,
    TemporalFusionEncoder,
    VariableSelectionNetwork,
)


class TemporalFusionTransformerModel(nn.Module):
    """Temporal Fusion Transformer neural network.

    Partially based on the implementation in github.com/kashif/pytorch-transformer-ts.

    Inputs feat_static_real, feat_static_cat and feat_dynamic_real are mandatory.
    Inputs feat_dynamic_cat, past_feat_dynamic_real and past_feat_dynamic_cat are optional.
    """

    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_feat_static_real: Optional[List[int]] = None,  #  Defaults to [1]
        c_feat_static_cat: Optional[List[int]] = None,  #  Defaults to [1]
        d_feat_dynamic_real: Optional[List[int]] = None,  # Defaults to [1]
        c_feat_dynamic_cat: Optional[List[int]] = None,  # Defaults to []
        d_past_feat_dynamic_real: Optional[List[int]] = None,  # Defaults to []
        c_past_feat_dynamic_cat: Optional[List[int]] = None,  # Defaults to []
        quantiles: Optional[List[float]] = None,
        num_heads: int = 4,
        d_hidden: int = 32,
        d_var: int = 32,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.d_var = d_var
        self.dropout_rate = dropout_rate
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantiles = quantiles

        self.d_feat_static_real = d_feat_static_real or [1]
        self.d_feat_dynamic_real = d_feat_dynamic_real or [1]
        self.d_past_feat_dynamic_real = d_past_feat_dynamic_real or []
        self.c_feat_static_cat = c_feat_static_cat or [1]
        self.c_feat_dynamic_cat = c_feat_dynamic_cat or []
        self.c_past_feat_dynamic_cat = c_past_feat_dynamic_cat or []

        self.num_feat_static = len(self.d_feat_static_real) + len(
            self.c_feat_static_cat
        )
        self.num_feat_dynamic = len(self.d_feat_dynamic_real) + len(
            self.c_feat_dynamic_cat
        )
        self.num_past_feat_dynamic = len(self.d_past_feat_dynamic_real) + len(
            self.c_past_feat_dynamic_cat
        )

        self.scaler = StdScaler(dim=1, keepdim=True)

        self.target_proj = nn.Linear(in_features=1, out_features=self.d_var)
        # Past-only dynamic features
        if self.d_past_feat_dynamic_real:
            self.past_feat_dynamic_proj = FeatureProjector(
                feature_dims=self.d_past_feat_dynamic_real,
                embedding_dims=[self.d_var]
                * len(self.d_past_feat_dynamic_real),
            )
        else:
            self.past_feat_dynamic_proj = None

        if self.c_past_feat_dynamic_cat:
            self.past_feat_dynamic_embed = FeatureEmbedder(
                cardinalities=self.c_past_feat_dynamic_cat,
                embedding_dims=[self.d_var]
                * len(self.c_past_feat_dynamic_cat),
            )
        else:
            self.past_feat_dynamic_embed = None

        # Known dynamic features
        if self.d_feat_dynamic_real:
            self.feat_dynamic_proj = FeatureProjector(
                feature_dims=self.d_feat_dynamic_real,
                embedding_dims=[self.d_var] * len(self.d_feat_dynamic_real),
            )
        else:
            self.feat_dynamic_proj = None

        if self.c_feat_dynamic_cat:
            self.feat_dynamic_embed = FeatureEmbedder(
                cardinalities=self.c_feat_dynamic_cat,
                embedding_dims=[self.d_var] * len(self.c_feat_dynamic_cat),
            )
        else:
            self.feat_dynamic_embed = None

        # Static features
        if self.d_feat_static_real:
            self.feat_static_proj = FeatureProjector(
                feature_dims=self.d_feat_static_real,
                embedding_dims=[self.d_var] * len(self.d_feat_static_real),
            )
        else:
            self.feat_static_proj = None

        if self.c_feat_static_cat:
            self.feat_static_embed = FeatureEmbedder(
                cardinalities=self.c_feat_static_cat,
                embedding_dims=[self.d_var] * len(self.c_feat_static_cat),
            )
        else:
            self.feat_static_embed = None

        self.static_selector = VariableSelectionNetwork(
            d_hidden=self.d_var,
            num_vars=self.num_feat_static,
            dropout=self.dropout_rate,
        )
        self.ctx_selector = VariableSelectionNetwork(
            d_hidden=self.d_var,
            num_vars=self.num_past_feat_dynamic + self.num_feat_dynamic + 1,
            add_static=True,
            dropout=self.dropout_rate,
        )
        self.tgt_selector = VariableSelectionNetwork(
            d_hidden=self.d_var,
            num_vars=self.num_feat_dynamic,
            add_static=True,
            dropout=self.dropout_rate,
        )
        self.selection = GatedResidualNetwork(
            d_hidden=self.d_var,
            dropout=self.dropout_rate,
        )
        self.enrichment = GatedResidualNetwork(
            d_hidden=self.d_var,
            dropout=self.dropout_rate,
        )
        self.state_h = GatedResidualNetwork(
            d_hidden=self.d_var,
            d_output=self.d_hidden,
            dropout=self.dropout_rate,
        )
        self.state_c = GatedResidualNetwork(
            d_hidden=self.d_var,
            d_output=self.d_hidden,
            dropout=self.dropout_rate,
        )
        self.temporal_encoder = TemporalFusionEncoder(
            d_input=self.d_var,
            d_hidden=self.d_hidden,
        )
        self.temporal_decoder = TemporalFusionDecoder(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_hidden=self.d_hidden,
            d_var=self.d_var,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
        )
        self.output = QuantileOutput(quantiles=self.quantiles)
        self.output_proj = self.output.get_args_proj(in_features=self.d_hidden)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "feat_static_real": Input(
                    shape=(batch_size, sum(self.d_feat_static_real)),
                    dtype=torch.float,
                ),
                "feat_static_cat": Input(
                    shape=(batch_size, len(self.c_feat_static_cat)),
                    dtype=torch.long,
                ),
                "feat_dynamic_real": Input(
                    shape=(
                        batch_size,
                        self.context_length + self.prediction_length,
                        sum(self.d_feat_dynamic_real),
                    ),
                    dtype=torch.float,
                ),
                "feat_dynamic_cat": Input(
                    shape=(
                        batch_size,
                        self.context_length + self.prediction_length,
                        len(self.c_feat_dynamic_cat),
                    ),
                    dtype=torch.long,
                ),
                "past_feat_dynamic_real": Input(
                    shape=(
                        batch_size,
                        self.context_length,
                        sum(self.d_past_feat_dynamic_real),
                    ),
                    dtype=torch.float,
                ),
                "past_feat_dynamic_cat": Input(
                    shape=(
                        batch_size,
                        self.context_length,
                        len(self.c_past_feat_dynamic_cat),
                    ),
                    dtype=torch.long,
                ),
            },
            torch.zeros,
        )

    def input_types(self) -> Dict[str, torch.dtype]:
        return {
            "past_target": torch.float,
            "past_observed_values": torch.float,
            "feat_static_real": torch.float,
            "feat_static_cat": torch.long,
            "feat_dynamic_real": torch.float,
            "feat_dynamic_cat": torch.long,
            "past_feat_dynamic_real": torch.float,
            "past_feat_dynamic_cat": torch.long,
        }

    def _preprocess(
        self,
        past_target: torch.Tensor,  # [N, T]
        past_observed_values: torch.Tensor,  # [N, T]
        feat_static_real: torch.Tensor,  # [N, D_sr]
        feat_static_cat: torch.Tensor,  # [N, D_sc]
        feat_dynamic_real: torch.Tensor,  # [N, T + H, D_dr]
        feat_dynamic_cat: torch.Tensor,  # [N, T + H, D_dc]
        past_feat_dynamic_real: torch.Tensor,  # [N, T, D_pr]
        past_feat_dynamic_cat: torch.Tensor,  # [N, T, D_pc]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        past_target, loc, scale = self.scaler(
            data=past_target, weights=past_observed_values
        )

        past_covariates = [self.target_proj(past_target.unsqueeze(-1))]
        future_covariates = []
        static_covariates = []
        if self.past_feat_dynamic_proj is not None:
            projs = self.past_feat_dynamic_proj(past_feat_dynamic_real)
            past_covariates.extend(projs)
        if self.past_feat_dynamic_embed is not None:
            embs = self.past_feat_dynamic_embed(past_feat_dynamic_cat)
            past_covariates.extend(embs)

        if self.feat_dynamic_proj is not None:
            projs = self.feat_dynamic_proj(feat_dynamic_real)
            for proj in projs:
                ctx_proj = proj[..., : self.context_length, :]
                tgt_proj = proj[..., self.context_length :, :]
                past_covariates.append(ctx_proj)
                future_covariates.append(tgt_proj)
        if self.feat_dynamic_embed is not None:
            embs = self.feat_dynamic_embed(feat_dynamic_cat)
            for emb in embs:
                ctx_emb = emb[..., : self.context_length, :]
                tgt_emb = emb[..., self.context_length :, :]
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
            loc,
            scale,
        )

    def forward(
        self,
        past_target: torch.Tensor,  # [N, T]
        past_observed_values: torch.Tensor,  # [N, T]
        feat_static_real: Optional[torch.Tensor],  # [N, D_sr]
        feat_static_cat: Optional[torch.Tensor],  # [N, D_sc]
        feat_dynamic_real: Optional[torch.Tensor],  # [N, T + H, D_dr]
        feat_dynamic_cat: Optional[torch.Tensor] = None,  # [N, T + H, D_dc]
        past_feat_dynamic_real: Optional[torch.Tensor] = None,  # [N, T, D_pr]
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,  # [N, T, D_pc]
    ) -> torch.Tensor:
        (
            past_covariates,  # [[N, T, d_var], ...]
            future_covariates,  # [[N, H, d_var], ...]
            static_covariates,  # [[N, d_var], ...]
            loc,  # [N, 1]
            scale,  # [N, 1]
        ) = self._preprocess(
            past_target=past_target,
            past_observed_values=past_observed_values,
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
            feat_dynamic_real=feat_dynamic_real,
            feat_dynamic_cat=feat_dynamic_cat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
        )

        static_var, _ = self.static_selector(static_covariates)  # [N, d_var]
        c_selection = self.selection(static_var).unsqueeze(1)  # [N, 1, d_var]
        c_enrichment = self.enrichment(static_var).unsqueeze(1)
        c_h = self.state_h(static_var)  # [N, self.d_hidden]
        c_c = self.state_c(static_var)  # [N, self.d_hidden]
        states = [c_h.unsqueeze(0), c_c.unsqueeze(0)]

        ctx_input, _ = self.ctx_selector(
            past_covariates, c_selection
        )  # [N, T, d_var]
        tgt_input, _ = self.tgt_selector(
            future_covariates, c_selection
        )  # [N, H, d_var]

        encoding = self.temporal_encoder(
            ctx_input, tgt_input, states
        )  # [N, T + H, d_hidden]
        decoding = self.temporal_decoder(
            encoding, c_enrichment, past_observed_values
        )  # [N, H, d_hidden]
        preds = self.output_proj(decoding)
        output = preds * scale.unsqueeze(-1) + loc.unsqueeze(-1)
        return output.transpose(1, 2)  # [N, Q, H]

    def loss(
        self,
        past_target: torch.Tensor,  # [N, T]
        past_observed_values: torch.Tensor,  # [N, T]
        future_target: torch.Tensor,  # [N, H]
        future_observed_values: torch.Tensor,  # [N, H]
        feat_static_real: torch.Tensor,  # [N, D_sr]
        feat_static_cat: torch.Tensor,  # [N, D_sc]
        feat_dynamic_real: torch.Tensor,  # [N, T + H, D_dr]
        feat_dynamic_cat: Optional[torch.Tensor] = None,  # [N, T + H, D_dc]
        past_feat_dynamic_real: Optional[torch.Tensor] = None,  # [N, T, D_pr]
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,  # [N, T, D_pc]
    ) -> torch.Tensor:
        preds = self.forward(
            past_target=past_target,
            past_observed_values=past_observed_values,
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
            feat_dynamic_real=feat_dynamic_real,
            feat_dynamic_cat=feat_dynamic_cat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
        )  # [N, Q, T]
        loss = self.output.quantile_loss(
            y_true=future_target, y_pred=preds.transpose(1, 2)
        )  # [N, T]
        loss = weighted_average(loss, future_observed_values)  # [N]
        return loss.mean()
