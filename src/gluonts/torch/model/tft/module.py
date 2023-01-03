import torch
import torch.nn as nn

from typing import List, Optional, Dict, Tuple
from gluonts.core.component import validated
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler


from .layers import (
    FeatureEmbedder,
    FeatureProjector,
    VariableSelectionNetwork,
    GatedResidualNetwork,
    TemporalFusionDecoder,
    TemporalFusionEncoder,
)


class TemporalFusionTransformerModel(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_past_feat_dynamic_real: List[int],
        c_past_feat_dynamic_cat: List[int],
        d_feat_dynamic_real: List[int],
        c_feat_dynamic_cat: List[int],
        d_feat_static_real: List[int],
        c_feat_static_cat: List[int],
        quantiles: Optional[List[float]] = None,
        num_heads: int = 4,
        d_hidden: int = 32,
        d_var: int = 32,
        dropout_rate: float = 0.1,
        scaling: bool = True,
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

        self.d_feat_static_real = d_feat_static_real
        self.d_feat_dynamic_real = d_feat_dynamic_real
        self.d_past_feat_dynamic_real = d_past_feat_dynamic_real
        self.c_feat_static_cat = c_feat_static_cat or []
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

        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

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
        # self.output = IncrementalQuantileOutput(quantiles=self.quantiles)
        # self.output_proj = self.output.get_quantile_proj()
        self.output_proj = nn.Linear(
            in_features=self.d_hidden, out_features=len(self.quantiles)
        )

    def input_shapes(self, batch_size=1) -> Dict[str, Tuple[int, ...]]:
        return {
            "past_target": (batch_size, self.context_length),
            "past_observed_values": (batch_size, self.context_length),
            "past_feat_dynamic_real": (
                batch_size,
                self.context_length,
                sum(self.d_past_feat_dynamic_real),
            ),
            "past_feat_dynamic_cat": (
                batch_size,
                self.context_length,
                len(self.c_past_feat_dynamic_cat),
            ),
            "feat_dynamic_real": (
                batch_size,
                self.context_length + self.prediction_length,
                sum(self.d_feat_dynamic_real),
            ),
            "feat_dynamic_cat": (
                batch_size,
                self.context_length + self.prediction_length,
                len(self.c_feat_dynamic_cat),
            ),
            "feat_static_real": (batch_size, sum(self.d_feat_static_real)),
            "feat_static_cat": (batch_size, len(self.c_feat_static_cat)),
        }

    def input_types(self) -> Dict[str, torch.dtype]:
        return {
            "past_target": torch.float,
            "past_observed_values": torch.float,
            "past_feat_dynamic_real": torch.float,
            "past_feat_dynamic_cat": torch.long,
            "feat_dynamic_real": torch.float,
            "feat_dynamic_cat": torch.long,
            "feat_static_real": torch.float,
            "feat_static_cat": torch.long,
        }

    def _preprocess(
        self,
        past_target: torch.Tensor,  # [N, T]
        past_observed_values: torch.Tensor,  # [N, T]
        past_feat_dynamic_real: torch.Tensor,  # [N, T, D_pr]
        past_feat_dynamic_cat: torch.Tensor,  # [N, T, D_pc]
        feat_dynamic_real: torch.Tensor,  # [N, T + H, D_dr]
        feat_dynamic_cat: torch.Tensor,  # [N, T + H, D_dc]
        feat_static_real: torch.Tensor,  # [N, D_sr]
        feat_static_cat: torch.Tensor,  # [N, D_sc]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
    ]:
        past_target, scale = self.scaler(
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
            scale,
        )

    def forward(
        self,
        past_target: torch.Tensor,  # [N, T]
        past_observed_values: torch.Tensor,  # [N, T]
        past_feat_dynamic_real: torch.Tensor,  # [N, T, D_pr]
        past_feat_dynamic_cat: torch.Tensor,  # [N, T, D_pc]
        feat_dynamic_real: torch.Tensor,  # [N, T + H, D_dr]
        feat_dynamic_cat: torch.Tensor,  # [N, T + H, D_dc]
        feat_static_real: torch.Tensor,  # [N, D_sr]
        feat_static_cat: torch.Tensor,  # [N, D_sc]
    ) -> torch.Tensor:
        (
            past_covariates,  # [[N, T, d_var], ...]
            future_covariates,  # [[N, H, d_var], ...]
            static_covariates,  # [[N, d_var], ...]
            scale,  # [N, 1]
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
        return preds * scale.unsqueeze(-1)

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_feat_dynamic_real: torch.Tensor,
        past_feat_dynamic_cat: torch.Tensor,
        feat_dynamic_real: torch.Tensor,
        feat_dynamic_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ):
        preds = self.forward(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
        )  # [N, T, Q]
        loss = self.quantile_loss(future_target, preds)  # [N, T]
        return masked_average(loss, future_observed_values)  # [N]
