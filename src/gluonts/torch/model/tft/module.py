import torch
import torch.nn as nn

from typing import List, Optional, Dict, Tuple
from gluonts.core.component import validated

from .layers import (
    FeatureEmbedder,
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
        num_feat_static_real: int,
        num_feat_dynamic_real: int,
        num_past_feat_dynamic_real: int,
        cardinalities_static: List[int],
        cardinalities_dynamic: List[int],
        cardinalities_past_dynamic: List[int],
        quantiles: Optional[List[float]] = None,
        num_heads: int = 4,
        d_hidden: int = 32,
        d_var: int = 32,
        dropout_rate: float = 0.1,
        scaling: bool = True,
    ) -> None:
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

        self.num_feat_static_real = num_feat_static_real
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_past_feat_dynamic_real = num_past_feat_dynamic_real
        self.cardinalities_static = cardinalities_static or []
        self.cardinalities_dynamic = cardinalities_dynamic or []
        self.cardinalities_past_dynamic = cardinalities_past_dynamic or []
        self.num_feat_static_cat = len(self.cardinalities_static)
        self.num_feat_dynamic_cat = len(self.cardinalities_dynamic)
        self.num_past_feat_dynamic_cat = len(self.cardinalities_past_dynamic)

        self.num_feat_static = (
            self.num_feat_static_real + self.num_feat_static_cat
        )
        self.num_feat_dynamic = (
            self.num_feat_dynamic_real + self.num_feat_dynamic_cat
        )
        self.num_past_feat_dynamic = (
            self.num_past_feat_dynamic_real + self.num_past_feat_dynamic_cat
        )

        self.target_proj = nn.Linear(in_features=1, out_features=self.d_var)
        # Past dynamic features
        if self.num_past_feat_dynamic_real:
            self.past_feat_dynamic_proj = nn.Linear(
                in_features=self.num_past_feat_dynamic_real,
                out_features=self.d_var,
            )
        else:
            self.past_feat_dynamic_proj = None

        if self.cardinalities_past_dynamic:
            self.past_feat_dynamic_embed = FeatureEmbedder(
                cardinalities=cardinalities_past_dynamic,
                embedding_dims=[d_var] * self.num_past_feat_dynamic_cat,
            )
        else:
            self.past_feat_dynamic_embed = None

        # Known dynamic features
        if self.num_feat_dynamic_real:
            self.feat_dynamic_proj = nn.Linear(
                in_features=self.num_feat_dynamic_real,
                out_features=self.d_var,
            )
        else:
            self.feat_dynamic_proj = None

        if self.cardinalities_dynamic:
            self.feat_dynamic_embed = FeatureEmbedder(
                cardinalities=cardinalities_dynamic,
                embedding_dims=[d_var] * self.num_feat_dynamic_cat,
            )
        else:
            self.feat_dynamic_embed = None

        # Static features
        if self.num_feat_static_real:
            self.feat_static_proj = nn.Linear(
                in_features=self.num_feat_static_real,
                out_features=self.d_var,
            )
        else:
            self.feat_static_proj = None

        if self.cardinalities_static:
            self.feat_static_embed = FeatureEmbedder(
                cardinalities=cardinalities_static,
                embedding_dims=[d_var] * self.num_feat_static_cat,
            )
        else:
            self.feat_static_embed = None

        self.static_selector = VariableSelectionNetwork(
            d_hidden=self.d_var,
            num_vars=self.num_feat_static,
            dropout=self.dropout_rate,
        )
        self.context_selector = VariableSelectionNetwork(
            d_hidden=self.d_var,
            num_vars=self.num_past_feat_dynamic + self.num_feat_dynamic + 1,
            add_static=True,
            dropout=self.dropout_rate,
        )
        self.target_selector = VariableSelectionNetwork(
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
            output_dim=self.d_hidden,
            dropout=self.dropout_rate,
        )
        self.state_c = GatedResidualNetwork(
            d_hidden=self.d_var,
            output_dim=self.d_hidden,
            dropout_rate=self.dropout_rate,
        )
        self.temporal_encoder = TemporalFusionEncoder(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
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
        self.output = IncrementalQuantileOutput(quantiles=self.quantiles)
        self.output_proj = self.output.get_quantile_proj()

    def input_shapes(self, batch_size=1) -> Dict[str, Tuple[int, ...]]:
        return {
            "past_target": (batch_size, self.context_length),
            "past_observed_values": (batch_size, self.context_length),
            "past_feat_dynamic_real": (
                batch_size,
                self.context_length,
                self.num_feat_dynamic_real,
            ),
            "past_feat_dynamic_cat": (
                batch_size,
                self.context_length,
                self.num_past_feat_dynamic_cat,
            ),
            "feat_dynamic_real": (
                batch_size,
                self.context_length + self.prediction_length,
                self.num_feat_dynamic_real,
            ),
            "feat_dynamic_cat": (
                batch_size,
                self.context_length + self.prediction_length,
                self.num_feat_dynamic_cat,
            ),
            "feat_static_real": (batch_size, self.num_feat_static_real),
            "feat_static_cat": (batch_size, self.num_feat_static_cat),
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

        static_var, _ = self.static_selector(static_covariates)
        c_selection = self.selection(static_var).expand_dims(axis=1)
        c_enrichment = self.enrichment(static_var).expand_dims(axis=1)
        c_h = self.state_h(static_var)
        c_c = self.state_c(static_var)

        ctx_input, _ = self.ctx_selector(past_covariates, c_selection)
        tgt_input, _ = self.tgt_selector(future_covariates, c_selection)

        encoding = self.temporal_encoder(ctx_input, tgt_input, [c_h, c_c])
        decoding = self.temporal_decoder(
            encoding, c_enrichment, past_observed_values
        )
        preds = self.output_proj(decoding)
        return self._postprocess(preds, offset, scale)

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
        preds = self.forward()  # [N, T, Q]
        loss = self.quantile_loss(future_target, preds)  # [N, T]
        return masked_average(loss, future_observed_values)  # [N]
