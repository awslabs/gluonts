from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn

from pts.modules import FeatureEmbedder as BaseFeatureEmbedder


class FeatureProjector(nn.Module):
    def __init__(
        self,
        feature_dims: List[int],
        embedding_dims: List[int],
    ):
        super().__init__()

        self.__num_features = len(feature_dims)
        if self.__num_features > 1:
            self.feature_slices = (
                feature_dims[0:1] + np.cumsum(feature_dims)[:-1].tolist()
            )
        else:
            self.feature_slices = feature_dims
        self.feature_dims = feature_dims

        self._projector = nn.ModuleList(
            [
                nn.Linear(in_features=in_feature, out_features=out_features)
                for in_feature, out_features in zip(self.feature_dims, embedding_dims)
            ]
        )

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        if self.__num_features > 1:
            real_feature_slices = torch.tensor_split(
                features, self.feature_slices[1:], dim=-1
            )
        else:
            real_feature_slices = [features]

        return [
            proj(real_feature_slice)
            for proj, real_feature_slice in zip(self._projector, real_feature_slices)
        ]


class FeatureEmbedder(BaseFeatureEmbedder):
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        concat_features = super(FeatureEmbedder, self).forward(features=features)

        if self.__num_features > 1:
            features = torch.chunk(concat_features, self.__num_features, dim=-1)
        else:
            features = [concat_features]

        return features


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int = -1, nonlinear: bool = True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = torch.chunk(x, 2, dim=self.dim)
        if self.nonlinear:
            val = torch.tanh(val)
        return torch.sigmoid(gate) * val


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        d_input: Optional[int] = None,
        d_output: Optional[int] = None,
        d_static: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        d_input = d_input or d_hidden
        d_static = d_static or 0
        if d_output is None:
            d_output = d_input
            self.add_skip = False
        else:
            if d_output != d_input:
                self.add_skip = True
                self.skip_proj = nn.Linear(in_features=d_input, out_features=d_output)
            else:
                self.add_skip = False

        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_input + d_static, out_features=d_hidden),
            nn.ELU(),
            nn.Linear(in_features=d_hidden, out_features=d_hidden),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_output * 2),
            GatedLinearUnit(nonlinear=False),
        )

        self.lnorm = nn.LayerNorm(d_output)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.add_skip:
            skip = self.skip_proj(x)
        else:
            skip = x

        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.mlp(x)
        x = self.lnorm(x + skip)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_vars: int,
        dropout: float = 0.0,
        add_static: bool = False,
    ):
        super().__init__()
        self.weight_network = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_input=d_hidden * n_vars,
            d_output=n_vars,
            d_static=d_hidden if add_static else None,
            dropout=dropout,
        )

        self.variable_network = nn.ModuleList(
            [
                GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)
                for _ in range(n_vars)
            ]
        )

    def forward(
        self, variables: List[torch.Tensor], static: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flatten = torch.cat(variables, dim=-1)
        if static is not None:
            static = static.expand_as(variables[0])
        weight = self.weight_network(flatten, static)
        weight = torch.softmax(weight.unsqueeze(-2), dim=-1)

        var_encodings = [net(var) for var, net in zip(variables, self.variable_network)]
        var_encodings = torch.stack(var_encodings, dim=-1)

        var_encodings = torch.sum(var_encodings * weight, dim=-1)

        return var_encodings, weight


class TemporalFusionEncoder(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=d_input, hidden_size=d_hidden, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=d_input, hidden_size=d_hidden, batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            GatedLinearUnit(nonlinear=False),
        )
        if d_input != d_hidden:
            self.skip_proj = nn.Linear(in_features=d_input, out_features=d_hidden)
            self.add_skip = True
        else:
            self.add_skip = False

        self.lnorm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        ctx_input: torch.Tensor,
        tgt_input: torch.Tensor,
        states: List[torch.Tensor],
    ):
        ctx_encodings, states = self.encoder_lstm(ctx_input, states)

        tgt_encodings, _ = self.decoder_lstm(tgt_input, states)

        encodings = torch.cat((ctx_encodings, tgt_encodings), dim=1)
        skip = torch.cat((ctx_input, tgt_input), dim=1)
        if self.add_skip:
            skip = self.skip_proj(skip)
        encodings = self.gate(encodings)
        encodings = self.lnorm(skip + encodings)
        return encodings


class TemporalFusionDecoder(nn.Module):
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_hidden: int,
        d_var: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.enrich = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_static=d_var,
            dropout=dropout,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_hidden, num_heads=n_head, dropout=dropout
        )

        self.att_net = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            GatedLinearUnit(nonlinear=False),
        )
        self.att_lnorm = nn.LayerNorm(d_hidden)

        self.ff_net = nn.Sequential(
            GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            GatedLinearUnit(nonlinear=False),
        )
        self.ff_lnorm = nn.LayerNorm(d_hidden)

        self.register_buffer(
            "attn_mask",
            self._generate_subsequent_mask(
                prediction_length, prediction_length + context_length
            ),
        )

    @staticmethod
    def _generate_subsequent_mask(
        target_length: int, source_length: int
    ) -> torch.Tensor:
        mask = (torch.triu(torch.ones(source_length, target_length)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(
        self, x: torch.Tensor, static: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        static = static.repeat((1, self.context_length + self.prediction_length, 1))

        skip = x[:, self.context_length :, ...]
        x = self.enrich(x, static)

        mask_pad = torch.ones_like(mask)[:, 0:1, ...]
        mask_pad = mask_pad.repeat((1, self.prediction_length))
        key_padding_mask = torch.cat((mask, mask_pad), dim=1).bool()

        query_key_value = x.permute(1, 0, 2)

        attn_output, _ = self.attention(
            query=query_key_value[-self.prediction_length :, ...],
            key=query_key_value,
            value=query_key_value,
            # key_padding_mask=key_padding_mask, # does not work on GPU :-(
            attn_mask=self.attn_mask,
        )
        att = self.att_net(attn_output.permute(1, 0, 2))

        x = x[:, self.context_length :, ...]
        x = self.att_lnorm(x + att)
        x = self.ff_net(x)
        x = self.ff_lnorm(x + skip)

        return x
