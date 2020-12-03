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

# Third-party import
import mxnet as mx
from mxnet import init
from mxnet.gluon import HybridBlock, Parameter, nn
from mxnet.gluon.contrib.nn import HybridConcurrent

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureAssembler, FeatureEmbedder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.util import weighted_average

from ._layers import PosFFN, SelfAttention


class SelfAttentionBlock(HybridBlock):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        m_ffn: int,
        n_head: int,
        kernel_sizes: List[int],
        dist_enc: Optional[str],
        pre_ln: bool,
        dropout: float,
        temperature: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_hidden = d_hidden
        self.m_ffn = m_ffn
        self.pre_ln = pre_ln

        with self.name_scope():
            self.attention = SelfAttention(
                d_hidden=d_hidden,
                kernel_sizes=kernel_sizes,
                n_head=n_head,
                bias=True,
                bidirectional=False,
                dist_enc=dist_enc,
                share_values=False,
                dropout=dropout,
                temperature=temperature,
                prefix="attn_",
            )
            self.lnorm = nn.LayerNorm(axis=-1)
            self.dropout = nn.Dropout(dropout)
            self.ffn = PosFFN(
                d_model=d_hidden,
                d_hidden=d_hidden * m_ffn,
                pre_ln=pre_ln,
                dropout=dropout,
                prefix="ffn_",
            )

    def hybrid_forward(
        self,
        F,
        x: Tensor,
        mask: Tensor,
    ) -> Tensor:
        skip = x
        if self.pre_ln:
            x = self.lnorm(x)
        x = self.attention(x, mask)
        x = x + skip
        if not self.pre_ln:
            x = self.lnorm(x)
        x = self.ffn(x)
        return x


class SelfAttentionNetwork(HybridBlock):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_hidden: int,
        m_ffn: int,
        n_head: int,
        n_layers: int,
        n_output: int,
        cardinalities: List[int],
        kernel_sizes: Optional[List[int]],
        dist_enc: Optional[str],
        pre_ln: bool,
        dropout: float,
        temperature: float,
        normalizer_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if kernel_sizes is None or len(kernel_sizes) == 0:
            self.kernel_sizes = (1,)
        else:
            self.kernel_sizes = kernel_sizes
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_hidden = d_hidden
        assert (n_output % 2 == 1) and (n_output <= 9)
        self.quantiles = sum(
            [[i / 10, 1.0 - i / 10] for i in range(1, (n_output + 1) // 2)],
            [0.5],
        )
        self.normalizer_eps = normalizer_eps

        with self.name_scope():
            self._blocks = []
            for layer in range(n_layers):
                block = SelfAttentionBlock(
                    d_hidden=self.d_hidden,
                    m_ffn=m_ffn,
                    kernel_sizes=self.kernel_sizes,
                    n_head=n_head,
                    dist_enc=dist_enc,
                    pre_ln=pre_ln,
                    dropout=dropout,
                    temperature=temperature,
                )
                self.register_child(block=block, name=f"block_{layer+1}")
                self._blocks.append(block)

            self.target_proj = nn.Dense(
                units=self.d_hidden,
                in_units=1,
                use_bias=True,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="target_proj_",
            )
            self.covar_proj = nn.Dense(
                units=self.d_hidden,
                use_bias=True,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="covar_proj_",
            )
            if cardinalities:
                self.embedder = FeatureEmbedder(
                    cardinalities=cardinalities,
                    embedding_dims=[self.d_hidden] * len(cardinalities),
                    prefix="embedder_",
                )
            self.output = QuantileOutput(quantiles=self.quantiles)
            self.output_proj = self.output.get_quantile_proj()
            self.loss = self.output.get_loss()

    def _preprocess(
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        past_feat_dynamic_real: Tensor,
        past_feat_dynamic_cat: Tensor,
        future_target: Tensor,
        future_feat_dynamic_real: Tensor,
        future_feat_dynamic_cat: Tensor,
        feat_static_real: Tensor,
        feat_static_cat: Tensor,
    ) -> Tuple[
        Tensor,
        Optional[Tensor],
        Tensor,
        Tensor,
        Optional[Tensor],
        Tensor,
        Tensor,
    ]:
        obs = past_target * past_observed_values
        count = F.sum(past_observed_values, axis=1, keepdims=True)
        offset = F.sum(obs, axis=1, keepdims=True) / (
            count + self.normalizer_eps
        )
        scale = F.sum(obs ** 2, axis=1, keepdims=True) / (
            count + self.normalizer_eps
        )
        scale = scale - offset ** 2
        scale = scale.sqrt()

        past_target = (past_target - offset) / (scale + self.normalizer_eps)
        if future_target is not None:
            future_target = (future_target - offset) / (
                scale + self.normalizer_eps
            )

        def _assemble_covariates(
            feat_dynamic_real: Tensor,
            feat_dynamic_cat: Tensor,
            feat_static_real: Tensor,
            feat_static_cat: Tensor,
            is_past: bool,
        ) -> Tensor:
            covariates = []
            if feat_dynamic_real.shape[-1] > 0:
                covariates.append(feat_dynamic_real)
            if feat_static_real.shape[-1] > 0:
                covariates.append(
                    feat_static_real.expand_dims(axis=1).repeat(
                        axis=1,
                        repeats=self.context_length
                        if is_past
                        else self.prediction_length,
                    )
                )
            if len(covariates) > 0:
                covariates = F.concat(*covariates, dim=-1)
                covariates = self.covar_proj(covariates)
            else:
                covariates = None

            categories = []
            if feat_dynamic_cat.shape[-1] > 0:
                categories.append(feat_dynamic_cat)
            if feat_static_cat.shape[-1] > 0:
                categories.append(
                    feat_static_cat.expand_dims(axis=1).repeat(
                        axis=1,
                        repeats=self.context_length
                        if is_past
                        else self.prediction_length,
                    )
                )
            if len(categories) > 0:
                categories = F.concat(*categories, dim=-1)
                embeddings = self.embedder(categories)
                embeddings = F.reshape(
                    embeddings, shape=(0, 0, -4, self.d_hidden, -1)
                ).sum(axis=-1)
                if covariates is not None:
                    covariates = covariates + embeddings
                else:
                    covariates = embeddings
            else:
                pass

            return covariates

        past_covariates = _assemble_covariates(
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
            is_past=True,
        )
        future_covariates = _assemble_covariates(
            future_feat_dynamic_real,
            future_feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
            is_past=False,
        )
        past_observed_values = F.broadcast_logical_and(
            past_observed_values,
            F.logical_not(past_is_pad),
        )

        return (
            past_target,
            past_covariates,
            past_observed_values,
            future_target,
            future_covariates,
            offset,
            scale,
        )

    def _postprocess(
        self, F, preds: Tensor, offset: Tensor, scale: Tensor
    ) -> Tensor:
        offset = F.expand_dims(offset, axis=-1)
        scale = F.expand_dims(scale, axis=-1)
        preds = preds * (scale + self.normalizer_eps) + offset
        return preds

    def _forward_step(
        self,
        F,
        horizon: int,
        target: Tensor,
        covars: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        target = F.expand_dims(target, axis=-1)
        mask = F.expand_dims(mask, axis=-1)
        value = self.target_proj(target)
        if covars is not None:
            value = value + covars
        for block in self._blocks:
            value = block(value, mask)
        value = F.slice_axis(value, axis=1, begin=-horizon, end=None)
        preds = self.output_proj(value)
        return preds


class SelfAttentionTrainingNetwork(SelfAttentionNetwork):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
        past_feat_dynamic_real: Tensor,
        past_feat_dynamic_cat: Tensor,
        future_feat_dynamic_real: Tensor,
        future_feat_dynamic_cat: Tensor,
        feat_static_real: Tensor,
        feat_static_cat: Tensor,
    ) -> Tensor:
        (
            past_target,
            past_covariates,
            past_observed_values,
            future_target,
            future_covariates,
            offset,
            scale,
        ) = self._preprocess(
            F,
            past_target,
            past_observed_values,
            past_is_pad,
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            future_target,
            future_feat_dynamic_real,
            future_feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
        )

        target = F.concat(past_target, future_target, dim=1)
        covars = F.concat(past_covariates, future_covariates, dim=1)
        observed_values = F.concat(
            past_observed_values, future_observed_values, dim=1
        )

        target = F.slice_axis(target, axis=1, begin=0, end=-1)
        covars = F.slice_axis(covars, axis=1, begin=0, end=-1)
        observed_values = F.slice_axis(
            observed_values, axis=1, begin=0, end=-1
        )

        preds = self._forward_step(
            F, self.prediction_length, target, covars, observed_values
        )
        preds = self._postprocess(F, preds, offset, scale)
        future_target = future_target * (scale + self.normalizer_eps) + offset
        loss = self.loss(future_target, preds)
        loss = weighted_average(F, loss, future_observed_values)
        return loss.mean()


class SelfAttentionPredictionNetwork(SelfAttentionNetwork):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        past_feat_dynamic_real: Tensor,
        past_feat_dynamic_cat: Tensor,
        future_feat_dynamic_real: Tensor,
        future_feat_dynamic_cat: Tensor,
        feat_static_real: Tensor,
        feat_static_cat: Tensor,
    ) -> Tensor:
        (
            past_target,
            past_covariates,
            past_observed_values,
            _,
            future_covariates,
            offset,
            scale,
        ) = self._preprocess(
            F,
            past_target,
            past_observed_values,
            past_is_pad,
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            None,
            future_feat_dynamic_real,
            future_feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
        )

        target = past_target
        covars = past_covariates
        observed_values = past_observed_values

        preds = []
        for step in range(self.prediction_length):
            forecast = self._forward_step(
                F, 1, target, covars, observed_values
            )
            preds.append(forecast)
            next_target = F.slice_axis(forecast, axis=-1, begin=0, end=1)
            next_target = F.squeeze(next_target, axis=-1)
            next_covars = F.slice_axis(
                future_covariates, axis=1, begin=step, end=step + 1
            )
            next_observed_value = F.ones_like(next_target)

            target = F.concat(target, next_target, dim=1)
            covars = F.concat(covars, next_covars, dim=1)
            observed_values = F.concat(
                observed_values, next_observed_value, dim=1
            )
        preds = F.concat(*preds, dim=1)
        preds = self._postprocess(F, preds, offset, scale)
        preds = F.swapaxes(preds, dim1=1, dim2=2)
        return preds
