from typing import Optional, List, Tuple, Union

import mxnet as mx
from mxnet import init
from mxnet.gluon import nn, HybridBlock, Parameter
from mxnet.gluon.contrib.nn import HybridConcurrent
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.mx.block.feature import FeatureEmbedder, FeatureAssembler
from gluonts.mx.block.scaler import MeanScaler, NOPScaler

from ._layers import CausalConv1D, DualSelfAttention, PosFFN

OptTensor = Union[List, Tensor]
is_tensor = lambda obj: isinstance(obj, (mx.ndarray.NDArray, mx.symbol.Symbol))


class SelfAttentionBlock(HybridBlock):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        m_ffn: int,
        n_groups: int,
        n_head: int,
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
            self.attention = DualSelfAttention(
                d_hidden=d_hidden,
                n_groups=n_groups,
                n_head=n_head,
                bias=True,
                bidirectional=False,
                dist_enc=dist_enc,
                share_values=False,
                dropout=dropout,
                temperature=temperature,
            )
            self.lnorm = nn.LayerNorm(axis=-1)
            self.dropout = nn.Dropout(dropout)
            self.ffn = PosFFN(
                d_model=d_hidden,
                d_hidden=d_hidden * m_ffn,
                pre_ln=pre_ln,
                dropout=dropout,
            )

    def hybrid_forward(
        self, F, value: Tensor, shape: Tensor, mask: Tensor,
    ):
        v = value
        s = shape
        if self.pre_ln:
            value = self.lnorm(value)
        value, shape = self.attention(value, shape, mask)
        value = value + v
        shape = shape + s
        if not self.pre_ln:
            value = self.lnorm(value)
        value = self.ffn(value)
        return value, shape


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
        kernel_sizes: List[int],
        dist_enc: Optional[str],
        pre_ln: bool,
        dropout: float,
        temperature: float,
        normalizer_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(kernel_sizes) == 0:
            self.kernel_sizes = (1,)
            self.n_groups = 1
        else:
            self.kernel_sizes = kernel_sizes
            self.n_groups = len(kernel_sizes)
        assert d_hidden % self.n_groups == 0
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_hidden = d_hidden
        assert (n_output % 2 == 1) and (n_output <= 9)
        self.quantiles = sum(
            [[i / 10, 1.0 - i / 10] for i in range(1, (n_output + 1) // 2)],
            [0.5],
        )
        self.n_output = n_output
        self.normalizer_eps = normalizer_eps

        with self.name_scope():
            self.shape_proj = HybridConcurrent(axis=-1, prefix="shape_proj_")
            for ksize in self.kernel_sizes:
                self.shape_proj.add(
                    CausalConv1D(
                        channels=self.d_hidden // self.n_groups,
                        kernel_size=ksize,
                        prefix=f"conv{ksize}_",
                    )
                )
            self.value_proj = nn.Dense(
                units=self.d_hidden,
                use_bias=False,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="value_proj_",
            )
            self._blocks = []
            for layer in range(n_layers):
                block = SelfAttentionBlock(
                    d_hidden=self.d_hidden,
                    m_ffn=m_ffn,
                    n_groups=self.n_groups,
                    n_head=n_head,
                    dist_enc=dist_enc,
                    pre_ln=pre_ln,
                    dropout=dropout,
                    temperature=temperature,
                )
                self.register_child(block=block, name=f"block_{layer+1}")
                self._blocks.append(block)

            self.covar_proj = nn.Dense(
                units=self.d_hidden,
                use_bias=True,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="covar_proj_",
            )
            self.embedder = FeatureEmbedder(
                cardinalities=cardinalities,
                embedding_dims=[self.d_hidden] * len(cardinalities),
                prefix="embedder_",
            )
            self.output_proj = nn.Dense(
                n_output,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="output_proj_",
            )

    def _preprocess(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Tensor,
        past_feat_dynamic_cat: OptTensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_target: OptTensor,
        future_feat_dynamic_real: OptTensor,
        future_feat_dynamic_cat: OptTensor,
        feat_static_real: OptTensor,
        feat_static_cat: OptTensor,
    ) -> Tuple[Tensor, OptTensor, Tensor, Tensor, OptTensor, Tensor, Tensor]:
        obs = past_target * past_observed_values
        count = F.sum(past_observed_values, axis=1, keepdims=True)
        offset = F.sum(obs, axis=1, keepdims=True) / count
        scale = F.sum(obs ** 2, axis=1, keepdims=True) / count
        scale = scale - offset ** 2
        scale = scale.sqrt()

        past_target = (past_target - offset) / (scale + self.normalizer_eps)
        if future_target is not None:
            future_target = (future_target - offset) / (
                scale + self.normalizer_eps
            )

        def _assemble_covariates(
            feat_dynamic_real: OptTensor,
            feat_dynamic_cat: OptTensor,
            feat_static_real: OptTensor,
            feat_static_cat: OptTensor,
            is_past: bool,
        ) -> Tensor:
            covariates = []
            if is_tensor(feat_dynamic_real):
                covariates.append(feat_dynamic_real)
            if is_tensor(feat_static_real):
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
            if is_tensor(feat_dynamic_cat):
                categories.append(feat_dynamic_cat)
            if is_tensor(feat_static_cat):
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
        past_observed_values = F.logical_and(
            past_observed_values, F.logical_not(past_is_pad),
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
        value = self.value_proj(target)
        shape = self.shape_proj(target)
        if covars is not None:
            shape = shape + covars
        for block in self._blocks:
            value, shape = block(value, shape, mask)
        value = F.slice_axis(value, axis=1, begin=-horizon, end=None)
        preds = self.output_proj(value)
        return preds


class SelfAttentionTrainingNetwork(SelfAttentionNetwork):
    def quantile_loss(
        self,
        F,
        target: Tensor,
        quantile_forecast: Tensor,
        observed_values: Tensor,
    ) -> Tensor:
        forecasts = F.split(
            quantile_forecast, axis=-1, num_outputs=self.n_output
        )
        losses = []
        for forecast, quantile in zip(forecasts, self.quantiles):
            forecast = F.squeeze(forecast, axis=-1)
            diff = forecast - target
            weight = F.broadcast_lesser_equal(target, forecast) - quantile
            loss = F.abs(diff * weight) * 2.0
            loss = F.where(observed_values, loss, F.zeros_like(loss))
            losses.append(loss)
        loss = F.stack(*losses, axis=-1).sum(axis=-1)
        return loss

    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: OptTensor,
        past_feat_dynamic_cat: OptTensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_target: Tensor,
        future_feat_dynamic_real: OptTensor,
        future_feat_dynamic_cat: OptTensor,
        future_observed_values: Tensor,
        feat_static_real: OptTensor,
        feat_static_cat: OptTensor,
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
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            past_observed_values,
            past_is_pad,
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
            F, self.prediction_length, target, covars, 1.0 - observed_values
        )
        preds = self._postprocess(F, preds, offset, scale)
        loss = self.quantile_loss(
            F, future_target, preds, future_observed_values
        )

        return loss.mean()


class SelfAttentionPredictionNetwork(SelfAttentionNetwork):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Optional[Tensor],
        past_feat_dynamic_cat: Optional[Tensor],
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_feat_dynamic_real: Optional[Tensor],
        future_feat_dynamic_cat: Optional[Tensor],
        feat_static_real: Optional[Tensor],
        feat_static_cat: Optional[Tensor],
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
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            past_observed_values,
            past_is_pad,
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
                F, 1, target, covars, 1.0 - observed_values
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
        return preds
