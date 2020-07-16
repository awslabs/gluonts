from typing import Optional, List, Tuple

import mxnet as mx
from mxnet import init
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.mx.block.feature import FeatureEmbedder, FeatureAssembler
from gluonts.mx.block.scaler import MeanScaler, NOPScaler

from ._layers import CausalConv1D, DualSelfAttention, PosFFN



class SelfAttentionBlock(HybridBlock):
    @validated()
    def __init__(self,
                 d_hidden: int,
                 m_ffn: int,
                 n_groups: int,
                 n_head: int,
                 dist_enc: Optional[str],
                 pre_ln: bool,
                 dropout: float,
                 temperature: float,
                 **kwargs):
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
                share_values= False,
                dropout=dropout,
                temperature=temperature,
            )
            self.lnorm = nn.LayerNorm(axis=-1)
            self.dropout = nn.Dropout(dropout)
            self.ffn = PosFFN(
                d_model=d_hidden,
                d_hidden=d_hidden*m_ffn,
                pre_ln=pre_ln,
                dropout=dropout,
            )
        
    def hybrid_forward(self, 
                       F, 
                       value: Tensor, 
                       shape: Tensor,
                       mask: Optional[Tensor] = None):
        v = value
        s = shape
        if self.pre_ln:
            value = self.lnorm(value)
        value, shape = self.attention(value, shape, mask=mask)
        value = value + v
        shape = shape + s
        if not self.pre_ln:
            value = self.lnorm(value)
        value = self.ffn(value)
        return value, shape



class SelfAttentionNetwork(HybridBlock):
    @validated()
    def __init__(self,
                 context_length: int,
                 prediction_length: int,
                 d_data: int,
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
                 **kwargs):
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
        self.d_data = d_data
        self.d_hidden = d_hidden
        assert (n_output % 2 == 1) and (n_output <= 9)
        quantiles = sum([[i/10, 1.0-i/10] for i in range(1, (n_output+1)//2)], [0.5])
        self.n_output = n_output
        self.quantiles = nn.Parameter('quantiles', 
                                      shape=(self.n_output,), 
                                      init=init.Constant(quantiles),
                                      differentiable=False)
        # self.offset = nn.Parameter('offset',
        #                            shape=(0,1,self.d_data),
        #                            init=init.Zero(),
        #                            differentiable=True)
        # self.scale = nn.Parameter('scale',
        #                           shape=(0,1,self.d_data),
        #                           init=init.One(),
        #                           differentiable=True)
        self.normalizer_eps = normalizer_eps
        
        with self.name_scope():
            self.shape_proj = HybridConcurrent(axis=-1, prefix='shape_proj_')
            for ksize in self.kernel_sizes:
                self.shape_proj.add(CausalConv1D(
                    channels=self.d_hidden//self.n_groups, 
                    kernel_size=ksize, 
                    prefix=f'conv{ksize}_'
                ))
            self.value_proj = nn.Dense(
                units=self.d_hidden, 
                use_bias=False, 
                flatten=False, 
                weight_initializer=init.Xavier(),
                prefix='value_proj_'
            )
            self._blocks = []
            for layer in range(self.n_layers):
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
                self.register_child(
                    block=block,
                    name=f'block_{layer+1}'
                )
                self._blocks.append(block)
            
            self.covar_proj = nn.Dense(
                units=self.d_hidden, 
                use_bias=True, 
                flatten=False, 
                weight_initializer=init.Xavier(),
                prefix='covar_proj_',
            )
            self.embedder = FeatureEmbedder(
                cardinalities=cardinalities,
                embedding_dims=[self.d_hidden] * len(cardinalities),
                prefix='embedder_',
            )
            self.output_proj = nn.Dense(
                self.d_data * n_output,
                weight_initializer=init.Xavier(),
                prefix='output_proj_',
            )

    def _preprocess(self,
                    F,
                    past_target: Tensor,
                    past_feat_real: Optional[Tensor],
                    past_feat_cat: Optional[Tensor],
                    past_observed_values: Tensor,
                    future_target: Tensor,
                    future_feat_real: Optional[Tensor],
                    future_feat_cat: Optional[Tensor],
                    static_feat_real: Optional[Tensor],
                    static_feat_cat: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        obs = past_target * past_observed_values
        count = F.sum(past_observed_values, axis=1, keepdims=True)
        offset = obs.sum(axis=1, keepdims=True).div(count)
        scale = obs.power(2).sum(axis=1, keepdims=True).div(count)
        scale = scale - offset ** 2
        scale = scale.sqrt()
        
        past_target = (past_target - offset) / (scale+self.normalizer_eps)
        future_target = (future_target - offset) / (scale+self.normalizer_eps)

        def _assemble_covariates(
            dynamic_feat_real: Optional[Tensor],
            dynamic_feat_cat: Optional[Tensor],
            static_feat_real: Optional[Tensor],
            static_feat_cat: Optional[Tensor],
        ) -> Tensor:
            covariates = []
            if dynamic_feat_real is not None:
                covariates.append(dynamic_feat_real)
            if static_feat_real is not None:
                covariates.append(static_feat_real.expand_dims(axis=1).repeat(
                    axis=1, repeats=(1, self.context_length, 1)
                ))
            if len(covariates) > 0:
                covariates = F.concat(*covariates, dim=-1)
                covariates = self.covar_proj(covariates)
            else:
                covariates = None

            categories = []
            if dynamic_feat_cat is not None:
                categories.append(dynamic_feat_cat)
            if static_feat_cat is not None:
                categories.append(static_feat_cat.expand_dims(axis=1).repeat(
                    axis=1, repeats=(1,self.context_length, 1)
                ))
            if len(categories) > 0:
                categories = F.concat(*categories, dim=-1)
                embeddings = self.embedder(categories)
                embeddings = F.reshape(embeddings, shape=(0,0,-4,self.d_hidden,-1)).sum(axis=-1)
                if covariates is not None:
                    covariates = covariates + embeddings
                else:
                    covariates = embeddings
            else:
                pass
            
            return covariates
            
        
        past_covariates = _assemble_covariates(
            past_feat_real, past_feat_cat,
            static_feat_real, static_feat_cat,
        )
        future_covariates = _assemble_covariates(
            future_feat_real, future_feat_cat,
            static_feat_real, static_feat_cat,
        )
        
        return (past_target, 
                past_covariates,
                future_target, 
                future_covariates,
                offset,
                scale)
        
    def _postprocess(self,
                     F, 
                     preds: Tensor,
                     offset: Tensor,
                     scale: Tensor) -> Tensor:
        offset = F.expand_dims(offset, axis=-1)
        scale = F.expand_dims(scale, axis=-1)
        preds = preds * (scale + self.normalizer_eps) + offset
        return preds
    
    def _forward_step(self,
              F,
              horizon: int,
              target: Tensor,
              covars: Optional[Tensor],
              mask: Optional[Tensor]) -> Tensor:
        value = self.value_proj(target)
        shape = self.shape_proj(target)
        if covars is not None:
            shape = shape + covars
        for block in self._blocks:
            value, shape = block(value, shape, mask=mask)
        value = F.slice_axis(value, axis=1, begin=-horizon, end=None)
        preds = self.output_proj(values)
        preds = preds.reshape(0,0,-4,self.d_data,self.n_output)
        return preds



class SelfAttentionTrainingNetwork(SelfAttentionNetwork):
    @staticmethod
    def quantile_loss(F, 
                      target: Tensor, 
                      quantile_forecast: Tensor,
                      quantiles: Tensor,
                      observed_values: Tensor) -> Tensor:
        target = F.expand_dims(target, axis=-1)
        diff = F.broadcast_sub(quantile_forecast, target)
        weight = F.broadcast_sub(F.broadcast_lesser_equal(target, quantile_forecast), quantiles) 
        loss = F.abs(diff * weight) * 2.0
        loss = F.sum(loss, axis=-1)
        loss = F.where(observed_values, F.zeros_like(loss), loss)
        return loss
    
    def hybrid_forward(self, 
                       F,
                       past_target: Tensor,
                       past_feat_real: Optional[Tensor],
                       past_feat_cat: Optional[Tensor],
                       past_observed_values: Tensor,
                       future_target: Tensor,
                       future_feat_real: Optional[Tensor],
                       future_feat_cat: Optional[Tensor],
                       future_observed_values: Tensor,
                       static_feat_real: Optional[Tensor],
                       static_feat_cat: Optional[Tensor],
                       quantiles: Tensor) -> Tensor:
        past_target, past_covariates, future_target, future_covariates, offset, scale = \
            self._preprocess(
                F, 
                past_target, past_feat_real, past_feat_cat, past_observed_values,
                future_target, future_feat_real, future_feat_cat, 
                static_feat_real, static_feat_cat,
            )
        
        target = F.concat(past_target, future_target, dim=1)
        covars = F.concat(past_covariates, future_covariates, dim=1)
        observed_values = F.concat(past_observed_values, future_observed_values, dim=1)
        
        target = F.slice_axis(target, axis=1, begin=0, end=-1)
        covars = F.slice_axis(covars, axis=1, begin=0, end=-1)
        observed_values = F.slice_axis(observed_values, axis=1, begin=0, end=-1)
        
        preds = self._forward_step(F, self.prediction_length, target, covars, 1.0-observed_values)
        preds = self._postprocess(F, preds, offset, scale)
        loss = self.quantile_loss(F, future_target, preds, quantiles)
        
        return loss.mean()
    
    
    
class SelfAttentionPredictionNetwork(SelfAttentionNetwork):
    def hybrid_forward(self,
                       F,
                       past_target: Tensor,
                       past_feat_real: Optional[Tensor],
                       past_feat_cat: Optional[Tensor],
                       past_observed_values: Tensor,
                       future_feat_real: Optional[Tensor],
                       future_feat_cat: Optional[Tensor],
                       static_feat_real: Optional[Tensor],
                       static_feat_cat: Optional[Tensor]) -> Tensor:
        past_target, past_covariates, future_target, future_covariates, offset, scale = \
            self._preprocess(
                F, 
                past_target, past_feat_real, past_feat_cat, past_observed_values,
                future_target, future_feat_real, future_feat_cat, 
                static_feat_real, static_feat_cat,
            )
        
        target = past_target
        covars = past_covariates
        observed_values = past_observed_values
        
        preds = []
        for step in range(self.prediction_length):
            forecast = self._forward_step(F, 1, target, covars, 1.0-observed_values)
            preds.append(forecast)
            next_target = F.slice_axis(forecast, axis=-1, begin=0, end=1)
            next_target = F.squeeze(next_target, axis=-1)
            next_covars = F.slice_axis(future_covariates, axis=1, begin=step, end=step+1)
            next_observed_value = F.ones_like(next_target)
            
            target = F.concat(target, next_target, axis=1)
            covars = F.concat(covars, next_covars, axis=1)
            observed_values = F.concat(observed_values, next_observed_value, axis=1)
        preds = F.concat(*preds, axis=1)
        preds = self._postprocess(F, preds, offset, scale)
        return preds