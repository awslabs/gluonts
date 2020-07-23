from typing import List, Optional, Tuple

import numpy as np
import mxnet as nx
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn, HybridBlock

from gluonts.model.common import Tensor
from gluonts.mx.block.feature import FeatureEmbedder, FeatureAssembler
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.support.util import weighted_average
from ._layers import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    TemporalFusionEncoder,
    TemporalFusionDecoder,
)


class FeatureProjector(HybridBlock):
    """
    Project a sequence of numerical features.

    Parameters
    ----------
    feature_dims
        dimensions for each numerical feature.

    embedding_dims
        number of dimensions to embed each numerical feature.

    dtype
        Data type of the embedded features.
    """

    @validated()
    def __init__(
        self,
        feature_dims: List[int],
        embedding_dims: List[int],
        dtype: DType = np.float32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert (
            len(feature_dims) > 0
        ), "Length of `cardinalities` list must be greater than zero"
        assert len(feature_dims) == len(
            embedding_dims
        ), "Length of `embedding_dims` and `embedding_dims` should match"
        assert all(
            [c > 0 for c in feature_dims]
        ), "Elements of `feature_dims` should be > 0"
        assert all(
            [d > 0 for d in embedding_dims]
        ), "Elements of `embedding_dims` should be > 0"

        self.feature_dims = feature_dims
        self.dtype = dtype

        def create_projector(i: int, c: int, d: int) -> nn.Dense:
            projection = nn.Dense(
                unit=d,
                in_units=c,
                flatten=True,
                dtype=self.dtype,
                prefix=f"real_{i}_projection_",
            )
            self.register_child(projection)
            return projection

        with self.name_scope():
            self.__projectors = [
                create_embedding(i, c, d)
                for i, (c, d) in enumerate(zip(feature_dims, embedding_dims))
            ]

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, features: Tensor) -> Tensor:
        """

        Parameters
        ----------
        F

        features
            Numerical features with shape: (N,T,C) or (N,C), where C is the
            number of numerical features.

        Returns
        -------
        concatenated_tensor: Tensor
            Concatenated tensor of embeddings whth shape: (N,T,C) or (N,C),
            where C is the sum of the embedding dimensions for each numerical
            feature, i.e. C = sum(self.config.embedding_dims).
        """

        if self.__num_features > 1:
            # we slice the last dimension, giving an array of length self.__num_features with shape (N,T) or (N)
            real_feature_slices = F.split_v2(
                features, tuple(np.cumsum(self.feature_dims)[:-1]), axis=-1,
            )
        else:
            # F.split will iterate over the second-to-last axis if the last axis is one
            real_feature_slices = [features]

        return F.concat(
            *[
                proj(real_feature_slice)
                for proj, real_feature_slice in zip(
                    self.__projectors, real_feature_slices
                )
            ],
            dim=-1,
        )


class TemporalFusionTransformerNetwork(HybridBlock):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_var: int,
        d_hidden: int,
        n_head: int,
        n_output: int,
        d_past_feat_dynamic_real: List[int],
        c_past_feat_dynamic_cat: List[int],
        d_feat_dynamic_real: List[int],
        c_feat_dynamic_cat: List[int],
        d_feat_static_real: List[int],
        c_feat_static_cat: List[int],
        dropout: float = 0.0,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_var = d_var
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.n_output = n_output
        self.quantiles = sum(
            [[i / 10, 1.0 - i / 10] for i in range(1, (n_output + 1) // 2)],
            [0.5],
        )
        self.normalize_eps = 1e-5

        self.d_past_feat_dynamic_real = d_past_feat_dynamic_real
        self.c_past_feat_dynamic_cat = c_past_feat_dynamic_cat
        self.d_feat_dynamic_real = d_feat_dynamic_real
        self.c_feat_dynamic_cat = c_feat_dynamic_cat
        self.d_feat_static_real = d_feat_static_real
        self.c_feat_static_cat = c_feat_static_cat
        self.n_past_feat_dynamic = len(self.d_past_feat_dynamic_real) + len(
            self.c_past_feat_dynamic_cat
        )
        self.n_feat_dynamic = len(self.d_feat_dynamic_real) + len(
            self.c_feat_dynamic_cat
        )
        self.n_feat_static = len(self.d_feat_static_real) + len(
            self.c_feat_static_cat
        )

        with self.name_scope():
            if len(self.d_past_feat_dynamic_real) > 0:
                self.past_feat_dynamic_proj = FeatureProjector(
                    feature_dims=self.d_past_feat_dynamic_real,
                    embedding_dims=[self.d_var]
                    * len(self.d_past_feat_dynamic_real),
                    prefix="past_feat_dynamic_",
                )
            if len(self.c_past_feat_dynamic_cat) > 0:
                self.past_feat_dynamic_embed = FeatureEmbedder(
                    cardinalities=self.c_past_feat_dynamic_cat,
                    embedding_dims=[self.d_var]
                    * len(self.c_past_feat_dynamic_cat),
                    prefix="past_feat_dynamic_",
                )
            if len(self.d_feat_dynamic_real) > 0:
                self.feat_dynamic_proj = FeatureProjector(
                    feature_dims=self.d_feat_dynamic_real,
                    embedding_dims=[self.d_var]
                    * len(self.d_feat_dynamic_real),
                    prefix="feat_dynamic_",
                )
            if len(self.c_feat_dynamic_cat) > 0:
                self.feat_dynamic_embed = FeatureEmbedder(
                    cardinalities=self.c_feat_dynamic_cat,
                    embedding_dims=[self.d_var] * len(self.c_feat_dynamic_cat),
                    prefix="feat_dynamic_",
                )
            if len(self.d_feat_static_real) > 0:
                self.feat_static_proj = FeatureProjector(
                    feature_dims=self.d_feat_static_real,
                    embedding_dims=[self.d_var] * len(self.d_feat_static_real),
                    prefix="feat_static_",
                )
            if len(self.c_feat_static_cat) > 0:
                self.feat_static_embed = FeatureEmbedder(
                    cardinalities=self.c_feat_static_cat,
                    embedding_dims=[self.d_var] * len(self.c_feat_static_cat),
                    prefix="feat_static_",
                )

            self.static_selector = VariableSelectionNetwork(
                d_hidden=self.d_var,
                n_vars=self.n_feat_static,
                dropout=dropout,
            )
            self.ctx_selector = VariableSelectionNetwork(
                d_hidden=self.d_var,
                n_vars=self.n_past_feat_dynamic + self.n_feat_dynamic + 1,
                add_static=True,
                dropout=dropout,
            )
            self.tgt_selector = VariableSelectionNetwork(
                d_hidden=self.d_var,
                n_vars=self.n_feat_dynamic,
                add_static=True,
                dropout=dropout,
            )
            self.selection = GatedResidualNetwork(
                d_hidden=self.d_var, dropout=dropout,
            )
            self.enrichment = GatedResidualNetwork(
                d_hidden=self.d_var, dropout=dropout,
            )
            self.state_h = GatedResidualNetwork(
                d_hidden=self.d_var, d_output=self.d_hidden, dropout=dropout,
            )
            self.state_c = GatedResidualNetwork(
                d_hidden=self.d_var, d_output=self.d_hidden, dropout=dropout,
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
                n_head=self.n_head,
                dropout=dropout,
            )
            self.output = QuantileOutput(quantiles=self.quantiles)
            self.output_proj = self.output.get_quantile_proj()
            self.loss = self.output.get_loss()

    def _preprocess(
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_feat_dynamic_real: Optional[Tensor],
        past_feat_dynamic_cat: Optional[Tensor],
        feat_dynamic_real: Optional[Tensor],
        feat_dynamic_cat: Optional[Tensor],
        feat_static_real: Optional[Tensor],
        feat_static_cat: Optional[Tensor],
    ):
        obs = past_target * past_observed_values
        count = F.sum(past_observed_values, axis=1, keepdims=True)
        offset = F.sum(obs, axis=1, keepdims=True) / count
        scale = F.sum(obs ** 2, axis=1, keepdims=True) / count
        scale = scale - offset ** 2
        scale = scale.sqrt()
        past_target = (past_target - offset) / (scale + self.normalizer_eps)
        past_target = F.expand_dims(past_target, axis=-1)

        past_covariates = []
        future_covariates = []
        static_covariates = []
        if past_feat_dynamic_real is not None:
            proj = self.past_feat_dynamic_proj(past_feat_dynamic_real)
            past_covariates.extend(F.split_v2(proj, self.d_var, axis=-1))
        if past_feat_dynamic_cat is not None:
            emb = self.past_feat_dynamic_embed(past_feat_dynamic_cat)
            past_covariates.extend(F.split_v2(proj, self.d_var, axis=-1))
        if feat_dynamic_real is not None:
            proj = self.feat_dynamic_proj(feat_dynamic_real)
            ctx_proj = F.slice_axis(
                proj, axis=1, begin=0, end=self.context_length
            )
            tgt_proj = F.slice_axis(
                proj, axis=1, begin=self.context_length, end=None
            )
            past_covariates.extend(F.split_v2(ctx_proj, self.d_var, axis=-1))
            future_covariates.extend(F.split_v2(tgt_proj, self.d_var, axis=-1))
        if feat_dynamic_cat is not None:
            emb = self.feat_dynamic_embed(feat_dynamic_cat)
            ctx_emb = F.slice_axis(
                emb, axis=1, begin=0, end=self.context_length
            )
            tgt_emb = F.slice_axis(
                emb, axis=1, begin=self.context_length, end=None
            )
            past_covariates.extend(F.split_v2(ctx_emb, self.d_var, axis=-1))
            future_covariates.extend(F.split_v2(tgt_emb, self.d_var, axis=-1))
        if feat_static_real is not None:
            proj = self.feat_static_proj(feat_static_real)
            static_covariates.extend(F.split_v2(proj, self.d_var, axis=-1))
        if feat_static_cat is not None:
            emb = self.feat_static_embed(feat_static_cat)
            static_covariates.extend(F.split_v2(emb, self.d_var, axis=-1))

        return (
            past_target,
            past_covariates,
            future_covariates,
            static_covariates,
            offset,
            scale,
        )

    def _postprocess(
        self, F, preds: Tensor, offset: Tensor, scale: Tensor,
    ) -> Tensor:
        offset = F.expand_dims(offset, axis=-1)
        scale = F.expand_dims(scale, axis=-1)
        preds = preds * (scale + self.normalizer_eps) + offset
        return preds

    def _forward(
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_covariates: Tensor,
        future_covariates: Tensor,
        static_covariates: Tensor,
    ):
        static_var, _ = self.static_selector(static_covariates)
        c_selection = self.selection(static_var)
        c_enrichment = self.enrichment(static_var)
        c_h = self.state_h(static_var)
        c_c = self.state_c(static_var)

        ctx_input, _ = self.ctx_selector(
            [past_target] + past_covariates, c_selection,
        )
        tgt_input, _ = self.tgt_selector(future_covariates, c_selection,)
        encoding = self.temporal_encoder(ctx_input, tgt_input, [c_h, c_c])
        decoding = self.temporal_decoder(encoding, c_enrichment)
        preds = self.output_proj(decoding)

        return preds


class TemporalFusionTransformerTrainingNetwork(
    TemporalFusionTransformerNetwork
):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
        past_feat_dynamic_real: Optional[Tensor],
        past_feat_dynamic_cat: Optional[Tensor],
        feat_dynamic_real: Optional[Tensor],
        feat_dynamic_cat: Optional[Tensor],
        feat_static_real: Optional[Tensor],
        feat_static_cat: Optional[Tensor],
    ) -> Tensor:
        (
            past_target,
            past_covariates,
            future_covariates,
            static_covariates,
            offset,
            scale,
        ) = self._preprocess(
            F,
            past_target,
            past_observed_values,
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
        )

        preds = self._forward(
            F,
            past_target,
            past_observed_values,
            past_covariates,
            future_covariates,
            static_covariates,
        )

        preds = self._postprocess(F, preds, offset, scale)

        loss = self.loss(F, preds, F.expand_dims(future_target, axis=-1))
        loss = weighted_average(F, loss, future_observed_values)
        return loss.mean()


class TemporalFusionTransformerPredictionNetwork(
    TemporalFusionTransformerNetwork
):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_feat_dynamic_real: Optional[Tensor],
        past_feat_dynamic_cat: Optional[Tensor],
        feat_dynamic_real: Optional[Tensor],
        feat_dynamic_cat: Optional[Tensor],
        feat_static_real: Optional[Tensor],
        feat_static_cat: Optional[Tensor],
    ):

        (
            past_target,
            past_covariates,
            future_covariates,
            static_covariates,
            offset,
            scale,
        ) = self._preprocess(
            F,
            past_target,
            past_observed_values,
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
        )

        preds = self._forward(
            F,
            past_target,
            past_observed_values,
            past_covariates,
            future_covariates,
            static_covariates,
        )

        preds = self._postprocess(F, preds, offset, scale,)

        return preds
