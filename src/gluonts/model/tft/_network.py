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

from typing import List, Optional, Tuple

import mxnet as nx
import numpy as np
from mxnet import gluon, init
from mxnet.gluon import HybridBlock, nn

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder as BaseFeatureEmbedder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.util import weighted_average

from ._layers import (
    GatedResidualNetwork,
    TemporalFusionDecoder,
    TemporalFusionEncoder,
    VariableSelectionNetwork,
)


class FeatureEmbedder(BaseFeatureEmbedder):
    def hybrid_forward(self, F, features: Tensor) -> List[Tensor]:
        concat_features = super(FeatureEmbedder, self).hybrid_forward(
            F, features
        )
        if self.__num_features > 1:
            features = F.split(
                concat_features, num_outputs=self.__num_features, axis=-1
            )
        else:
            features = [concat_features]
        return features


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
            feature_dims
        ), "Length of `cardinalities` list must be greater than zero"
        assert len(feature_dims) == len(
            embedding_dims
        ), "Length of `embedding_dims` and `embedding_dims` should match"
        assert all(
            c > 0 for c in feature_dims
        ), "Elements of `feature_dims` should be > 0"
        assert all(
            d > 0 for d in embedding_dims
        ), "Elements of `embedding_dims` should be > 0"

        self.feature_dims = feature_dims
        self.dtype = dtype
        self.__num_features = len(feature_dims)

        def create_projector(i: int, c: int, d: int) -> nn.Dense:
            projection = nn.Dense(
                units=d,
                in_units=c,
                flatten=False,
                dtype=self.dtype,
                prefix=f"real_{i}_projection_",
            )
            self.register_child(projection)
            return projection

        with self.name_scope():
            self.__projectors = [
                create_projector(i, c, d)
                for i, (c, d) in enumerate(zip(feature_dims, embedding_dims))
            ]

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, features: Tensor) -> List[Tensor]:
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
                features,
                tuple(np.cumsum(self.feature_dims)[:-1]),
                axis=-1,
            )
        else:
            # F.split will iterate over the second-to-last axis if the last axis is one
            real_feature_slices = [features]

        return [
            proj(real_feature_slice)
            for proj, real_feature_slice in zip(
                self.__projectors, real_feature_slices
            )
        ]


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
        **kwargs,
    ):
        super(TemporalFusionTransformerNetwork, self).__init__(**kwargs)
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
            self.target_proj = nn.Dense(
                units=self.d_var,
                in_units=1,
                flatten=False,
                prefix=f"target_projection_",
            )
            if self.d_past_feat_dynamic_real:
                self.past_feat_dynamic_proj = FeatureProjector(
                    feature_dims=self.d_past_feat_dynamic_real,
                    embedding_dims=[self.d_var]
                    * len(self.d_past_feat_dynamic_real),
                    prefix="past_feat_dynamic_",
                )
            else:
                self.past_feat_dynamic_proj = None

            if self.c_past_feat_dynamic_cat:
                self.past_feat_dynamic_embed = FeatureEmbedder(
                    cardinalities=self.c_past_feat_dynamic_cat,
                    embedding_dims=[self.d_var]
                    * len(self.c_past_feat_dynamic_cat),
                    prefix="past_feat_dynamic_",
                )
            else:
                self.past_feat_dynamic_embed = None

            if self.d_feat_dynamic_real:
                self.feat_dynamic_proj = FeatureProjector(
                    feature_dims=self.d_feat_dynamic_real,
                    embedding_dims=[self.d_var]
                    * len(self.d_feat_dynamic_real),
                    prefix="feat_dynamic_",
                )
            else:
                self.feat_dynamic_proj = None

            if self.c_feat_dynamic_cat:
                self.feat_dynamic_embed = FeatureEmbedder(
                    cardinalities=self.c_feat_dynamic_cat,
                    embedding_dims=[self.d_var] * len(self.c_feat_dynamic_cat),
                    prefix="feat_dynamic_",
                )
            else:
                self.feat_dynamic_embed = None

            if self.d_feat_static_real:
                self.feat_static_proj = FeatureProjector(
                    feature_dims=self.d_feat_static_real,
                    embedding_dims=[self.d_var] * len(self.d_feat_static_real),
                    prefix="feat_static_",
                )
            else:
                self.feat_static_proj = None

            if self.c_feat_static_cat:
                self.feat_static_embed = FeatureEmbedder(
                    cardinalities=self.c_feat_static_cat,
                    embedding_dims=[self.d_var] * len(self.c_feat_static_cat),
                    prefix="feat_static_",
                )
            else:
                self.feat_static_embed = None

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
                d_hidden=self.d_var,
                dropout=dropout,
            )
            self.enrichment = GatedResidualNetwork(
                d_hidden=self.d_var,
                dropout=dropout,
            )
            self.state_h = GatedResidualNetwork(
                d_hidden=self.d_var,
                d_output=self.d_hidden,
                dropout=dropout,
            )
            self.state_c = GatedResidualNetwork(
                d_hidden=self.d_var,
                d_output=self.d_hidden,
                dropout=dropout,
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
        past_feat_dynamic_real: Tensor,
        past_feat_dynamic_cat: Tensor,
        feat_dynamic_real: Tensor,
        feat_dynamic_cat: Tensor,
        feat_static_real: Tensor,
        feat_static_cat: Tensor,
    ):
        obs = F.broadcast_mul(past_target, past_observed_values)
        count = F.sum(past_observed_values, axis=1, keepdims=True)
        offset = F.broadcast_div(
            F.sum(obs, axis=1, keepdims=True),
            count + self.normalize_eps,
        )
        scale = F.broadcast_div(
            F.sum(obs ** 2, axis=1, keepdims=True),
            count + self.normalize_eps,
        )
        scale = F.broadcast_sub(scale, offset ** 2)
        scale = F.sqrt(scale)
        past_target = F.broadcast_div(
            F.broadcast_sub(past_target, offset),
            scale + self.normalize_eps,
        )
        past_target = F.expand_dims(past_target, axis=-1)

        past_covariates = []
        future_covariates = []
        static_covariates: List[Tensor] = []
        proj = self.target_proj(past_target)
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
                ctx_proj = F.slice_axis(
                    proj, axis=1, begin=0, end=self.context_length
                )
                tgt_proj = F.slice_axis(
                    proj, axis=1, begin=self.context_length, end=None
                )
                past_covariates.append(ctx_proj)
                future_covariates.append(tgt_proj)
        if self.feat_dynamic_embed is not None:
            embs = self.feat_dynamic_embed(feat_dynamic_cat)
            for emb in embs:
                ctx_emb = F.slice_axis(
                    emb, axis=1, begin=0, end=self.context_length
                )
                tgt_emb = F.slice_axis(
                    emb, axis=1, begin=self.context_length, end=None
                )
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
        F,
        preds: Tensor,
        offset: Tensor,
        scale: Tensor,
    ) -> Tensor:
        offset = F.expand_dims(offset, axis=-1)
        scale = F.expand_dims(scale, axis=-1)
        preds = F.broadcast_add(
            F.broadcast_mul(preds, (scale + self.normalize_eps)),
            offset,
        )
        return preds

    def _forward(
        self,
        F,
        past_observed_values: Tensor,
        past_covariates: Tensor,
        future_covariates: Tensor,
        static_covariates: Tensor,
    ):
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
        past_feat_dynamic_real: Tensor,
        past_feat_dynamic_cat: Tensor,
        feat_dynamic_real: Tensor,
        feat_dynamic_cat: Tensor,
        feat_static_real: Tensor,
        feat_static_cat: Tensor,
    ) -> Tensor:
        (
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
            past_observed_values,
            past_covariates,
            future_covariates,
            static_covariates,
        )

        preds = self._postprocess(F, preds, offset, scale)

        loss = self.loss(future_target, preds)
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
        past_feat_dynamic_real: Tensor,
        past_feat_dynamic_cat: Tensor,
        feat_dynamic_real: Tensor,
        feat_dynamic_cat: Tensor,
        feat_static_real: Tensor,
        feat_static_cat: Tensor,
    ):

        (
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
            past_observed_values,
            past_covariates,
            future_covariates,
            static_covariates,
        )

        preds = self._postprocess(F, preds, offset, scale)
        preds = F.swapaxes(preds, dim1=1, dim2=2)
        return preds
