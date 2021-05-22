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

from typing import List, Optional

import mxnet as mx

from gluonts.core.component import validated
from gluonts.model.deepstate.issm import ISSM
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution.lds import LDS, LDSArgsProj, ParameterBounds
from gluonts.mx.util import make_nd_diag, weighted_average


class DeepStateNetwork(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        past_length: int,
        prediction_length: int,
        issm: ISSM,
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: List[int],
        scaling: bool = True,
        noise_std_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        prior_cov_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        innovation_bounds: ParameterBounds = ParameterBounds(1e-6, 0.01),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.past_length = past_length
        self.prediction_length = prediction_length
        self.issm = issm
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling

        assert len(cardinality) == len(
            embedding_dimension
        ), "embedding_dimension should be a list with the same size as cardinality"
        self.univariate = self.issm.output_dim() == 1

        self.noise_std_bounds = noise_std_bounds
        self.prior_cov_bounds = prior_cov_bounds
        self.innovation_bounds = innovation_bounds

        with self.name_scope():
            self.prior_mean_model = mx.gluon.nn.Dense(
                units=self.issm.latent_dim(), flatten=False
            )
            self.prior_cov_diag_model = mx.gluon.nn.Dense(
                units=self.issm.latent_dim(),
                activation="sigmoid",
                flatten=False,
            )
            self.lstm = mx.gluon.rnn.HybridSequentialRNNCell()
            self.lds_proj = LDSArgsProj(
                output_dim=self.issm.output_dim(),
                noise_std_bounds=self.noise_std_bounds,
                innovation_bounds=self.innovation_bounds,
            )
            for k in range(num_layers):
                cell = mx.gluon.rnn.LSTMCell(hidden_size=num_cells)
                cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
                cell = (
                    mx.gluon.rnn.ZoneoutCell(cell, zoneout_states=dropout_rate)
                    if dropout_rate > 0.0
                    else cell
                )
                self.lstm.add(cell)
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality, embedding_dims=embedding_dimension
            )
            if scaling:
                self.scaler = MeanScaler(keepdims=False)
            else:
                self.scaler = NOPScaler(keepdims=False)

    def compute_lds(
        self,
        F,
        feat_static_cat: Tensor,
        seasonal_indicators: Tensor,
        time_feat: Tensor,
        length: int,
        prior_mean: Optional[Tensor] = None,
        prior_cov: Optional[Tensor] = None,
        lstm_begin_state: Optional[List[Tensor]] = None,
    ):
        # embed categorical features and expand along time axis
        embedded_cat = self.embedder(feat_static_cat)
        repeated_static_features = embedded_cat.expand_dims(axis=1).repeat(
            axis=1, repeats=length
        )

        # construct big features tensor (context)
        features = F.concat(time_feat, repeated_static_features, dim=2)

        output, lstm_final_state = self.lstm.unroll(
            inputs=features,
            begin_state=lstm_begin_state,
            length=length,
            merge_outputs=True,
        )

        if prior_mean is None:
            prior_input = F.slice_axis(output, axis=1, begin=0, end=1).squeeze(
                axis=1
            )

            prior_mean = self.prior_mean_model(prior_input)
            prior_cov_diag = (
                self.prior_cov_diag_model(prior_input)
                * (self.prior_cov_bounds.upper - self.prior_cov_bounds.lower)
                + self.prior_cov_bounds.lower
            )
            prior_cov = make_nd_diag(F, prior_cov_diag, self.issm.latent_dim())

        (
            emission_coeff,
            transition_coeff,
            innovation_coeff,
        ) = self.issm.get_issm_coeff(seasonal_indicators)

        noise_std, innovation, residuals = self.lds_proj(output)

        lds = LDS(
            emission_coeff=emission_coeff,
            transition_coeff=transition_coeff,
            innovation_coeff=F.broadcast_mul(innovation, innovation_coeff),
            noise_std=noise_std,
            residuals=residuals,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            latent_dim=self.issm.latent_dim(),
            output_dim=self.issm.output_dim(),
            seq_length=length,
        )

        return lds, lstm_final_state


class DeepStateTrainingNetwork(DeepStateNetwork):

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
        past_seasonal_indicators: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
    ) -> Tensor:
        lds, _ = self.compute_lds(
            F,
            feat_static_cat=feat_static_cat,
            seasonal_indicators=past_seasonal_indicators.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            time_feat=past_time_feat.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            length=self.past_length,
        )

        _, scale = self.scaler(past_target, past_observed_values)

        observed_context = past_observed_values.slice_axis(
            axis=1, begin=-self.past_length, end=None
        )

        ll, _, _ = lds.log_prob(
            x=past_target.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            observed=observed_context.min(axis=-1, keepdims=False),
            scale=scale,
        )

        return weighted_average(
            F=F, x=-ll, axis=1, weights=observed_context.squeeze(axis=-1)
        )


class DeepStatePredictionNetwork(DeepStateNetwork):
    @validated()
    def __init__(self, num_parallel_samples: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
        past_seasonal_indicators: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        future_seasonal_indicators: Tensor,
        future_time_feat: Tensor,
    ) -> Tensor:
        lds, lstm_state = self.compute_lds(
            F,
            feat_static_cat=feat_static_cat,
            seasonal_indicators=past_seasonal_indicators.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            time_feat=past_time_feat.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            length=self.past_length,
        )

        _, scale = self.scaler(past_target, past_observed_values)

        observed_context = past_observed_values.slice_axis(
            axis=1, begin=-self.past_length, end=None
        )

        _, final_mean, final_cov = lds.log_prob(
            x=past_target.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            observed=observed_context.min(axis=-1, keepdims=False),
            scale=scale,
        )

        lds_prediction, _ = self.compute_lds(
            F,
            feat_static_cat=feat_static_cat,
            seasonal_indicators=future_seasonal_indicators,
            time_feat=future_time_feat,
            length=self.prediction_length,
            lstm_begin_state=lstm_state,
            prior_mean=final_mean,
            prior_cov=final_cov,
        )

        samples = lds_prediction.sample(
            num_samples=self.num_parallel_samples, scale=scale
        )

        # convert samples from
        # (num_samples, batch_size, prediction_length, target_dim)
        # to
        # (batch_size, num_samples, prediction_length, target_dim)
        # and squeeze last axis in the univariate case
        if self.univariate:
            return samples.transpose(axes=(1, 0, 2, 3)).squeeze(axis=3)
        else:
            return samples.transpose(axes=(1, 0, 2, 3))
