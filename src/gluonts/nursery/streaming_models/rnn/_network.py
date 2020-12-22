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

import mxnet as mx
import numpy as np

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.regularization import (
    ActivationRegularizationLoss,
    TemporalActivationRegularizationLoss,
)
from gluonts.mx.distribution import Distribution, DistributionOutput
from gluonts.mx.distribution.distribution import getF
from gluonts.mx.util import weighted_average

# Relative imports
from ._parameter import apply_weight_drop

rnn_type_map = {"lstm": mx.gluon.rnn.LSTM, "gru": mx.gluon.rnn.GRU}


class StreamingRnnNetworkBase(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        distr_output: DistributionOutput,
        cardinality: List[int],
        embedding_dimension: List[int],
        dropout_type: Optional[str] = None,
        dropout_rate: float = 0.1,
        skip_initial_window: int = 0,
        dtype: DType = np.float32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.distr_output = distr_output
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.skip_initial_window = skip_initial_window
        self.dtype = dtype

        with self.name_scope():
            self.distr_args_proj = self.distr_output.get_args_proj()
            if self.dropout_type == "variational" and self.dropout_rate:
                self.rnn = mx.gluon.rnn.GRU(
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=0,
                    layout="NTC",
                )
                apply_weight_drop(
                    self.rnn, ".*i2h_weight", rate=self.dropout_rate, axes=(1,)
                )
                apply_weight_drop(
                    self.rnn, ".*h2h_weight", rate=self.dropout_rate, axes=(1,)
                )
            else:
                # the last layer of rnn has no dropout
                self.rnn = mx.gluon.rnn.GRU(
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout_rate,
                    layout="NTC",
                )

            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=embedding_dimension,
                dtype=self.dtype,
            )

    def _normalize(self, F, t: Tensor, scale: Tensor) -> Tensor:
        return F.broadcast_div(t, scale).clip(0, 10.0)

    def get_distr_args_and_state(
        self,
        lags: Tensor,  # (batch_size, time)
        scale: Tensor,
        feat_static_cat: Tensor,
        rnn_state: Optional[List[Tensor]] = None,
        F=None,
    ) -> Tuple[Tuple[Tensor, ...], Optional[List[Tensor]], Tensor]:
        F = getF(lags) if F is None else F

        lags = self._normalize(F, lags, scale=scale.expand_dims(axis=-1)) - 1.0

        embedded_cat = self.embedder(feat_static_cat)
        input_tensor = F.concat(
            lags,
            F.broadcast_like(
                embedded_cat.expand_dims(axis=-2),
                lags,
                lhs_axes=-2,
                rhs_axes=-2,
            ),
            # F.log(scale + 1e-8).expand_dims(axis=-1).ones_like(),
            dim=-1,
        )

        if rnn_state is None:
            output = self.rnn(input_tensor)
        else:
            output = self.rnn(input_tensor, rnn_state)

        output, new_rnn_state = (
            output if rnn_state is not None else (output, None)
        )
        distr_args = self.distr_args_proj(output)
        # cache the output of rnn layers, so that it can be used for regularization later
        # assume no dropout for outputs, so can be directly used for activation regularization
        return distr_args, new_rnn_state, output

    def get_distr(
        self,
        lags: Tensor,
        scale: Tensor,
        feat_static_cat: Tensor = None,
        rnn_state: Optional[List[Tensor]] = None,
        F=None,
    ) -> Tuple[Distribution, Tensor]:
        if feat_static_cat is None:
            feat_static_cat = F.broadcast_like(
                F.zeros(shape=(1,)).expand_dims(axis=0),
                lags,
                lhs_axes=0,
                rhs_axes=0,
            )

        distr_args, _, rnn_outputs = self.get_distr_args_and_state(
            lags,
            scale,
            feat_static_cat,
            rnn_state=rnn_state,
            F=F,
        )
        return self.distr_output.distribution(distr_args), rnn_outputs


class StreamingRnnTrainNetwork(StreamingRnnNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    @validated()
    def __init__(self, alpha: float = 0, beta: float = 0, **kwargs) -> None:
        super().__init__(**kwargs)

        # regularizartion weights
        self.alpha = alpha
        self.beta = beta

        # layout is NTC
        if alpha:
            self.ar_loss = ActivationRegularizationLoss(
                alpha, time_axis=1, batch_axis=0
            )
        if beta:
            self.tar_loss = TemporalActivationRegularizationLoss(
                beta, time_axis=1, batch_axis=0
            )

    def hybrid_forward(
        self,
        F,
        lags: Tensor,
        label_target: Tensor,
        scale: Tensor,
        observed_values: Tensor,
        feat_static_cat: Tensor,
    ) -> Tensor:

        distr, rnn_outputs = self.get_distr(lags, scale, feat_static_cat, F=F)
        if self.alpha or self.beta:
            # rnn_outputs is already merged into a single tensor
            assert not isinstance(rnn_outputs, list)

        label_target = self._normalize(F, label_target, scale=scale)

        loss = distr.loss(label_target)

        weighted_loss = weighted_average(
            F=F,
            x=loss.slice_axis(
                axis=-1, begin=self.skip_initial_window, end=None
            ),
            weights=observed_values.slice_axis(
                axis=-1, begin=self.skip_initial_window, end=None
            ),
            axis=1,
        )

        # add activation regularization
        if self.alpha:
            ar_loss = self.ar_loss(
                rnn_outputs.slice_axis(
                    axis=-2, begin=self.skip_initial_window, end=None
                )
            )
            weighted_loss = weighted_loss + ar_loss
        if self.beta:
            tar_loss = self.tar_loss(
                rnn_outputs.slice_axis(
                    axis=-2, begin=self.skip_initial_window, end=None
                )
            )
            weighted_loss = weighted_loss + tar_loss
        return weighted_loss


class StreamingRnnPredictNetwork(StreamingRnnNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        lags: Tensor,
        scale: Tensor,
        feat_static_cat: Tensor,
        network_state: List[Tensor],
    ) -> Tuple[Tuple[Tensor, ...], Tensor, List[Tensor]]:
        rnn_state = network_state
        rnn_state_t = [t.swapaxes(0, 1) for t in rnn_state]

        distr_args, new_t_state, _ = self.get_distr_args_and_state(
            lags, scale, feat_static_cat, rnn_state=rnn_state_t, F=F
        )
        assert new_t_state is not None

        new_state = [t.swapaxes(0, 1) for t in new_t_state]
        return distr_args, scale, new_state
