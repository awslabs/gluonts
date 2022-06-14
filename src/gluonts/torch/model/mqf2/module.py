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

import torch

from gluonts.core.component import validated
from gluonts.torch.model.deepar.module import DeepARModel
from gluonts.torch.distributions import DistributionOutput

from cpflows.flows import ActNorm
from cpflows.icnn import PICNN
from .icnn_utils import DeepConvexNet, SequentialNet


class MQF2MultiHorizonModel(DeepARModel):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        distr_output: DistributionOutput,
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
        icnn_hidden_size: int = 20,
        icnn_num_layers: int = 2,
        is_energy_score: bool = True,
        threshold_input: float = 100,
        es_num_samples: int = 50,
        estimate_logdet: bool = False,
    ) -> None:
        r"""
        Model class for the model MQF2 proposed in the paper
        ``Multivariate Quantile Function Forecaster``
        by Kan, Aubet, Januschowski, Park, Benidis, Ruthotto, Gasthaus

        This is the multi-horizon (multivariate in time step) variant of MQF2

        This class is based on gluonts.torch.model.deepar.module.DeepARModel

        Refer to MQF2MultiHorizonEstimator for the description of parameters
        """

        super().__init__(
            freq=freq,
            context_length=context_length,
            prediction_length=prediction_length,
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_real=num_feat_static_real,
            num_feat_static_cat=num_feat_static_cat,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            distr_output=distr_output,
            lags_seq=lags_seq,
            scaling=scaling,
            num_parallel_samples=num_parallel_samples,
        )

        self.threshold_input = threshold_input
        self.es_num_samples = es_num_samples

        convexnet = PICNN(
            dim=prediction_length,
            dimh=icnn_hidden_size,
            dimc=hidden_size,
            num_hidden_layers=icnn_num_layers,
            symm_act_first=True,
        )
        deepconvexnet = DeepConvexNet(
            convexnet,
            prediction_length,
            is_energy_score=is_energy_score,
            estimate_logdet=estimate_logdet,
        )

        if is_energy_score:
            networks = [deepconvexnet]
        else:
            networks = [
                ActNorm(prediction_length),
                deepconvexnet,
                ActNorm(prediction_length),
            ]

        self.picnn = SequentialNet(networks)

    def unroll_lagged_rnn(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unrolls the RNN encoder over the context window of the time series
        Returns the hidden state of the RNN and the scale.

        Parameters
        ----------
        feat_static_cat
            static categorial features (batch_size, num_feat_static_cat)
        feat_static_real
            static real-valued features (batch_size, num_feat_static_real)
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target
            Past target values (batch_size, history_length)
        past_observed_values
            Indicate whether or not the values were observed
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target
            Future target values (batch_size, prediction_length)

        Returns
        -------
        hidden_state
            RNN hidden state (batch_size, context_length, hidden_size)
        scale
            Scale calculated from the context window (batch_size, 1)
        """

        _, scale, hidden_state, _, _ = super().unroll_lagged_rnn(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        hidden_state = hidden_state[:, : self.context_length]

        return hidden_state, scale

    @torch.jit.ignore
    def output_distribution(
        self,
        picnn: SequentialNet,
        hidden_state: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        inference: bool = False,
    ) -> torch.distributions.Distribution:
        """
        Returns the MQF2Distribution instance.

        Parameters
        ----------
        picnn
            A SequentialNet instance of a
            partially input convex neural network (picnn)
        hidden_state
            RNN hidden state (batch_size, context_length, hidden_size)
        scale
            scaling of the data (batch_size, 1)
        inference
            If True, pass only the last hidden state
            to the forecaster for prediction
            Otherwise, pass all the hidden states to train the forecaster

        Returns
        -------
        MQF2Distribution instance
            MQF2 parametrized by hidden_state
        """

        if inference:
            hidden_state = hidden_state[:, -1]

        return self.distr_output.distribution(picnn, hidden_state, scale=scale)

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generates the predicted sample paths.

        Parameters
        ----------
        feat_static_cat
            Static categorial features (batch_size, num_feat_static_cat)
        feat_static_real
            Static real-valued features (batch_size, num_feat_static_real)
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target
            Past target values (batch_size, history_length)
        past_observed_values
            Indicator whether or not the values were observed
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        num_parallel_samples
            Number of parallel sample paths generated for each time series

        Returns
        -------
        sample_paths
            Sample paths (batch_size, num_parallel_samples, prediction_length)
        """

        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        # TODO in future: add function to make use of all relevant time feat
        hidden_state, scale = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        distr = self.output_distribution(
            self.picnn, hidden_state, inference=True
        )

        unscaled_future_samples = distr.sample(
            sample_shape=(num_parallel_samples,)
        )

        return unscaled_future_samples * scale.unsqueeze(-1)
