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

# Standard library imports
from typing import List, Tuple
from itertools import product

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.distribution import EmpiricalDistribution
from gluonts.mx.util import assert_shape, weighted_average
from gluonts.mx.distribution import LowrankMultivariateGaussian
from gluonts.model.deepvar._network import (
    DeepVARNetwork,
    DeepVARTrainingNetwork,
    DeepVARPredictionNetwork,
)


class DeepVARHierarchicalNetwork(DeepVARNetwork):
    @validated()
    def __init__(
        self,
        M,
        A,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        conditioning_length: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        seq_axis: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            history_length=history_length,
            context_length=context_length,
            prediction_length=prediction_length,
            distr_output=distr_output,
            dropout_rate=dropout_rate,
            lags_seq=lags_seq,
            target_dim=target_dim,
            conditioning_length=conditioning_length,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            scaling=scaling,
            **kwargs,
        )

        self.M = M
        self.M_broadcast = None
        self.A = A
        self.seq_axis = seq_axis

    def reconcile_samples(self, samples):
        """
        Computes coherent samples by projecting unconstrained `samples` using the matrix `self.M`.

        Parameters
        ----------
        samples
            Unconstrained samples

        Returns
        -------
        Coherent samples
            Tensor, shape same as that of `samples`.

        """
        proj_matrix_shape = self.M.shape  # (num_ts, num_ts)

        num_iter_dims = len(self.seq_axis) if self.seq_axis else 0

        # Expand `M` depending on the shape of samples:
        # If seq_axis = None, during training the first axis is only `batch_size`,
        # in which case `M` would be expanded 3 times; during prediction it would be expanded 2 times since the first
        # axis is `batch_size x  num_parallel_samples`.
        M_expanded = self.M
        for i in range(len(samples.shape[num_iter_dims:-1])):
            M_expanded = M_expanded.expand_dims(axis=0)

        # If seq_axis = None broadcast M to (num_samples, batch_size, seq_len, m, m) during training
        # and to (num_samples * batch_size, seq_len, m, m) during prediction
        # Else broadcast to the appropriate remaining dimension
        _shape = (
            list(samples.shape[:-1])
            if not self.seq_axis
            else [
                samples.shape[i]
                for i in range(len(samples.shape[:-1]))
                if i not in self.seq_axis
            ]
        )
        self.M_broadcast = mx.nd.broadcast_to(
            M_expanded,
            shape=_shape + list(proj_matrix_shape),
        )

        if self.seq_axis:
            # bring the axis to iterate in the beginning
            samples = mx.nd.moveaxis(
                samples, self.seq_axis, list(range(len(self.seq_axis)))
            )

            out = []
            for idx in product(
                *[
                    range(x)
                    for x in [
                        samples.shape[d] for d in range(len(self.seq_axis))
                    ]
                ]
            ):
                s = samples[idx]
                out.append(
                    mx.nd.linalg.gemm2(
                        self.M_broadcast, s.expand_dims(-1)
                    ).squeeze(axis=-1)
                )

            # put the axis in the correct order again
            out = mx.nd.concat(*out, dim=0).reshape(samples.shape)
            out = mx.nd.moveaxis(
                out, list(range(len(self.seq_axis))), self.seq_axis
            )
            return out
        else:
            return mx.nd.linalg.gemm2(
                self.M_broadcast, samples.expand_dims(-1)
            ).squeeze(axis=-1)

    def train_hybrid_forward(
        self,
        F,
        target_dimension_indicator: Tensor,
        past_time_feat: Tensor,
        past_target_cdf: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
        future_target_cdf: Tensor,
        future_observed_values: Tensor,
    ) -> Tuple[Tensor, ...]:
        """
        Computes the loss for training DeepVARHierarchical model, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        F
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """

        seq_len = self.context_length + self.prediction_length

        # unroll the decoder in "training mode", i.e. by providing future data
        # as well
        rnn_outputs, _, scale, lags_scaled, inputs = self.unroll_encoder(
            F=F,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        # put together target sequence
        # (batch_size, seq_len, target_dim)
        target = F.concat(
            past_target_cdf.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            future_target_cdf,
            dim=1,
        )

        # assert_shape(target, (-1, seq_len, self.target_dim))

        distr, distr_args = self.distr(
            time_features=inputs,
            rnn_outputs=rnn_outputs,
            scale=scale,
            lags_scaled=lags_scaled,
            target_dimension_indicator=target_dimension_indicator,
            seq_len=self.context_length + self.prediction_length,
        )

        # Determine which epoch we are currently in.
        self.batch_no += 1
        epoch_no = self.batch_no // self.num_batches_per_epoch + 1
        epoch_frac = epoch_no / self.epochs

        # Sample from multivariate Gaussian distribution if we are using CRPS or LH-sample loss
        # Samples shape: (num_samples, batch_size, seq_len, target_dim)
        if self.sample_LH or (self.CRPS_weight > 0.0):
            raw_samples = distr.sample_rep(
                num_samples=self.num_samples_for_loss, dtype="float32"
            )

            if (
                self.coherent_train_samples
                and epoch_frac > self.warmstart_epoch_frac
            ):
                coherent_samples = self.reconcile_samples(raw_samples)
                assert_shape(coherent_samples, raw_samples.shape)
                samples = coherent_samples
            else:
                samples = raw_samples

        if self.sample_LH:  # likelihoods on samples
            # Compute mean and variance
            mu = samples.mean(axis=0)
            var = mx.nd.square(samples - samples.mean(axis=0)).mean(axis=0)
            likelihoods = (
                -LowrankMultivariateGaussian(
                    dim=samples.shape[-1], rank=0, mu=mu, D=var
                )
                .log_prob(target)
                .expand_dims(axis=-1)
            )
        else:  # likelihoods on network params
            likelihoods = -distr.log_prob(target).expand_dims(axis=-1)
        assert_shape(likelihoods, (-1, seq_len, 1))

        # Pick loss function approach. This avoids sampling if we are only training with likelihoods on params
        if self.CRPS_weight > 0.0:
            loss_CRPS = EmpiricalDistribution(samples=samples).crps_univariate(
                obs=target
            )
            loss_unmasked = (
                self.CRPS_weight * loss_CRPS
                + self.likelihood_weight * likelihoods
            )
        else:  # CRPS_weight = 0.0 (asserted non-negativity above)
            loss_unmasked = likelihoods

        # get mask values
        past_observed_values = F.broadcast_minimum(
            past_observed_values, 1 - past_is_pad.expand_dims(axis=-1)
        )

        # (batch_size, subseq_length, target_dim)
        observed_values = F.concat(
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            future_observed_values,
            dim=1,
        )

        # mask the loss at one time step if one or more observations is missing
        # in the target dimensions (batch_size, subseq_length, 1)
        loss_weights = observed_values.min(axis=-1, keepdims=True)

        assert_shape(loss_weights, (-1, seq_len, 1))  # -1 is batch axis size

        loss = weighted_average(
            F=F, x=loss_unmasked, weights=loss_weights, axis=1
        )

        assert_shape(loss, (-1, -1, 1))

        self.distribution = distr

        return (loss, likelihoods) + distr_args

    def reconciliation_error(self, samples):
        r"""
        Computes the reconciliation error defined by the L-infinity norm of the constraint violation:
                    || Ax ||_{\inf}

        Parameters
        ----------
        samples
            Samples

        Returns
        -------
        Reconciliation error
            Float

        """
        constraint_mat_shape = self.A.shape

        A_expanded = self.A.expand_dims(axis=0)
        A_broadcast = mx.nd.broadcast_to(
            A_expanded, shape=samples.shape[0:1] + constraint_mat_shape
        )
        return mx.nd.max(
            mx.nd.abs(
                mx.nd.linalg_gemm2(A_broadcast, samples, transpose_b=True)
            )
        ).asnumpy()[0]


class DeepVARHierarchicalTrainingNetwork(
    DeepVARHierarchicalNetwork, DeepVARTrainingNetwork
):
    def __init__(
        self,
        num_samples_for_loss: int,
        likelihood_weight: float,
        CRPS_weight: float,
        coherent_train_samples: bool,
        warmstart_epoch_frac: float,
        epochs: float,
        num_batches_per_epoch: float,
        sample_LH: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_samples_for_loss = num_samples_for_loss
        self.likelihood_weight = likelihood_weight
        self.CRPS_weight = CRPS_weight
        self.coherent_train_samples = coherent_train_samples
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.epochs = epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.batch_no = 0
        self.sample_LH = sample_LH

        # Assert CRPS_weight, likelihood_weight, and coherent_train_samples have harmonious values
        assert self.CRPS_weight >= 0.0, "CRPS weight must be non-negative"
        assert (
            self.likelihood_weight >= 0.0
        ), "Likelihood weight must be non-negative!"
        assert (
            self.likelihood_weight + self.CRPS_weight > 0.0
        ), "At least one of CRPS or likelihood weights must be non-zero"
        if self.CRPS_weight == 0.0 and self.coherent_train_samples:
            assert "No sampling being performed. coherent_train_samples flag is ignored"
        if not self.sample_LH == 0.0 and self.coherent_train_samples:
            assert "No sampling being performed. coherent_train_samples flag is ignored"
        if self.likelihood_weight == 0.0 and self.sample_LH:
            assert (
                "likelihood_weight is 0 but sample likelihoods are still being calculated. "
                "Set sample_LH=0 when likelihood_weight=0"
            )


class DeepVARHierarchicalPredictionNetwork(
    DeepVARHierarchicalNetwork, DeepVARPredictionNetwork
):
    @validated()
    def __init__(
        self,
        num_parallel_samples: int,
        assert_reconciliation: bool,
        coherent_pred_samples: bool,
        **kwargs,
    ) -> None:
        super().__init__(num_parallel_samples=num_parallel_samples, **kwargs)
        self._post_process_samples = coherent_pred_samples

        self.assert_reconciliation = assert_reconciliation

    def post_process_samples(self, samples: Tensor):
        """
        Reconcile samples.

        Parameters
        ----------
        samples
            Tensor of shape (num_parallel_samples*batch_size, 1, target_dim)

        Returns
        -------
            Tensor of coherent samples.

        """
        coherent_samples = self.reconcile_samples(samples=samples)

        assert_shape(coherent_samples, samples.shape)

        # assert that A*X_proj ~ 0
        if self.assert_reconciliation:
            assert self.reconciliation_error(samples=coherent_samples) < 1e-2

        return coherent_samples
