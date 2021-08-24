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
from typing import List, Optional

import numpy as np
from mxnet.gluon import HybridBlock
import mxnet as mx

# First-party imports
from gluonts.core.component import validated
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.mx.model.predictor import Predictor
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters
from gluonts.time_feature import TimeFeature
from gluonts.transform import Transformation


# Relative imports
from ._network import (
    DeepVARHierarchicalPredictionNetwork,
    DeepVARHierarchicalTrainingNetwork,
)


def projection_mat(S, return_constraint_mat: bool = False):
    """
    Generates the projection matrix given the aggregation matrix `S`.

    Parameters
    ----------
    S
        Summation or aggregation matrix. Shape: (total_num_time_series, num_base_time_series)
    return_constraint_mat
        Return the coefficient matrix of the linear constraints?

    Returns
    -------
    Tensor
        Projection matrix, shape (total_num_time_series, total_num_time_series)
    Tensor (if `return_constraint_mat` is True)
        Coefficient matrix of the linear constraints, shape (num_agg_time_series, num_time_series)

    """
    # Re-arrange S matrix to form A matrix (coefficient matrix of the linear constraints)
    # S = [S_agg|I_m_K]^T dim:(m,m_K)
    # A = [I_magg | -S_agg] dim:(m_agg,m)

    m, m_K = S.shape
    m_agg = m - m_K

    # The top `m_agg` rows of the matrix `S` give the aggregation constraint matrix.
    S_agg = S[:m_agg, :]
    A = np.hstack((np.eye(m_agg), -S_agg))

    M = np.eye(m) - A.T @ np.linalg.pinv(A @ A.T) @ A

    if return_constraint_mat:
        return mx.nd.array(M), mx.nd.array(A)
    else:
        return mx.nd.array(M)


class DeepVARHierarchicalEstimator(DeepVAREstimator):
    """
    Constructs a DeepVARHierarchical estimator, which is a hierachical extension of DeepVAR.

    The model has been described in the ICML 2021 paper:
    http://proceedings.mlr.press/v139/rangapuram21a.html


    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    target_dim
        Dimensionality of the input dataset
    S
        Summation or aggregation matrix.
    num_samples_for_loss
        Number of samples to draw from the predicted distribution to compute the training loss.
    likelihood_weight
        Weight for the negative log-likelihood loss. Default: 0.0.
        If not zero, then negative log-likelihood (times `likelihood_weight`) is added to the default CRPS loss.
    CRPS_weight
        Weight for the CRPS loss component. Default: 1.0.
        If zero, then loss is only negative log-likelihood (times `likelihood_weight`).
        If non-zero, then loss is CRPS (times 'CRPS_weight') added to the default negative log-likelihood loss
    sample_LH
        Boolean flag to switch between likelihoods from samples or NN parameters. Default: False
    assert_reconciliation
        Flag to indicate whether to assert if the (projected) samples generated during prediction are coherent.
    coherent_train_samples
        Flag to indicate whether sampling/projection is being done during training
    coherent_pred_samples
        Flag to indicate whether sampling/projection is being done during prediction
    trainer
        Trainer object to be used (default: Trainer())
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    num_layers
        Number of RNN layers (default: 2)
    num_cells
        Number of RNN cells for each layer (default: 40)
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm')
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference. This is a model optimization that does not affect
        the accuracy (default: 100)
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    cardinality
        Number of values of each categorical feature (default: [1])
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: 5])
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: LowrankMultivariateGaussianOutput with dim=target_dim and
        rank=0, i.e., covariance matrix is assumed to be diagonal).
        Note that target dim of the DistributionOutput and the estimator constructor
        call need to match.
    rank
        **Not used**. Multivariate Gaussian with diagonal covariance matrix is used.
        This means setting rank = 0.
    scaling
        Whether to automatically scale the target values (default: true)
    pick_incomplete
        Whether training examples can be sampled with only a part of
        past_length time-units
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq)
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    conditioning_length
        Set maximum length for conditioning the marginal transformation
    use_marginal_transformation
        **Not used**. It is set to False.
    batch_size
        The size of the batches to be used training and prediction.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        S: np.ndarray,
        num_samples_for_loss: int = 200,
        likelihood_weight: float = 0.0,
        CRPS_weight: float = 1.0,
        assert_reconciliation: bool = False,
        coherent_train_samples: bool = True,
        coherent_pred_samples: bool = True,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        distr_output: Optional[DistributionOutput] = None,
        rank: Optional[int] = 0,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        conditioning_length: int = 200,
        use_marginal_transformation: bool = False,
        batch_size: int = 32,
        warmstart_epoch_frac: float = 0.0,
        sample_LH: bool = False,
        seq_axis: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            target_dim=target_dim,
            context_length=context_length,
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            num_parallel_samples=num_parallel_samples,
            dropout_rate=dropout_rate,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            distr_output=distr_output,
            rank=rank,
            scaling=scaling,
            pick_incomplete=pick_incomplete,
            lags_seq=lags_seq,
            time_features=time_features,
            conditioning_length=conditioning_length,
            use_marginal_transformation=use_marginal_transformation,
            trainer=trainer,
            batch_size=batch_size,
            **kwargs,
        )

        # Assert that projection is *not* being done only during training
        assert coherent_pred_samples or (
            not coherent_train_samples
        ), "Cannot project only during training (and not during prediction)"

        self.M, self.A = projection_mat(S, return_constraint_mat=True)
        self.num_samples_for_loss = num_samples_for_loss
        self.likelihood_weight = likelihood_weight
        self.CRPS_weight = CRPS_weight
        self.assert_reconciliation = assert_reconciliation
        self.coherent_train_samples = coherent_train_samples
        self.coherent_pred_samples = coherent_pred_samples
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.sample_LH = sample_LH
        self.seq_axis = seq_axis

    def create_training_network(self) -> DeepVARHierarchicalTrainingNetwork:
        return DeepVARHierarchicalTrainingNetwork(
            M=self.M,
            A=self.A,
            num_samples_for_loss=self.num_samples_for_loss,
            likelihood_weight=self.likelihood_weight,
            CRPS_weight=self.CRPS_weight,
            seq_axis=self.seq_axis,
            coherent_train_samples=self.coherent_train_samples,
            warmstart_epoch_frac=self.warmstart_epoch_frac,
            epochs=self.trainer.epochs,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            sample_LH=self.sample_LH,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            conditioning_length=self.conditioning_length,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = DeepVARHierarchicalPredictionNetwork(
            M=self.M,
            A=self.A,
            assert_reconciliation=self.assert_reconciliation,
            coherent_pred_samples=self.coherent_pred_samples,
            target_dim=self.target_dim,
            num_parallel_samples=self.num_parallel_samples,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            conditioning_length=self.conditioning_length,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            output_transform=self.output_transform,
        )
