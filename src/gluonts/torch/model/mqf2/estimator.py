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

from typing import List, Optional, Dict, Any

from gluonts.torch.model.deepar.estimator import DeepAREstimator
from gluonts.torch.modules.loss import NegativeLogLikelihood, EnergyScore
from gluonts.torch.distributions import MQF2DistributionOutput

from . import MQF2MultiHorizonLightningModule, MQF2MultiHorizonModel

from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature


class MQF2MultiHorizonEstimator(DeepAREstimator):
    r"""
    Estimator class for the model MQF2 proposed in the paper
    ``Multivariate Quantile Function Forecaster``
    by Kan, Aubet, Januschowski, Park, Benidis, Ruthotto, Gasthaus

    This is the multi-horizon (multivariate in time step) variant of MQF2

    This class is based on gluonts.torch.model.deepar.estimator.DeepAREstimator

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    num_layers
        Number of RNN layers
    hidden_size
        Hidden state size of RNN
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    dropout_rate
        Dropout regularization parameter
    num_feat_dynamic_real
        Number of dynamic real-valued features
    num_feat_static_cat
        Number of static categorial features
    num_feat_static_real
        Number of static real-valued features
    cardinality
        Number of values of each categorical feature
    embedding_dimension
        Dimension of the embeddings for categorical features
    scaling
        Whether to automatically scale the target values (default: true)
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq)
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference. This is a model optimization that does not affect
        the accuracy (default: 100)
    icnn_hidden_size
        Hidden layer size of the input convex neural network (icnn)
    icnn_num_layers
        Number of layers of the input convex neural network (icnn)
    is_energy_score
        If True, use energy score as objective function
        otherwise use maximum likelihood
        as objective function (normalizing flows)
    es_num_samples
        Number of samples drawn to approximate the energy score
    beta
        Hyperparameter of the energy score (power of the two terms)
    threshold_input
        Clamping threshold of the (scaled) input when maximum likelihood
        is used as objective function
        this is used to make the forecaster more robust
        to outliers in training samples
    estimate_logdet
        When maximum likelihood is used as the objective function,
        specify whether to use the logdet estimator
        introduced in the paper
        ``Convex potential flows: Universal probability distributions
        with optimal transport and convex optimization``
        If True, the logdet estimator (can be numerically unstable) is used
        otherwise, the logdet is directly computed
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        dropout_rate: float = 0.1,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        icnn_hidden_size: int = 20,
        icnn_num_layers: int = 2,
        is_energy_score: bool = True,
        es_num_samples: int = 50,
        beta: float = 1.0,
        threshold_input: float = 100.0,
        estimate_logdet: bool = False,
    ) -> None:

        assert (
            1 <= beta < 2
        ), "beta should be in [1,2) for energy score to be strictly proper"

        assert (
            threshold_input > 0
        ), "clamping threshold for input must be positive"

        distr_output = MQF2DistributionOutput(
            prediction_length=prediction_length,
            is_energy_score=is_energy_score,
            threshold_input=threshold_input,
            es_num_samples=es_num_samples,
            beta=beta,
        )

        loss = EnergyScore() if is_energy_score else NegativeLogLikelihood()

        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            num_layers=num_layers,
            hidden_size=hidden_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_cat=num_feat_static_cat,
            num_feat_static_real=num_feat_static_real,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            distr_output=distr_output,
            loss=loss,
            scaling=scaling,
            lags_seq=lags_seq,
            time_features=time_features,
            num_parallel_samples=num_parallel_samples,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            trainer_kwargs=trainer_kwargs,
        )

        self.icnn_num_layers = icnn_num_layers
        self.icnn_hidden_size = icnn_hidden_size
        self.is_energy_score = is_energy_score
        self.es_num_samples = es_num_samples
        self.threshold_input = threshold_input
        self.estimate_logdet = estimate_logdet

    def create_lightning_module(self) -> MQF2MultiHorizonLightningModule:
        model = MQF2MultiHorizonModel(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=(
                1 + self.num_feat_dynamic_real + len(self.time_features)
            ),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
            icnn_num_layers=self.icnn_num_layers,
            icnn_hidden_size=self.icnn_hidden_size,
            is_energy_score=self.is_energy_score,
            threshold_input=self.threshold_input,
            es_num_samples=self.es_num_samples,
            estimate_logdet=self.estimate_logdet,
        )

        return MQF2MultiHorizonLightningModule(
            model=model,
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
