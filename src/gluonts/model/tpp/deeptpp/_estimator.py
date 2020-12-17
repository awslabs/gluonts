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
from functools import partial
from typing import Optional

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.mx.model.estimator import GluonEstimator, TrainOutput
from gluonts.model.predictor import Predictor
from gluonts.model.tpp import PointProcessGluonPredictor
from gluonts.model.tpp.distribution import TPPDistributionOutput, WeibullOutput
from gluonts.mx.batchify import batchify
from gluonts.mx.trainer import Trainer
from gluonts.transform import (
    Chain,
    ContinuousTimeInstanceSplitter,
    ContinuousTimeUniformSampler,
    RenameFields,
    Transformation,
)

from ._network import DeepTPPPredictionNetwork, DeepTPPTrainingNetwork


class DeepTPPEstimator(GluonEstimator):
    r"""
    DeepTPP is a multivariate point process model based on an RNN.

    After each event :math:`(\tau_i, m_i)`, we feed the inter-arrival time
    :math:`\tau_i` and the mark :math:`m_i` into the RNN. The state :math:`h_i`
    of the RNN represents the history embedding. We use :math:`h_i` to
    parametrize the distribution over the next inter-arrival time
    :math:`p(\tau_{i+1} | h_i)` and the distribution over the next mark
    :math:`p(m_{i+1} | h_i)`. The distribution over the marks is always
    categorical, but different choices are possible for the distribution over
    inter-arrival times - see :code:`gluonts.model.tpp.distribution`.

    The model is a generalization of the approaches described in [DDT+16]_,
    [TWJ19]_ and [SBG20]_.

    References
    ----------


    Parameters
    ----------
    prediction_interval_length
        The length of the interval (in continuous time) that the estimator will
        predict at prediction time.
    context_interval_length
        The length of intervals (in continuous time) that the estimator will be
        trained with.
    num_marks
        The number of marks (distinct processes), i.e., the cardinality of the
        mark set.
    time_distr_output
        TPPDistributionOutput for the distribution over the inter-arrival times.
        See :code:`gluonts.model.tpp.distribution` for possible choices.
    embedding_dim
        The dimension of vector embeddings for marks (used as input to the GRU).
    trainer
        :code:`gluonts.trainer.Trainer` object which will be used to train the
        estimator. Note that :code:`Trainer(hybridize=False)` must be set as
        :code:`DeepTPPEstimator` currently does not support hybridization.
    num_hidden_dimensions
        Number of hidden units in the GRU network.
    num_parallel_samples
        The number of samples returned by the :code:`Predictor` learned.
    num_training_instances
        The number of training instances to be sampled from each entry in the
        data set provided during training.
    freq
        Similar to the :code:`freq` of discrete-time models, specifies the time
        unit by which inter-arrival times are given.
    batch_size
        The size of the batches to be used training and prediction.
    """

    @validated()
    def __init__(
        self,
        prediction_interval_length: float,
        context_interval_length: float,
        num_marks: int,
        time_distr_output: TPPDistributionOutput = WeibullOutput(),
        embedding_dim: int = 5,
        trainer: Trainer = Trainer(hybridize=False),
        num_hidden_dimensions: int = 10,
        num_parallel_samples: int = 100,
        num_training_instances: int = 100,
        freq: str = "H",
        batch_size: int = 32,
    ) -> None:
        assert (
            not trainer.hybridize
        ), "DeepTPP currently only supports the non-hybridized training"

        super().__init__(trainer=trainer, batch_size=batch_size)

        assert (
            prediction_interval_length > 0
        ), "The value of `prediction_interval_length` should be > 0"
        assert (
            context_interval_length is None or context_interval_length > 0
        ), "The value of `context_interval_length` should be > 0"
        assert (
            num_hidden_dimensions > 0
        ), "The value of `num_hidden_dimensions` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"
        assert num_marks > 0, "The value of `num_marks` should be > 0"
        assert (
            num_training_instances > 0
        ), "The value of `num_training_instances` should be > 0"

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_interval_length = prediction_interval_length
        self.context_interval_length = (
            context_interval_length
            if context_interval_length is not None
            else prediction_interval_length
        )
        self.num_marks = num_marks
        self.time_distr_output = time_distr_output
        self.embedding_dim = embedding_dim
        self.num_parallel_samples = num_parallel_samples
        self.num_training_instances = num_training_instances
        self.freq = freq

    def create_training_network(self) -> HybridBlock:
        return DeepTPPTrainingNetwork(
            num_marks=self.num_marks,
            time_distr_output=self.time_distr_output,
            interval_length=self.prediction_interval_length,
            embedding_dim=self.embedding_dim,
            num_hidden_dimensions=self.num_hidden_dimensions,
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                ContinuousTimeInstanceSplitter(
                    past_interval_length=self.context_interval_length,
                    future_interval_length=self.prediction_interval_length,
                    train_sampler=ContinuousTimeUniformSampler(
                        num_instances=self.num_training_instances
                    ),
                ),
                RenameFields(
                    {
                        "past_target": "target",
                        "past_valid_length": "valid_length",
                    }
                ),
            ]
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: DeepTPPTrainingNetwork,
    ) -> Predictor:
        prediction_network = DeepTPPPredictionNetwork(
            num_marks=self.num_marks,
            prediction_interval_length=self.prediction_interval_length,
            interval_length=self.context_interval_length,
            embedding_dim=self.embedding_dim,
            num_hidden_dimensions=self.num_hidden_dimensions,
            time_distr_output=trained_network.time_distr_output,
            params=trained_network.collect_params(),
            num_parallel_samples=self.num_parallel_samples,
        )

        return PointProcessGluonPredictor(
            input_names=["target", "valid_length"],
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_interval_length=self.prediction_interval_length,
            freq=self.freq,
            ctx=self.trainer.ctx,
            input_transform=transformation,
        )

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> TrainOutput:
        return super().train_model(
            training_data,
            validation_data,
            num_workers,
            num_prefetch,
            shuffle_buffer_length,
            stack_fn=partial(batchify, variable_length=True),
            **kwargs,
        )
