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

from gluonts.dataset.parallelized_loader import batchify
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.estimator import GluonEstimator, TrainOutput
from gluonts.model.predictor import Predictor
from gluonts.model.tpp import PointProcessGluonPredictor
from gluonts.trainer import Trainer
from gluonts.transform import (
    Chain,
    ContinuousTimeUniformSampler,
    ContinuousTimeInstanceSplitter,
    RenameFields,
    Transformation
)

# Relative imports
from ._network import RMTPPTrainingNetwork, RMTPPPredictionNetwork


class RMTPPEstimator(GluonEstimator):
    """
    The "Recurrent Marked Temporal Point Process" is a marked point process model
    where the conditional intensity function and the mark distribution are
    specified by a recurrent neural network, as described in [Duetal2016]_.

    The model works on a multivariate temporal point process, i.e., the "points"
    take integer marks that index a finite set. RMTPP parameterizes the conditional
    intensity function for the next mark via a single hidden-layer LSTM, which
    takes the interarrival time since, and a vector embedding for the mark of,
    the previous point.

    .. [Duetal2016] Du, N., Dai, H., Trivedi, R., Upadhyay, U., Gomez-Rodriguez, M.,
        & Song, L. (2016, August). Recurrent marked temporal point processes: Embedding
        event history to vector. In Proceedings of the 22nd ACM SIGKDD International
        Conference on Knowledge Discovery and Data Mining (pp. 1555-1564). ACM.

    Parameters
    ----------
    context_interval_length
        The length of intervals (in continuous time) that the estimator will be
        trained with.
    prediction_interval_length
        The length of the interval (in continuous time) that the estimator will
        predict at prediction time.
    num_marks
        The number of marks (disctinct processes), i.e., the cardinality of the
        mark set.
    embedding_dim
        The dimension of vector embeddings for marks (used as input to the LSTM).
    trainer
        :code:`gluonts.trainer.Trainer` object which will be used to train the
        estimator. Note that :code:`Trainer(hybridize=False)` must be set as
        :code:`RMTPPEstimator` currently does not support hybridization.
    num_hidden_dimensions
        Number of hidden units in the (single) hidden layer of the LSTM.
    num_parallel_samples
        The number of samples returned by the :code:`Predictor` learned.
    num_training_instances
        The number of training instances to be sampled from each entry in the
        data set provided during training.
    freq
        Similar to the :code:`freq` of discrete-time models, specifies the time
        unit by which interarrival times are given.
    """

    @validated()
    def __init__(
        self,
        prediction_interval_length: float,
        context_interval_length: float,
        num_marks: int,
        embedding_dim: int = 5,
        trainer: Trainer = Trainer(hybridize=False),
        num_hidden_dimensions: int = 10,
        num_parallel_samples: int = 100,
        num_training_instances: int = 100,
        freq: str = "H",
    ) -> None:
        assert (
            not trainer.hybridize
        ), "RMTPP currently only supports the non-hybridized training"

        super().__init__(trainer=trainer)

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
        self.embedding_dim = embedding_dim
        self.num_parallel_samples = num_parallel_samples
        self.num_training_instances = num_training_instances
        self.freq = freq

    def create_training_network(self) -> HybridBlock:
        return RMTPPTrainingNetwork(
            num_marks=self.num_marks,
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
                    {"past_target": "target", "past_valid_length": "valid_length"}
                ),
            ]
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        # trained_params = trained_network.collect_params()
        #
        # del trained_network.collect_params().get("decay_bias").init

        prediction_network = RMTPPPredictionNetwork(
            num_marks=self.num_marks,
            prediction_interval_length=self.prediction_interval_length,
            interval_length=self.context_interval_length,
            embedding_dim=self.embedding_dim,
            num_hidden_dimensions=self.num_hidden_dimensions,
            params=trained_network.collect_params(),
            num_parallel_samples=self.num_parallel_samples,
        )

        return PointProcessGluonPredictor(
            input_names=["target", "valid_length"],
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
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
        **kwargs,
    ) -> TrainOutput:
        return super().train_model(
            training_data,
            validation_data,
            num_workers,
            num_prefetch,
            batchify_fn=partial(batchify, variable_length=True),
            **kwargs,
        )
