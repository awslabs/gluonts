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
from typing import Callable, Optional

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.itertools import Cyclic
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.distribution import DistributionOutput, NegativeBinomialOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    RenameFields,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)
from gluonts.transform.convert import SwapAxes, ToIntervalSizeFormat
from gluonts.transform.feature import CountTrailingZeros

from ._network import DeepRenewalPredictionNetwork, DeepRenewalTrainingNetwork
from ._predictor import DeepRenewalProcessPredictor
from ._transform import AddAxisLength


class DeepRenewalProcessEstimator(GluonEstimator):
    """
    Implements a deep renewal process estimator designed to forecast
    intermittent time series sampled in discrete time, as described in.

    [TWJ19]_.

    In short, instead of viewing sparse time series as a univariate stochastic
    process, this estimator transforms a sparse time series [0, 0, 0, 3, 0, 0,
    7] to an interval-size format,[(4, 3), (3, 7)] where each ordered pair
    marks the time since the last positive time step(interval) and the value
    of the positive time step (size). Then, probabilistic prediction is
    performed on this transformed time series, as is customary in the
    intermittent demand literature, e.g., Croston's method.

    This construction is a self-modulated marked renewal process in discrete
    time as one assumes the (conditional) distribution of intervals are
    identical.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon
    context_length
        The number of time steps the model will condition on
    num_cells
        Number of hidden units used in the RNN cell (LSTM) and dense layer for
        projection to distribution arguments
    num_layers
        Number of layers in the LSTM
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    trainer
        Trainer object to be used (default: Trainer())
    interval_distr_output
        Distribution output object for the intervals. Must be a distribution
        with support on positive integers, where the first argument
        corresponds to the(conditional) mean.
    size_distr_output
        Distribution output object for the demand sizes. Must be a distribution
        with support on positive integers, where the first argument
        corresponds to the(conditional) mean.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    batch_size
        The size of the batches to be used training and prediction.
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference. This is a model optimization that does not affect the
        accuracy (default: 100)
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        num_cells: int,
        num_layers: int,
        dropout_rate: float = 0.1,
        interval_distr_output: DistributionOutput = NegativeBinomialOutput(),
        size_distr_output: DistributionOutput = NegativeBinomialOutput(),
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        trainer: Trainer = Trainer(hybridize=False),
        batch_size: int = 32,
        num_parallel_samples: int = 100,
        **kwargs,
    ):
        super().__init__(trainer=trainer, batch_size=batch_size, **kwargs)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.interval_distr_output = interval_distr_output
        self.size_distr_output = size_distr_output
        self.num_parallel_samples = num_parallel_samples

        self.train_sampler = (
            train_sampler
            if train_sampler is not None
            else ExpectedNumInstanceSampler(
                num_instances=5, min_future=prediction_length
            )
        )
        self.validation_sampler = (
            validation_sampler
            if validation_sampler is not None
            else ValidationSplitSampler(min_future=prediction_length)
        )

    def create_transformation(self) -> Transformation:
        return AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
        )

    @staticmethod
    def _create_post_split_transform():
        return Chain(
            [
                CountTrailingZeros(
                    new_field="time_remaining",
                    target_field="past_target",
                    as_array=True,
                ),
                ToIntervalSizeFormat(
                    target_field="past_target", discard_first=True
                ),
                RenameFields({"future_target": "sparse_future"}),
                AsNumpyArray(field="past_target", expected_ndim=2),
                SwapAxes(input_fields=["past_target"], axes=(0, 1)),
                AddAxisLength(target_field="past_target", axis=0),
            ]
        )

    def _stack_fn(self) -> Callable:
        return partial(
            batchify,
            ctx=self.trainer.ctx,
            dtype=self.dtype,
            variable_length=True,
            is_right_pad=False,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        train_transform = (
            self._create_instance_splitter("training")
            + self._create_post_split_transform()
            + SelectFields(["past_target", "valid_length"])
        )
        return TrainDataLoader(
            train_transform.apply(Cyclic(data)),
            batch_size=self.batch_size,
            stack_fn=self._stack_fn(),
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        validation_transform = (
            self._create_instance_splitter("validation")
            + self._create_post_split_transform()
            + SelectFields(["past_target", "valid_length"])
        )
        return ValidationDataLoader(
            validation_transform.apply(data),
            batch_size=self.batch_size,
            stack_fn=self._stack_fn(),
        )

    def create_training_network(self) -> DeepRenewalTrainingNetwork:
        return DeepRenewalTrainingNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_cells=self.num_cells,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            interval_distr_output=self.interval_distr_output,
            size_distr_output=self.size_distr_output,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_transform = (
            self._create_instance_splitter("test")
            + self._create_post_split_transform()
        )

        prediction_network = DeepRenewalPredictionNetwork(
            num_parallel_samples=self.num_parallel_samples,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_cells=self.num_cells,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            interval_distr_output=self.interval_distr_output,
            size_distr_output=self.size_distr_output,
        )

        copy_parameters(trained_network, prediction_network)

        return DeepRenewalProcessPredictor(
            input_transform=transformation + prediction_transform,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            input_names=["past_target", "time_remaining"],
        )
