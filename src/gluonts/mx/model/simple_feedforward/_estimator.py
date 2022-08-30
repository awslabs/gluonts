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
from typing import List

from mxnet.gluon import HybridBlock
from pydantic import Field

from gluonts.core import serde
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.env import env
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.mx.batchify import batchify
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import get_hybrid_forward_input_names
from gluonts.itertools import maybe_len
from gluonts.transform import (
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)
from gluonts.transform.feature import (
    DummyValueImputation,
    MissingValueImputation,
)

from ._network import (
    SimpleFeedForwardDistributionNetwork,
    SimpleFeedForwardSamplingNetwork,
    SimpleFeedForwardTrainingNetwork,
)


@serde.dataclass
class SimpleFeedForwardEstimator(GluonEstimator):
    """
    SimpleFeedForwardEstimator shows how to build a simple MLP model predicting
    the next target time-steps given the previous ones.

    Given that we want to define a gluon model trainable by SGD, we inherit the
    parent class `GluonEstimator` that handles most of the logic for fitting a
    neural-network.

    We thus only have to define:

    1. How the data is transformed before being fed to our model::

        def create_transformation(self) -> Transformation

    2. How the training happens::

        def create_training_network(self) -> HybridBlock

    3. how the predictions can be made for a batch given a trained network::

        def create_predictor(
             self,
             transformation: Transformation,
             trained_net: HybridBlock,
        ) -> Predictor


    Parameters
    ----------
    prediction_length
        Length of the prediction horizon
    trainer
        Trainer object to be used (default: Trainer())
    num_hidden_dimensions
        Number of hidden nodes in each layer (default: [40, 40])
    context_length
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    distr_output
        Distribution to fit (default: StudentTOutput())
    batch_normalization
        Whether to use batch normalization (default: False)
    mean_scaling
        Scale the network input by the data mean and the network output by
        its inverse (default: True)
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference. This is a model optimization that does not affect the
        accuracy (default: 100)
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    batch_size
        The size of the batches to be used training and prediction.
    """

    prediction_length: int = Field(..., gt=0)
    sampling: bool = True
    trainer: Trainer = Trainer()
    num_hidden_dimensions: List[int] = Field(None, gt=0)
    context_length: int = Field(None, gt=0)
    distr_output: DistributionOutput = StudentTOutput()
    imputation_method: MissingValueImputation = Field(None)
    batch_normalization: bool = False
    mean_scaling: bool = True
    num_parallel_samples: int = Field(100, gt=0)
    train_sampler: InstanceSampler = Field(None)
    validation_sampler: InstanceSampler = Field(None)
    batch_size: int = 32

    def __post_init_post_parse__(self):
        """
        Defines an estimator.

        All parameters should be serializable.
        """

        super().__init__(trainer=self.trainer, batch_size=self.batch_size)

        if self.num_hidden_dimensions is None:
            self.num_hidden_dimensions = list([40, 40])

        if self.context_length is None:
            self.context_length = self.prediction_length

        if self.imputation_method is None:
            self.imputation_method = DummyValueImputation(
                self.distr_output.value_in_support
            )

        if self.train_sampler is None:
            self.train_sampler = ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.prediction_length
            )

        if self.validation_sampler is None:
            self.validation_sampler = ValidationSplitSampler(
                min_future=self.prediction_length
            )

    # Here we do only a simple operation to convert the input data to a form
    # that can be digested by our model by only splitting the target in two, a
    # conditioning part and a to-predict part, for each training example.
    # For a more complex transformation example, see the `gluonts.model.deepar`
    # transformation that includes time features, age feature, observed values
    # indicator, ...
    def create_transformation(self) -> Transformation:
        return SelectFields(
            [
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
            ],
            allow_missing=True,
        ) + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
            dtype=self.dtype,
            imputation_method=self.imputation_method,
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
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            SimpleFeedForwardTrainingNetwork
        )
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("training")
        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            SimpleFeedForwardTrainingNetwork
        )
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    # defines the network, we get to see one batch to initialize it. the
    # network should return at least one tensor that is used as a loss to
    # minimize in the training loop. several tensors can be returned for
    # instance for analysis, see DeepARTrainingNetwork for an example.
    def create_training_network(self) -> HybridBlock:
        return SimpleFeedForwardTrainingNetwork(
            num_hidden_dimensions=self.num_hidden_dimensions,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            batch_normalization=self.batch_normalization,
            mean_scaling=self.mean_scaling,
        )

    # we now define how the prediction happens given that we are provided a
    # training network.
    def create_predictor(self, transformation, trained_network):
        prediction_splitter = self._create_instance_splitter("test")

        if self.sampling is True:
            prediction_network = SimpleFeedForwardSamplingNetwork(
                num_hidden_dimensions=self.num_hidden_dimensions,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                distr_output=self.distr_output,
                batch_normalization=self.batch_normalization,
                mean_scaling=self.mean_scaling,
                params=trained_network.collect_params(),
                num_parallel_samples=self.num_parallel_samples,
            )

            return RepresentableBlockPredictor(
                input_transform=transformation + prediction_splitter,
                prediction_net=prediction_network,
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                ctx=self.trainer.ctx,
            )

        else:
            prediction_network = SimpleFeedForwardDistributionNetwork(
                num_hidden_dimensions=self.num_hidden_dimensions,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                distr_output=self.distr_output,
                batch_normalization=self.batch_normalization,
                mean_scaling=self.mean_scaling,
                params=trained_network.collect_params(),
                num_parallel_samples=self.num_parallel_samples,
            )
            return RepresentableBlockPredictor(
                input_transform=transformation + prediction_splitter,
                prediction_net=prediction_network,
                batch_size=self.batch_size,
                forecast_generator=DistributionForecastGenerator(
                    self.distr_output
                ),
                prediction_length=self.prediction_length,
                ctx=self.trainer.ctx,
            )
