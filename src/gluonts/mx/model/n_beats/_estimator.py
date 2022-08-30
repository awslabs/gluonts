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
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
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

from ._network import (
    VALID_LOSS_FUNCTIONS,
    VALID_N_BEATS_STACK_TYPES,
    NBEATSPredictionNetwork,
    NBEATSTrainingNetwork,
)


@serde.dataclass
class NBEATSEstimator(GluonEstimator):
    """
    An Estimator based on a single (!) NBEATS Network (approximately) as
    described in the paper:  https://arxiv.org/abs/1905.10437. The actual
    NBEATS model is an ensemble of NBEATS Networks, and is implemented by the
    "NBEATSEnsembleEstimator".

    Noteworthy differences in this implementation compared to the paper:
    * The parameter L_H is not implemented; we sample training sequences
    using the default method in GluonTS using the "InstanceSplitter".

    Parameters
    ----------
    freq
        Time time granularity of the data
    prediction_length
        Length of the prediction. Also known as 'horizon'.
    context_length
        Number of time units that condition the predictions
        Also known as 'lookback period'.
        Default is 2 * prediction_length.
    trainer
        Trainer object to be used (default: Trainer())
    num_stacks:
        The number of stacks the network should contain.
        Default and recommended value for generic mode: 30
        Recommended value for interpretable mode: 2
    num_blocks
        The number of blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [1]
        Recommended value for interpretable mode: [3]
    block_layers
        Number of fully connected layers with ReLu activation per block.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [4]
        Recommended value for interpretable mode: [4]
    widths
        Widths of the fully connected layers with ReLu activation in the
        blocks. A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [512]
        Recommended value for interpretable mode: [256, 2048]
    sharing
        Whether the weights are shared with the other blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [False]
        Recommended value for interpretable mode: [True]
    expansion_coefficient_lengths
        If the type is "G" (generic), then the length of the expansion
        coefficient. If type is "T" (trend), then it corresponds to the degree
        of the polynomial. If the type is "S" (seasonal) then its not used. A
        list of ints of length 1 or 'num_stacks'. Default value for generic
        mode: [32] Recommended value for interpretable mode: [3]
    stack_types
        One of the following values: "G" (generic), "S" (seasonal) or "T"
        (trend).
        A list of strings of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: ["G"]
        Recommended value for interpretable mode: ["T","S"]
    loss_function
        The loss function (also known as metric) to use for training the
        network. Unlike other models in GluonTS this network does not use a
        distribution.
        One of the following: "sMAPE", "MASE" or "MAPE".
        The default value is "MAPE".
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    batch_size
        The size of the batches to be used training and prediction.
    scale
        if True scales the input observations by the mean
    kwargs
        Arguments passed to 'GluonEstimator'.
    """

    freq: str
    prediction_length: int = Field(..., gt=0)
    context_length: int = Field(None, gt=0)
    trainer: Trainer = Trainer()
    num_stacks: int = Field(30, gt=0)
    widths: List[int] = Field(None)
    num_blocks: List[int] = Field(None)
    num_block_layers: List[int] = Field(None)
    expansion_coefficient_lengths: List[int] = Field(None)
    sharing: List[bool] = Field(None)
    stack_types: List[str] = Field(None)
    loss_function: str = Field(default="MAPE")
    train_sampler: InstanceSampler = Field(None)
    validation_sampler: InstanceSampler = Field(None)
    batch_size: int = 32
    scale: bool = False

    def __post_init_post_parse__(self):
        super().__init__(batch_size=self.batch_size, trainer=self.trainer)
        assert (
            self.loss_function is None
            or self.loss_function in VALID_LOSS_FUNCTIONS
        ), (
            "The loss function has to be one of the following:"
            f" {VALID_LOSS_FUNCTIONS}."
        )

        if self.context_length is None:
            self.context_length = 2 * self.prediction_length

        self.widths = self._validate_nbeats_argument(
            argument_value=self.widths,
            argument_name="widths",
            default_value=[512],
            validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'widths' should be > 0",
        )
        self.num_blocks = self._validate_nbeats_argument(
            argument_value=self.num_blocks,
            argument_name="num_blocks",
            default_value=[1],
            validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'num_blocks' should be > 0",
        )
        self.num_block_layers = self._validate_nbeats_argument(
            argument_value=self.num_block_layers,
            argument_name="num_block_layers",
            default_value=[4],
            validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'block_layers' should be > 0",
        )
        self.sharing = self._validate_nbeats_argument(
            argument_value=self.sharing,
            argument_name="sharing",
            default_value=[False],
            validation_condition=lambda val: True,
            invalidation_message="",
        )
        self.expansion_coefficient_lengths = self._validate_nbeats_argument(
            argument_value=self.expansion_coefficient_lengths,
            argument_name="expansion_coefficient_lengths",
            default_value=[32],
            validation_condition=lambda val: val > 0,
            invalidation_message=(
                "Values of 'expansion_coefficient_lengths' should be > 0"
            ),
        )
        self.stack_types = self._validate_nbeats_argument(
            argument_value=self.stack_types,
            argument_name="stack_types",
            default_value=["G"],
            validation_condition=lambda val: val in VALID_N_BEATS_STACK_TYPES,
            invalidation_message=(
                "Values of 'stack_types' should be one of"
                f" {VALID_N_BEATS_STACK_TYPES}"
            ),
        )
        if self.train_sampler is None:
            self.train_sampler = ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.prediction_length
            )

        if self.validation_sampler is None:
            self.validation_sampler = ValidationSplitSampler(
                min_future=self.prediction_length
            )

    def _validate_nbeats_argument(
        self,
        argument_value,
        argument_name,
        default_value,
        validation_condition,
        invalidation_message,
    ):
        # set default value if applicable
        new_value = (
            argument_value if argument_value is not None else default_value
        )

        # check whether dimension of argument matches num_stack dimension
        assert len(new_value) == 1 or len(new_value) == self.num_stacks, (
            f"Invalid lengths of argument {argument_name}: {len(new_value)}."
            f" Argument must have length 1 or {self.num_stacks} "
        )

        # check validity of actual values
        assert all(
            [validation_condition(val) for val in new_value]
        ), invalidation_message

        # make length of arguments consistent
        if len(new_value) == 1:
            return new_value * self.num_stacks
        else:
            return new_value

    def create_transformation(self) -> Transformation:
        return AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
            dtype=self.dtype,
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
        input_names = get_hybrid_forward_input_names(NBEATSTrainingNetwork)
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
        input_names = get_hybrid_forward_input_names(NBEATSTrainingNetwork)
        with env._let(max_idle_transforms=maybe_len(data) or 0):
            instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> HybridBlock:
        return NBEATSTrainingNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_stacks=self.num_stacks,
            widths=self.widths,
            num_blocks=self.num_blocks,
            num_block_layers=self.num_block_layers,
            expansion_coefficient_lengths=self.expansion_coefficient_lengths,
            sharing=self.sharing,
            stack_types=self.stack_types,
            loss_function=self.loss_function,
            freq=self.freq,
            scale=self.scale,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = NBEATSPredictionNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_stacks=self.num_stacks,
            widths=self.widths,
            num_blocks=self.num_blocks,
            num_block_layers=self.num_block_layers,
            expansion_coefficient_lengths=self.expansion_coefficient_lengths,
            sharing=self.sharing,
            stack_types=self.stack_types,
            params=trained_network.collect_params(),
            scale=self.scale,
        )

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
