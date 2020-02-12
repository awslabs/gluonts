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

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.trainer import Trainer
from gluonts.transform import (
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    Transformation,
)

# Relative imports
from ._network import (
    NBEATSPredictionNetwork,
    NBEATSTrainingNetwork,
    VALID_N_BEATS_STACK_TYPES,
    VALID_LOSS_FUNCTIONS,
)


class NBEATSEstimator(GluonEstimator):
    """
    An Estimator based on a single (!) NBEATS Network (approximately) as described
    in the paper:  https://arxiv.org/abs/1905.10437.
    The actual NBEATS model is an ensemble of NBEATS Networks, and is implemented by
    the "NBEATSEnsembleEstimator".

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
        Widths of the fully connected layers with ReLu activation in the blocks.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [512]
        Recommended value for interpretable mode: [256, 2048]
    sharing
        Whether the weights are shared with the other blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [False]
        Recommended value for interpretable mode: [True]
    expansion_coefficient_lengths
        If the type is "G" (generic), then the length of the expansion coefficient.
        If type is "T" (trend), then it corresponds to the degree of the polynomial.
        If the type is "S" (seasonal) then its not used.
        A list of ints of length 1 or 'num_stacks'.
        Default value for generic mode: [32]
        Recommended value for interpretable mode: [3]
    stack_types
        One of the following values: "G" (generic), "S" (seasonal) or "T" (trend).
        A list of strings of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: ["G"]
        Recommended value for interpretable mode: ["T","S"]
    loss_function
        The loss funtion (also known as metric) to use for training the network.
        Unlike other models in GluonTS this network does not use a distribution.
        One of the following: "sMAPE", "MASE" or "MAPE".
        The default value is "MAPE".
    kwargs
        Arguments passed to 'GluonEstimator'.
    """

    # The validated() decorator makes sure that parameters are checked by
    # Pydantic and allows to serialize/print models. Note that all parameters
    # have defaults except for `freq` and `prediction_length`. which is
    # recommended in GluonTS to allow to compare models easily.
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        num_stacks: int = 30,
        widths: Optional[List[int]] = None,
        num_blocks: Optional[List[int]] = None,
        num_block_layers: Optional[List[int]] = None,
        expansion_coefficient_lengths: Optional[List[int]] = None,
        sharing: Optional[List[bool]] = None,
        stack_types: Optional[List[str]] = None,
        loss_function: Optional[str] = "MAPE",
        **kwargs,
    ) -> None:
        """
        Defines an estimator. All parameters should be serializable.
        """
        super().__init__(trainer=trainer, **kwargs)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert (
            num_stacks is None or num_stacks > 0
        ), "The value of `num_stacks` should be > 0"
        assert (
            loss_function is None or loss_function in VALID_LOSS_FUNCTIONS
        ), f"The loss function has to be one of the following: {VALID_LOSS_FUNCTIONS}."

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = (
            context_length
            if context_length is not None
            else 2 * prediction_length
        )
        # num_stacks has to be handled separately because other arguments have to match its length
        self.num_stacks = num_stacks
        self.loss_function = loss_function

        self.widths = self._validate_nbeats_argument(
            argument_value=widths,
            argument_name="widths",
            default_value=[512],
            validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'widths' should be > 0",
        )
        self.num_blocks = self._validate_nbeats_argument(
            argument_value=num_blocks,
            argument_name="num_blocks",
            default_value=[1],
            validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'num_blocks' should be > 0",
        )
        self.num_block_layers = self._validate_nbeats_argument(
            argument_value=num_block_layers,
            argument_name="num_block_layers",
            default_value=[4],
            validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'block_layers' should be > 0",
        )
        self.sharing = self._validate_nbeats_argument(
            argument_value=sharing,
            argument_name="sharing",
            default_value=[False],
            validation_condition=lambda val: True,
            invalidation_message="",
        )
        self.expansion_coefficient_lengths = self._validate_nbeats_argument(
            argument_value=expansion_coefficient_lengths,
            argument_name="expansion_coefficient_lengths",
            default_value=[32],
            validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'expansion_coefficient_lengths' should be > 0",
        )
        self.stack_types = self._validate_nbeats_argument(
            argument_value=stack_types,
            argument_name="stack_types",
            default_value=["G"],
            validation_condition=lambda val: val in VALID_N_BEATS_STACK_TYPES,
            invalidation_message=f"Values of 'stack_types' should be one of {VALID_N_BEATS_STACK_TYPES}",
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
            f"Invalid lengths of argument {argument_name}: {len(new_value)}. Argument must have "
            f"length 1 or {self.num_stacks} "
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

    # here we do only a simple operation to convert the input data to a form
    # that can be digested by our model by only splitting the target in two, a
    # conditioning part and a to-predict part, for each training example.
    # for a more complex transformation example, see the `gluonts.model.deepar`
    # transformation that includes time features, age feature, observed values
    # indicator, ...
    def create_transformation(self) -> Transformation:
        return Chain(
            [
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=[],
                )
            ]
        )

    # defines the network, we get to see one batch to initialize it.
    # the network should return at least one tensor that is used as a loss to minimize in the training loop.
    # several tensors can be returned for instance for analysis, see DeepARTrainingNetwork for an example.
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
        )

    # we now define how the prediction happens given that we are provided a
    # training network.
    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
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
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
