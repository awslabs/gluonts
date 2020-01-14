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
from gluonts.distribution import DistributionOutput, StudentTOutput
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
    VALID_N_BEATS_STACK_TYPES
)


class NBEATSNetworkEstimator(GluonEstimator):
    """
    An Estimator based on a single NBEATS network. The actual NBEATS model
    is an ensemble of NBEATS networks.

    Parameters
    ----------
    freq
        Time time granularity of the data
    prediction_length
        Length of the prediction horizon
    context_length
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    trainer
        Trainer object to be used (default: Trainer())
    num_stacks:
        The number of stacks the network should contain.
    widths:
        Widths of the fully connected layers with ReLu activation.
        A list of ints of length 1 or 'num_stacks'.
    blocks:
        The number of blocks blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
    block_layers:
        Number of fully connected layers with ReLu activation per block.
        A list of ints of length 1 or 'num_stacks'.
    sharing:
        Whether the weights are shared with the other blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
    expansion_coefficient_lengths:
        The number of the expansion coefficients.
        If type is "T" (trend), then it corresponds to the degree of the polynomial.
        A list of ints of length 1 or 'num_stacks'.
    stack_types:
        One of the following values: "G" (generic), "S" (seasonal) or "T" (trend).
        A list of strings of length 1 or 'num_stacks'.
    kwargs:
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
            num_stacks: Optional[int] = 30,  # 2
            widths: List[int] = None,  # [512] or [256, 2048]
            blocks: List[int] = None,  # [1] or [3]
            block_layers: List[int] = None,  # [4]
            expansion_coefficient_lengths: List[int] = None,  # [3] or [2, 8]
            sharing: List[bool] = None,  # [False] or [True]
            stack_types: List[str] = None,  # ["G"] or ["T", "S"]
            **kwargs
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

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        # num_stacks has to be handles separately because other arguments have to match its lengths
        self.num_stacks = num_stacks

        self.widths = self._validate_nbeats_argument(
            argument_value=widths, argument_name="widths", default_value=[512],
            validation_condition=lambda val: val > 0, invalidation_message="Values of 'widths' should be > 0"
        )
        self.blocks = self._validate_nbeats_argument(
            argument_value=blocks, argument_name="blocks", default_value=[1], validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'blocks' should be > 0"
        )
        self.block_layers = self._validate_nbeats_argument(
            argument_value=block_layers, argument_name="block_layers", default_value=[4], validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'block_layers' should be > 0"
        )
        self.sharing = self._validate_nbeats_argument(
            argument_value=sharing, argument_name="sharing", default_value=[False], validation_condition=lambda val: True,
            invalidation_message=""
        )
        self.expansion_coefficient_lengths = self._validate_nbeats_argument(
            argument_value=expansion_coefficient_lengths, argument_name="expansion_coefficient_lengths", default_value=[3], validation_condition=lambda val: val > 0,
            invalidation_message="Values of 'expansion_coefficient_lengths' should be > 0"
        )
        self.stack_types = self._validate_nbeats_argument(
            argument_value=stack_types, argument_name="stack_types", default_value=["G"], validation_condition=lambda val: val in VALID_N_BEATS_STACK_TYPES,
            invalidation_message=f"Values of 'stack_types' should be one of {VALID_N_BEATS_STACK_TYPES}"
        )
        self.prediction_length = prediction_length
        self.context_length = context_length

    def _validate_nbeats_argument(self, argument_value, argument_name, default_value, validation_condition,
                                  invalidation_message):
        # set default value if applicable
        new_value = default_value if argument_value is None else argument_name

        # check whether dimension of argument matches num_stack dimension
        assert len(new_value) == 1 or len(new_value) == self.num_stacks, (
            f"Invalid lengths of argument {new_value}: {len(new_value)}. Argument must have "
            f"length 1 or {self.num_stacks} "
        )

        # check validity of actual values
        assert all([validation_condition(val) for val in new_value]), invalidation_message

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

        )

    # we now define how the prediction happens given that we are provided a
    # training network.
    def create_predictor(
            self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = NBEATSPredictionNetwork(

            params=trained_network.collect_params()
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
