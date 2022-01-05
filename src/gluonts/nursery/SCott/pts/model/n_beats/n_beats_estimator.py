from typing import List, Optional

import torch
import torch.nn as nn

from pts import Trainer
from pts.dataset import FieldName
from pts.model import PTSEstimator, Predictor, PTSPredictor, copy_parameters
from pts.transform import (
    InstanceSplitter,
    Transformation,
    Chain,
    RemoveFields,
    ExpectedNumInstanceSampler,
)
from .n_beats_network import (
    NBEATSPredictionNetwork,
    NBEATSTrainingNetwork,
    VALID_N_BEATS_STACK_TYPES,
)
from pts.transform.sampler import CustomUniformSampler


class NBEATSEstimator(PTSEstimator):
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
        super().__init__(trainer=trainer, **kwargs)

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

    # Here we do only a simple operation to convert the input data to a form
    # that can be digested by our model by only splitting the target in two, a
    # conditioning part and a to-predict part, for each training example.
    def create_transformation(self, is_full_batch=False) -> Transformation:
        return Chain(
            [
                RemoveFields(
                    field_names=[
                        FieldName.FEAT_STATIC_REAL,
                        FieldName.FEAT_DYNAMIC_REAL,
                        FieldName.FEAT_DYNAMIC_CAT,
                    ]
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    # train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    train_sampler=CustomUniformSampler(),
                    past_length=self.context_length,
                    is_full_batch=is_full_batch,
                    future_length=self.prediction_length,
                    time_series_fields=[],
                ),
            ]
        )

    def create_training_network(
        self, device: torch.device
    ) -> NBEATSTrainingNetwork:
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
        ).to(device)

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
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
        ).to(device)

        copy_parameters(trained_network, prediction_network)

        return PTSPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
