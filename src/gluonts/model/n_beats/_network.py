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
from typing import List

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.block.scaler import MeanScaler, NOPScaler
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from typing import List

VALID_N_BEATS_STACK_TYPES = "G", "S", "T"

# TODO: rework seasonality and trend model


def linear_space(F, backcast_length, forecast_length, fwd_looking=True):
    ls = (
        F.arange(-float(backcast_length), float(forecast_length), 1)
        / backcast_length
    )
    if fwd_looking:
        ls = ls[backcast_length:]
    else:
        ls = ls[:backcast_length]
    return ls


def seasonality_model(
    F,
    thetas,
    expansion_coefficient_length,
    context_length,
    prediction_length,
    is_forecast,
):
    p = expansion_coefficient_length
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(
        F, context_length, prediction_length, fwd_looking=is_forecast
    )
    s1 = F.stack(*[F.cos(2 * np.pi * i * t) for i in range(p1)], axis=0)
    s2 = F.stack(*[F.sin(2 * np.pi * i * t) for i in range(p2)], axis=0)
    S = F.concat(s1, s2, dim=0)
    S = F.cast(S, np.float32)
    return F.dot(thetas, S)


def trend_model(
    F,
    thetas,
    expansion_coefficient_length,
    context_length,
    prediction_length,
    is_forecast,
):
    p = expansion_coefficient_length
    t = linear_space(
        F, context_length, prediction_length, fwd_looking=is_forecast
    )
    T = F.transpose(F.stack(*[t ** i for i in range(p)], axis=0))
    T = F.cast(T, np.float32)
    return F.dot(thetas, F.transpose(T))


class NBEATSBlock(mx.gluon.HybridBlock):
    """
    The NBEATS Block as described in the paper: https://arxiv.org/abs/1905.10437.
    Its configurable to any of the specific block
    types through the block_type parameter.

    Parameters
    ----------
    width:
        Width of the fully connected layers with ReLu activation.
    block_layers:
        Number of fully connected layers with ReLu activation.
    expansion_coefficient_length:
        The length of the expansion coefficient.
        If type is "T" (trend), then it corresponds to the degree of the polynomial.
    block_type:
        Either "G" (generic), "S" (seasonal) or "T" (trend).
    prediction_length:
        Also known as the 'prediction_length'.
    context_length:
        Also known as the 'context_length'.
    has_backcast:
        Only the last block of the network doesnt.
    kwargs:
        Arguments passed to 'HybridBlock'.
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        width: int,
        block_layers: int,
        expansion_coefficient_length: int,
        prediction_length: int,
        context_length: int,
        block_type: str,
        has_backcast: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.width = width
        self.block_layers = block_layers
        self.expansion_coefficient_length = expansion_coefficient_length
        self.block_type = block_type
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.has_backcast = has_backcast

        with self.name_scope():
            self.fc_stack = mx.gluon.nn.HybridSequential()
            for i in range(block_layers):
                self.fc_stack.add(
                    mx.gluon.nn.Dense(units=width, activation="relu")
                )
            if has_backcast:
                self.theta_backcast = mx.gluon.nn.Dense(
                    units=expansion_coefficient_length  # linear activation:
                )
            self.theta_forecast = mx.gluon.nn.Dense(
                units=expansion_coefficient_length  # linear activation:
            )
            if block_type == "G":
                if has_backcast:
                    self.backcast = mx.gluon.nn.Dense(
                        units=context_length  # linear activation:
                    )
                self.forecast = mx.gluon.nn.Dense(
                    units=prediction_length  # linear activation:
                )
            elif block_type == "S":
                if has_backcast:
                    self.backcast = mx.gluon.nn.HybridLambda(
                        lambda F, thetas: seasonality_model(
                            F,
                            thetas,
                            expansion_coefficient_length,
                            context_length,
                            prediction_length,
                            is_forecast=True,
                        )
                    )
                self.forecast = mx.gluon.nn.HybridLambda(
                    lambda F, thetas: seasonality_model(
                        F,
                        thetas,
                        expansion_coefficient_length,
                        context_length,
                        prediction_length,
                        is_forecast=False,
                    )
                )
            else:  # "T"
                if has_backcast:
                    self.backcast = mx.gluon.nn.HybridLambda(
                        lambda F, thetas: seasonality_model(
                            F,
                            thetas,
                            expansion_coefficient_length,
                            context_length,
                            prediction_length,
                            is_forecast=True,
                        )
                    )
                self.forecast = mx.gluon.nn.HybridLambda(
                    lambda F, thetas: seasonality_model(
                        F,
                        thetas,
                        expansion_coefficient_length,
                        context_length,
                        prediction_length,
                        is_forecast=False,
                    )
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.fc_stack(x)
        theta_f = self.theta_forecast(x)
        forecast = self.forecast(theta_f)

        if self.has_backcast:
            theta_b = self.theta_backcast(x)
            backcast = self.backcast(theta_b)
            return backcast, forecast

        return forecast


class NBEATSNetwork(mx.gluon.HybridBlock):
    """
    The NBEATS Network as described in the paper: https://arxiv.org/abs/1905.10437.
    This does not constitute the whole NBEATS model, which is en ensemble model
    comprised of a multitude of NBEATS Networks.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon
    context_length
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    num_stacks:
        The number of stacks the network should contain.
        Default and recommended value for generic mode: 30
        Recommended value for interpretable mode: 2
    widths:
        Widths of the fully connected layers with ReLu activation.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [512]
        Recommended value for interpretable mode: [256, 2048]
    blocks:
        The number of blocks blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [1]
        Recommended value for interpretable mode: [3]
    block_layers:
        Number of fully connected layers with ReLu activation per block.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [4]
        Recommended value for interpretable mode: [4]
    sharing:
        Whether the weights are shared with the other blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [False]
        Recommended value for interpretable mode: [True]
    expansion_coefficient_lengths:
        The number of the expansion coefficients.
        If type is "T" (trend), then it corresponds to the degree of the polynomial.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [3]
        Recommended value for interpretable mode: [2,8]
    stack_types:
        One of the following values: "G" (generic), "S" (seasonal) or "T" (trend).
        A list of strings of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: ["G"]
        Recommended value for interpretable mode: ["T","S"]
    kwargs:
        Arguments passed to 'HybridBlock'.
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        num_stacks: int,
        widths: List[int],
        blocks: List[int],
        block_layers: List[int],
        expansion_coefficient_lengths: List[int],
        sharing: List[bool],
        stack_types: List[str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_stacks = num_stacks
        self.widths = widths
        self.blocks = blocks
        self.block_layers = block_layers
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.stack_types = stack_types
        self.prediction_length = prediction_length
        self.context_length = context_length

        with self.name_scope():
            self.net_blocks: List[NBEATSBlock] = []

            # connect all the blocks correctly
            for stack_id in range(num_stacks):
                for block_id in range(blocks[stack_id]):
                    # in case sharing is enabled for the stack
                    params = (
                        self.net_blocks[-1].collect_params()
                        if (block_id > 0 and sharing[stack_id])
                        else None
                    )
                    # only last one does not have backcast
                    has_backcast = not (
                        stack_id == num_stacks - 1
                        and block_id == blocks[num_stacks - 1] - 1
                    )
                    net_block = NBEATSBlock(
                        width=self.widths[stack_id],
                        block_layers=self.block_layers[stack_id],
                        expansion_coefficient_length=self.expansion_coefficient_lengths[
                            stack_id
                        ],
                        prediction_length=prediction_length,
                        context_length=context_length,
                        block_type=self.stack_types[stack_id],
                        has_backcast=has_backcast,
                        params=params,
                    )
                    self.net_blocks.append(net_block)
                    self.register_child(
                        net_block, f"block_{stack_id}_{block_id}"
                    )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, past_target: Tensor, future_target: Tensor):
        if len(self.net_blocks) == 1:  # if first block is also last block
            return self.net_blocks[0](past_target)
        else:
            backcast, forecast = self.net_blocks[0](past_target)
            backcast = past_target - backcast
            # connect regular blocks (all except last)
            for i in range(1, len(self.net_blocks) - 1):
                b, f = self.net_blocks[i](backcast)
                backcast = backcast - b
                forecast = forecast + f
            # connect last block
            return forecast + self.net_blocks[-1](backcast)


class NBEATSTrainingNetwork(NBEATSNetwork):
    @validated()
    def __init__(
        self,
        loss_function: mx.gluon.loss.Loss,
        *args,
        **kwargs,  # TODO: make loss_function serializable somehow
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, past_target: Tensor, future_target: Tensor
    ) -> Tensor:
        """

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        future_target
            Tensor with future observations.
            Shape: (batch_size, prediction_length, target_dim).

        Returns
        -------
        Tensor
            Loss tensor. Shape: (batch_size, ).
        """
        # future_target never used
        forecast = super().hybrid_forward(
            F, past_target=past_target, future_target=future_target
        )

        # (batch_size, context_length, target_dim)
        loss = self.loss_function(forecast, future_target)

        # (batch_size, )
        return loss


class NBEATSPredictionNetwork(NBEATSNetwork):
    @validated()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, past_target: Tensor, future_target: Tensor = None
    ) -> Tensor:
        """

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        future_target
            Not used.

        Returns
        -------
        Tensor
            Prediction sample. Shape: (samples, batch_size, prediction_length).
        """
        # future_target never used
        forecasts = super().hybrid_forward(
            F, past_target=past_target, future_target=past_target
        )
        # dimension collapsed previously because we only have one sample each:
        forecasts = F.expand_dims(forecasts, axis=1)

        # (batch_size, num_samples, prediction_length)
        return forecasts
