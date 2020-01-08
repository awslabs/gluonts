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
from gluonts.distribution import Distribution, DistributionOutput
from gluonts.model.common import Tensor
from typing import List, Optional, Tuple

VALID_N_BEATS_STACK_TYPES = "G", "S", "T"


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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert (
            block_type in VALID_N_BEATS_STACK_TYPES
        ), f"Invalid N-Beats stack type: {block_type}. Valid stack types: {VALID_N_BEATS_STACK_TYPES}."

        self.width = width
        self.block_layers = block_layers
        self.expansion_coefficient_length = expansion_coefficient_length
        self.block_type = block_type
        self.prediction_length = prediction_length
        self.context_length = context_length

        with self.name_scope():
            self.fc_stack = mx.gluon.nn.HybridSequential()
            for i in range(block_layers):
                self.fc_stack.add(
                    mx.gluon.nn.Dense(units=width, activation="relu")
                )
            self.theta_backcast = mx.gluon.nn.Dense(
                units=expansion_coefficient_length, activation="linear"
            )
            self.theta_forecast = mx.gluon.nn.Dense(
                units=expansion_coefficient_length, activation="linear"
            )
            if block_type == "G":
                self.backcast = mx.gluon.nn.Dense(
                    units=context_length, activation="linear"
                )
                self.forecast = mx.gluon.nn.Dense(
                    units=prediction_length, activation="linear"
                )
            if block_type == "S":
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
        theta_b = self.theta_backcast(x)
        theta_f = self.theta_forecast(x)
        backcast = self.backcast(theta_b)
        forecast = self.forecast(theta_f)
        return backcast, forecast


def _reformat_nbeats_network_argument(
    num_stacks: int, argument, argument_name: str
):
    assert len(argument) == 1 or len(argument) == num_stacks, (
        f"Invalid lengths of argument {argument_name}: {len(argument)}. Argument must have "
        f"length 1 or {num_stacks} "
    )
    if len(argument) == 1:
        return argument * num_stacks
    else:
        return argument


class NBEATSNetwork(mx.gluon.HybridBlock):
    """

    Parameters
    ----------
    prediction_length:
        Also known as the 'prediction_length'.
    context_length:
        Also known as the 'context_length'.
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
        Arguments passed to 'HybridBlock'.
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        num_stacks: int = 2,
        widths: List[int] = [256, 2048],
        blocks: List[int] = [3],
        block_layers: List[int] = [4],
        expansion_coefficient_lengths: List[int] = [2, 8],
        sharing: List[bool] = [True],
        stack_types: List[str] = ["T", "S"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_stacks = num_stacks
        self.widths = _reformat_nbeats_network_argument(
            num_stacks, widths, "num_stacks"
        )
        self.blocks = _reformat_nbeats_network_argument(
            num_stacks, blocks, "blocks"
        )
        self.block_layers = _reformat_nbeats_network_argument(
            num_stacks, block_layers, "block_layers"
        )
        self.sharing = _reformat_nbeats_network_argument(
            num_stacks, sharing, "sharing"
        )
        self.expansion_coefficient_lengths = _reformat_nbeats_network_argument(
            num_stacks,
            expansion_coefficient_lengths,
            "expansion_coefficient_lengths",
        )
        self.stack_types = _reformat_nbeats_network_argument(
            num_stacks, stack_types, "stack_types"
        )
        self.prediction_length = prediction_length
        self.context_length = context_length

        with self.name_scope():
            self.net_blocks: List[NBEATSBlock] = []

            for stack_id in range(num_stacks):
                for block_id in range(blocks[stack_id]):
                    # in case sharing is enabled for the stack
                    params = (
                        self.net_blocks[-1].collect_params()
                        if (block_id > 0 and sharing[stack_id])
                        else None
                    )
                    self.net_blocks.append(
                        NBEATSBlock(
                            width=self.widths[stack_id],
                            block_layers=self.block_layers[stack_id],
                            expansion_coefficient_length=self.expansion_coefficient_lengths[
                                stack_id
                            ],
                            prediction_length=prediction_length,
                            context_length=context_length,
                            block_type=self.stack_types[stack_id],
                            params=params,
                        )
                    )

        def hybrid_forward(self, F, x, *args, **kwargs):
            backcast = x
            forecast = F.zeros_like(x)
            for i in range(len(self.net_blocks)):
                b, f = self.net_blocks[i](backcast)
                backcast -= b
                forecast += f
            return forecast
