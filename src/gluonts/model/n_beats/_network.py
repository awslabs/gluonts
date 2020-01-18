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
from gluonts.evaluation._base import get_seasonality

VALID_N_BEATS_STACK_TYPES = "G", "S", "T"
VALID_LOSS_FUNCTIONS = "sMAPE", "MASE", "MAPE"


def linear_space(F, backcast_length, forecast_length, fwd_looking=True):
    if fwd_looking:
        return F.arange(0, forecast_length) / forecast_length
    else:
        return F.arange(-backcast_length, 0) / backcast_length  # Option 01
        # return F.arange(backcast_length, 0, -1) / backcast_length # Option 02
        # return F.arange(0, backcast_length) / backcast_length # Option 03


def seasonality_model(
    F, thetas, num_coefficients, context_length, prediction_length, is_forecast
):
    """
    Creates a fourier series basis with num_coefficients/2 coefficients for sine and cosine each.
    """
    t = linear_space(
        F, context_length, prediction_length, fwd_looking=is_forecast
    )
    cosines = F.stack(
        *[F.cos(2 * np.pi * i * t) for i in range(int(num_coefficients / 2))]
    )
    sines = F.stack(
        *[F.sin(2 * np.pi * i * t) for i in range(int(num_coefficients / 2))]
    )
    S = F.concat(cosines, sines, dim=0)
    return F.dot(thetas, S)


def trend_model(
    F,
    thetas,
    polynomial_degree,
    context_length,
    prediction_length,
    is_forecast,
):
    """
    Creates a polynomial basis of degree polynomial_degree-1.
    """
    t = linear_space(
        F, context_length, prediction_length, fwd_looking=is_forecast
    )
    T = F.stack(*[t ** i for i in range(polynomial_degree)])
    return F.dot(thetas, T)


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
        If the type is "G" (generic), then the length of the expansion coefficient.
        If type is "T" (trend), then it corresponds to the degree of the polynomial.
        If the type is "S" (seasonal) then its not used.
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
                    mx.gluon.nn.Dense(units=self.width, activation="relu")
                )
            if block_type == "G":
                if has_backcast:
                    self.theta_backcast = mx.gluon.nn.Dense(
                        units=expansion_coefficient_length  # linear activation:
                    )
                    self.backcast = mx.gluon.nn.Dense(
                        units=context_length  # linear activation:
                    )
                self.theta_forecast = mx.gluon.nn.Dense(
                    units=expansion_coefficient_length  # linear activation:
                )
                self.forecast = mx.gluon.nn.Dense(
                    units=prediction_length  # linear activation:
                )
            elif block_type == "S":
                num_coefficients = 2 * int(
                    (prediction_length / 2) - 1
                )  # according to paper
                if has_backcast:
                    self.theta_backcast = mx.gluon.nn.Dense(
                        units=num_coefficients  # linear activation:
                    )
                    self.backcast = mx.gluon.nn.HybridLambda(
                        lambda F, thetas: seasonality_model(
                            F,
                            thetas,
                            num_coefficients=num_coefficients,
                            context_length=context_length,
                            prediction_length=prediction_length,
                            is_forecast=False,
                        )
                    )
                self.theta_forecast = mx.gluon.nn.Dense(
                    units=num_coefficients  # linear activation:
                )
                self.forecast = mx.gluon.nn.HybridLambda(
                    lambda F, thetas: seasonality_model(
                        F,
                        thetas,
                        num_coefficients=num_coefficients,
                        context_length=context_length,
                        prediction_length=prediction_length,
                        is_forecast=True,
                    )
                )
            else:  # "T"
                if self.has_backcast:
                    self.theta_backcast = mx.gluon.nn.Dense(
                        units=expansion_coefficient_length  # linear activation:
                    )
                    self.backcast = mx.gluon.nn.HybridLambda(
                        lambda F, thetas: trend_model(
                            F,
                            thetas,
                            polynomial_degree=expansion_coefficient_length,
                            context_length=context_length,
                            prediction_length=prediction_length,
                            is_forecast=False,
                        )
                    )
                self.theta_forecast = mx.gluon.nn.Dense(
                    units=expansion_coefficient_length  # linear activation:
                )
                self.forecast = mx.gluon.nn.HybridLambda(
                    lambda F, thetas: trend_model(
                        F,
                        thetas,
                        polynomial_degree=expansion_coefficient_length,
                        context_length=context_length,
                        prediction_length=prediction_length,
                        is_forecast=True,
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
        If the type is "G" (generic), then the length of the expansion coefficient.
        If type is "T" (trend), then it corresponds to the degree of the polynomial.
        If the type is "S" (seasonal) then its not used.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [2]
        Recommended value for interpretable mode: [2]
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

    # TODO: somehow interpretable mode not learning?
    def smape_loss(self, F, forecast: Tensor, future_target: Tensor) -> Tensor:
        r"""
        .. math::

            smape = (200/H)*mean(|Y - Y_hat| / (|Y| + |Y_hat|))

        According to paper: https://arxiv.org/abs/1905.10437.
        """

        denominator = F.abs(future_target) + F.abs(forecast)
        flag = denominator == 0

        smape = (200 / self.prediction_length) * F.mean(
            (F.abs(future_target - forecast) * (1 - flag))
            / (denominator + flag),
            axis=1,
        )

        return smape

    def mape_loss(self, F, forecast: Tensor, future_target: Tensor) -> Tensor:
        r"""
        .. math::

            mape = (100/H)*mean(|Y - Y_hat| / |Y|)

        According to paper: https://arxiv.org/abs/1905.10437.
        """

        denominator = F.abs(future_target)
        flag = denominator == 0

        mape = (100 / self.prediction_length) * F.mean(
            (F.abs(future_target - forecast) * (1 - flag))
            / (denominator + flag),
            axis=1,
        )

        return mape

    def mase_loss(
        self,
        F,
        forecast: Tensor,
        future_target: Tensor,
        past_target: Tensor,
        periodicity: int,
    ) -> Tensor:
        r"""
        .. math::

            mase = (1/H)*(mean(|Y - Y_hat|) / seasonal_error)

        According to paper: https://arxiv.org/abs/1905.10437.
        """
        factor = 1 / (
            self.context_length + self.prediction_length - periodicity
        )
        whole_target = F.concat(past_target, future_target, dim=1)
        seasonal_error = factor * F.mean(
            F.abs(
                F.slice_axis(whole_target, axis=1, begin=periodicity, end=None)
                - F.slice_axis(whole_target, axis=1, begin=0, end=-periodicity)
            ),
            axis=1,
        )
        flag = seasonal_error == 0

        mase = (
            F.mean(F.abs(future_target - forecast), axis=1) * (1 - flag)
        ) / (seasonal_error + flag)

        return mase


class NBEATSTrainingNetwork(NBEATSNetwork):
    @validated()
    def __init__(self, loss_function: str, freq: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.freq = freq

        # Covert frequency string like "2H" to whats called the periodicity m.
        # E.g. 12 in case of "2H" because of 12 times two hours in a day.
        self.periodicity = get_seasonality(self.freq)

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

        if self.loss_function == "sMAPE":
            loss = self.smape_loss(F, forecast, future_target)
        elif self.loss_function == "MAPE":
            loss = self.mape_loss(F, forecast, future_target)
        elif self.loss_function == "MASE":
            loss = self.mase_loss(
                F, forecast, future_target, past_target, self.periodicity
            )
        else:
            raise ValueError(
                f"Invalid value {self.loss_function} for argument loss_function."
            )

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
