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
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.evaluation import get_seasonality

VALID_N_BEATS_STACK_TYPES = "G", "S", "T"
VALID_LOSS_FUNCTIONS = "sMAPE", "MASE", "MAPE"


def linear_space(
    F, backcast_length: int, forecast_length: int, fwd_looking: bool
):
    if fwd_looking:
        return F.arange(0, forecast_length) / forecast_length
    else:
        return F.arange(-backcast_length, 0) / backcast_length  # Option 01
        # return F.arange(backcast_length, 0, -1) / backcast_length # Option 02
        # return F.arange(0, backcast_length) / backcast_length # Option 03


def seasonality_model(
    F,
    num_coefficients: int,
    context_length: int,
    prediction_length: int,
    is_forecast: bool,
) -> Tensor:
    """
    Creates a fourier series basis with num_coefficients coefficients for sine and cosine each.
    So the total number of learned coefficients amounts to 2*num_coefficients.
    """
    t = linear_space(
        F, context_length, prediction_length, fwd_looking=is_forecast
    )
    cosines = F.stack(
        *[F.cos(2 * np.pi * i * t) for i in range(num_coefficients)]
    )
    sines = F.stack(
        *[F.sin(2 * np.pi * i * t) for i in range(num_coefficients)]
    )
    S = F.concat(cosines, sines, dim=0)
    return S


def trend_model(
    F,
    num_coefficients: int,
    context_length: int,
    prediction_length: int,
    is_forecast: bool,
) -> Tensor:
    """
    Creates a polynomial basis of degree num_coefficients-1.
    """
    t = linear_space(
        F, context_length, prediction_length, fwd_looking=is_forecast
    )
    T = F.stack(*[t ** i for i in range(num_coefficients)])
    return T


class NBEATSBlock(mx.gluon.HybridBlock):
    """
    The NBEATS Block as described in the paper: https://arxiv.org/abs/1905.10437.

    Parameters
    ----------
    width
        Width of the fully connected layers with ReLu activation.
    num_block_layers
        Number of fully connected layers with ReLu activation.
    expansion_coefficient_length
        If the type is "G" (generic), then the length of the expansion coefficient.
        If type is "T" (trend), then it corresponds to the degree of the polynomial.
        If the type is "S" (seasonal) then its not used.
    prediction_length
        Length of the prediction. Also known as 'horizon'.
    context_length
        Number of time units that condition the predictions
        Also known as 'lookback period'.
    has_backcast
        Only the last block of the network doesn't.
    kwargs
        Arguments passed to 'HybridBlock'.
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        width: int,
        num_block_layers: int,
        expansion_coefficient_length: int,
        prediction_length: int,
        context_length: int,
        has_backcast: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.width = width
        self.num_block_layers = num_block_layers
        self.expansion_coefficient_length = expansion_coefficient_length
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.has_backcast = has_backcast

        # Parameter to control whether the basis matrix has
        # been initialised, if the block uses one
        self.basis_initialized = False

        # Only fc_stack defined concretely in this class
        with self.name_scope():
            self.fc_stack = mx.gluon.nn.HybridSequential()
            for i in range(self.num_block_layers):
                self.fc_stack.add(
                    mx.gluon.nn.Dense(
                        units=self.width,
                        activation="relu",
                        prefix=f"fc_stack_dense_{i}_",
                    )
                )

            # Subclasses will have to initialize these attributes appropriately:

            self.theta_backcast = None
            self.theta_forecast = None

            self.backcast = None
            self.forecast = None

            self.backcast_basis = None
            self.forecast_basis = None

    # This function is called upon first call of the hybrid_forward method
    def initialize_basis(self, F):
        pass

    def hybrid_forward(self, F, x, *args, **kwargs):
        # We do this to cache the constant basis matrix between forward passes
        if not self.basis_initialized:
            self.initialize_basis(F)
            self.basis_initialized = True

        x = self.fc_stack(x)
        theta_f = self.theta_forecast(x)
        forecast = self.forecast(theta_f)

        if self.has_backcast:
            theta_b = self.theta_backcast(x)
            backcast = self.backcast(theta_b)
            return backcast, forecast

        return forecast


class NBEATSGenericBlock(NBEATSBlock):
    """
    The NBEATS Block as described in the paper: https://arxiv.org/abs/1905.10437.
    This is the GenericBlock variant.

    Parameters
    ----------
    width
        Width of the fully connected layers with ReLu activation.
    num_block_layers
        Number of fully connected layers with ReLu activation.
    expansion_coefficient_length
        The length of the expansion coefficient.
    prediction_length
        Length of the prediction. Also known as 'horizon'.
    context_length
        Number of time units that condition the predictions
        Also known as 'lookback period'.
    has_backcast
        Only the last block of the network doesn't.
    kwargs
        Arguments passed to 'HybridBlock'.
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        with self.name_scope():
            if self.has_backcast:
                self.theta_backcast = mx.gluon.nn.Dense(
                    units=self.expansion_coefficient_length,
                    prefix=f"theta_backcast_dense_",  # linear activation:
                )
                self.backcast = mx.gluon.nn.Dense(
                    units=self.context_length,
                    prefix=f"backcast_dense_",  # linear activation:
                )
            self.theta_forecast = mx.gluon.nn.Dense(
                units=self.expansion_coefficient_length,
                prefix=f"theta_forecast_dense_",  # linear activation:
            )
            self.forecast = mx.gluon.nn.Dense(
                units=self.prediction_length,
                prefix=f"theta_dense_",  # linear activation:
            )


class NBEATSSeasonalBlock(NBEATSBlock):
    """
    The NBEATS Block as described in the paper: https://arxiv.org/abs/1905.10437.
    This is the Seasonal block variant.

    Parameters
    ----------
    width
        Width of the fully connected layers with ReLu activation.
    num_block_layers
        Number of fully connected layers with ReLu activation.
    expansion_coefficient_length
        Not used in this block type.
    prediction_length
        Length of the prediction. Also known as 'horizon'.
    context_length
        Number of time units that condition the predictions
        Also known as 'lookback period'.
    has_backcast
        Only the last block of the network doesn't.
    kwargs
        Arguments passed to 'HybridBlock'.
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # the number of coefficient in the fourier basis per sine and cosine each
        # determined depending on prediction length, as defined by paper
        # also: dont use floor because prediction_length=1 should be mapped to 0
        self.num_coefficients = int((self.prediction_length / 2) - 1) + 1

        with self.name_scope():
            if self.has_backcast:
                self.theta_backcast = mx.gluon.nn.Dense(
                    units=2 * self.num_coefficients,
                    prefix=f"theta_backcast_dense_",  # linear activation:
                )
                self.backcast = mx.gluon.nn.HybridLambda(
                    lambda F, thetas: F.dot(thetas, self.backcast_basis),
                    prefix=f"backcast_lambda_",
                )
            self.theta_forecast = mx.gluon.nn.Dense(
                units=2 * self.num_coefficients,
                prefix=f"theta_forecast_dense_",  # linear activation:
            )
            self.forecast = mx.gluon.nn.HybridLambda(
                lambda F, thetas: F.dot(thetas, self.forecast_basis),
                prefix=f"forecast_lambda_",
            )

    def initialize_basis(self, F):
        # these are essentially constant matrices of type F that
        # define the basis for the seasonal model
        if self.has_backcast:
            self.backcast_basis = seasonality_model(
                F,
                num_coefficients=self.num_coefficients,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                is_forecast=False,
            )
        self.forecast_basis = seasonality_model(
            F,
            num_coefficients=self.num_coefficients,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            is_forecast=True,
        )


class NBEATSTrendBlock(NBEATSBlock):
    """"
    The NBEATS Block as described in the paper: https://arxiv.org/abs/1905.10437.
    This is the Trend block variant.

    Parameters
    ----------
    width
        Width of the fully connected layers with ReLu activation.
    num_block_layers
        Number of fully connected layers with ReLu activation.
    expansion_coefficient_length
        The length of the number of expansion coefficients.
        This corresponds to degree of the polynomial basis as follows:
            expansion_coefficient_length-1
    prediction_length
        Length of the prediction. Also known as 'horizon'.
    context_length
        Number of time units that condition the predictions
        Also known as 'lookback period'.
    has_backcast
        Only the last block of the network doesn't.
    kwargs
        Arguments passed to 'HybridBlock'.
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        with self.name_scope():
            if self.has_backcast:
                self.theta_backcast = mx.gluon.nn.Dense(
                    units=self.expansion_coefficient_length,
                    prefix=f"theta_backcast_dense_",  # linear activation:
                )
                self.backcast = mx.gluon.nn.HybridLambda(
                    lambda F, thetas: F.dot(thetas, self.backcast_basis),
                    prefix=f"backcast_lambda_",
                )
            self.theta_forecast = mx.gluon.nn.Dense(
                units=self.expansion_coefficient_length,
                prefix=f"theta_forecast_dense_",  # linear activation:
            )
            self.forecast = mx.gluon.nn.HybridLambda(
                lambda F, thetas: F.dot(thetas, self.forecast_basis),
                prefix=f"forecast_lambda_",
            )

    def initialize_basis(self, F):
        # these are essentially constant matrices of type F that
        # define the basis for the trend model
        if self.has_backcast:
            self.backcast_basis = trend_model(
                F,
                num_coefficients=self.expansion_coefficient_length,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                is_forecast=False,
            )
        self.forecast_basis = trend_model(
            F,
            num_coefficients=self.expansion_coefficient_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            is_forecast=True,
        )


class NBEATSNetwork(mx.gluon.HybridBlock):
    """
    The NBEATS Network as described in the paper: https://arxiv.org/abs/1905.10437.
    This does not constitute the whole NBEATS model, which is en ensemble model
    comprised of a multitude of NBEATS Networks.

    Parameters
    ----------
    prediction_length
        Length of the prediction. Also known as 'horizon'.
    context_length
        Number of time units that condition the predictions
        Also known as 'lookback period'.
    num_stacks
        The number of stacks the network should contain.
        Default and recommended value for generic mode: 30
        Recommended value for interpretable mode: 2
    num_blocks
        The number of blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [1]
        Recommended value for interpretable mode: [3]
    num_block_layers
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
    kwargs
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
        num_blocks: List[int],
        num_block_layers: List[int],
        expansion_coefficient_lengths: List[int],
        sharing: List[bool],
        stack_types: List[str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_stacks = num_stacks
        self.widths = widths
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.stack_types = stack_types
        self.prediction_length = prediction_length
        self.context_length = context_length

        with self.name_scope():
            self.net_blocks: List[NBEATSBlock] = []

            # connect all the blocks correctly
            for stack_id in range(num_stacks):
                for block_id in range(num_blocks[stack_id]):
                    # in case sharing is enabled for the stack
                    params = (
                        self.net_blocks[-1].collect_params()
                        if (block_id > 0 and sharing[stack_id])
                        else None
                    )
                    # only last one does not have backcast
                    has_backcast = not (
                        stack_id == num_stacks - 1
                        and block_id == num_blocks[num_stacks - 1] - 1
                    )
                    if self.stack_types[stack_id] == "G":
                        net_block = NBEATSGenericBlock(
                            width=self.widths[stack_id],
                            num_block_layers=self.num_block_layers[stack_id],
                            expansion_coefficient_length=self.expansion_coefficient_lengths[
                                stack_id
                            ],
                            prediction_length=prediction_length,
                            context_length=context_length,
                            has_backcast=has_backcast,
                            params=params,
                        )
                    elif self.stack_types[stack_id] == "S":
                        net_block = NBEATSSeasonalBlock(
                            width=self.widths[stack_id],
                            num_block_layers=self.num_block_layers[stack_id],
                            expansion_coefficient_length=self.expansion_coefficient_lengths[
                                stack_id
                            ],
                            prediction_length=prediction_length,
                            context_length=context_length,
                            has_backcast=has_backcast,
                            params=params,
                        )
                    else:  # self.stack_types[stack_id] == "T"
                        net_block = NBEATSTrendBlock(
                            width=self.widths[stack_id],
                            num_block_layers=self.num_block_layers[stack_id],
                            expansion_coefficient_length=self.expansion_coefficient_lengths[
                                stack_id
                            ],
                            prediction_length=prediction_length,
                            context_length=context_length,
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

    def smape_loss(self, F, forecast: Tensor, future_target: Tensor) -> Tensor:
        r"""
        .. math::

            smape = (200/H)*mean(|Y - Y_hat| / (|Y| + |Y_hat|))

        According to paper: https://arxiv.org/abs/1905.10437.
        """

        # Stop gradient required for stable learning
        denominator = F.stop_gradient(F.abs(future_target) + F.abs(forecast))
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

        if self.loss_function == "MASE":
            assert (
                self.periodicity < self.context_length + self.prediction_length
            ), (
                "If the 'periodicity' of your data is less than 'context_length' + 'prediction_length' "
                "the seasonal_error cannot be calculated and thus 'MASE' cannot be used for optimization."
            )

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
            Prediction sample. Shape: (batch_size, 1, prediction_length).
        """
        # future_target never used
        forecasts = super().hybrid_forward(
            F, past_target=past_target, future_target=past_target
        )

        # dimension collapsed previously because we only have one sample each:
        forecasts = F.expand_dims(forecasts, axis=1)

        # (batch_size, 1, prediction_length)
        return forecasts
