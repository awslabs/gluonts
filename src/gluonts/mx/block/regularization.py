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

from typing import List, Optional

from mxnet.gluon.loss import Loss

from gluonts.core.component import validated
from gluonts.mx import Tensor


class ActivationRegularizationLoss(Loss):
    r"""
    .. math::

        L = \alpha \|h_t\|_2^2,

    where :math:`h_t` is the output of the RNN at timestep t.
    :math:`\alpha` is scaling coefficient.
    The implementation follows [MMS17]_.
    Computes Activation Regularization Loss. (alias: AR)

    Parameters
    ----------
    alpha
        The scaling coefficient of the regularization.
    weight
        Global scalar weight for loss.
    batch_axis
        The axis that represents mini-batch.
    time_axis
        The axis that represents time-step.
    """

    @validated()
    def __init__(
        self,
        alpha: float = 0.0,
        weight: Optional[float] = None,
        batch_axis: int = 1,
        time_axis: int = 0,
        **kwargs
    ):
        super(ActivationRegularizationLoss, self).__init__(
            weight, batch_axis, **kwargs
        )
        self._alpha = alpha
        self._batch_axis = batch_axis
        self._time_axis = time_axis

    def __repr__(self):
        s = "ActivationRegularizationLoss (alpha={alpha})"
        return s.format(alpha=self._alpha)

    def hybrid_forward(self, F, *states: List[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        states
            the stack outputs from RNN, which consists of output from each time step.

        Returns
        --------
        Tensor
            loss tensor with shape (batch_size,). Dimensions other than batch_axis are averaged out.
        """
        if self._alpha != 0 and states:
            means = []
            for state in states:
                if isinstance(state, list):
                    state = F.stack(*state, axis=self._time_axis)
                means.append(
                    self._alpha
                    * F.power(state, 2).mean(
                        axis=self._batch_axis, exclude=True
                    )
                )
            return F.add_n(*means)
        return F.zeros(1)


class TemporalActivationRegularizationLoss(Loss):
    r"""
    .. math::

        L = \beta \| h_t-h_{t+1} \|_2^2,

    where :math:`h_t` is the output of the RNN at timestep t,
    :math:`h_{t+1}` is the output of the RNN at timestep t+1, :math:`\beta` is scaling coefficient.
    The implementation follows [MMS17]_.
    Computes Temporal Activation Regularization Loss. (alias: TAR)

    Parameters
    ----------
    beta
        The scaling coefficient of the regularization.
    weight
        Global scalar weight for loss.
    batch_axis
        The axis that represents mini-batch.
    time_axis
        The axis that represents time-step.
    """

    @validated()
    def __init__(
        self,
        beta: float = 0,
        weight: Optional[float] = None,
        batch_axis: int = 1,
        time_axis: int = 0,
        **kwargs
    ):
        super(TemporalActivationRegularizationLoss, self).__init__(
            weight, batch_axis, **kwargs
        )
        self._beta = beta
        self._batch_axis = batch_axis
        self._time_axis = time_axis

    def __repr__(self):
        s = "TemporalActivationRegularizationLoss (beta={beta})"
        return s.format(beta=self._beta)

    def hybrid_forward(self, F, *states: List[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        states
            the stack outputs from RNN, which consists of output from each time step.

        Returns
        --------
        Tensor
            loss tensor with shape (batch_size,). Dimensions other than batch_axis are averaged out.
        """
        if self._beta != 0 and states:
            means = []
            for state in states:
                if isinstance(state, list):
                    state = F.stack(*state, axis=self._time_axis)
                sub_state_1 = F.slice_axis(
                    state, axis=self._time_axis, begin=1, end=None
                )
                sub_state_2 = F.slice_axis(
                    state, axis=self._time_axis, begin=0, end=-1
                )
                sub_state_diff = F.elemwise_sub(sub_state_1, sub_state_2)
                means.append(
                    self._beta
                    * F.power(sub_state_diff, 2).mean(
                        axis=self._batch_axis, exclude=True
                    )
                )
            return F.add_n(*means)
        return F.zeros(1)
