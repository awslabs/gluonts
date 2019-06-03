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
from typing import Dict, List, Tuple

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.core.component import validated
from gluonts.distribution.bijection import Bijection, InverseBijection
from gluonts.distribution.bijection_output import BijectionOutput
from gluonts.model.common import Tensor

# Relative imports
from .distribution import getF, softplus


class BoxCoxTranform(Bijection):
    r"""
    Implements Box-Cox transformation of a uni-variate random variable.
    The Box-Cox transformation of an observation :math:`z` is given by

    .. math::

        BoxCox(z; \lambda_1, \lambda_2) = \begin{cases}
                                              ((z + \lambda_2)^{\lambda_1} - 1) / \lambda_1, \quad & \text{if }
                                              \lambda_1 \neq 0, \\
                                              \log (z + \lambda_2), \quad & \text{otherwise.}
                                          \end{cases}

    Here, :math:`\lambda_1` and :math:`\lambda_2` are learnable parameters. Note that the domain
    of the transformation is not restricted.

    For numerical stability, instead of checking :math:`\lambda_1` is exactly zero, we use the condition

    .. math::

        |\lambda_1| < tol\_lambda\_1

    for a pre-specified tolerance `tol_lambda_1`.

    Inverse of the Box-Cox Transform is given by

    .. math::

        BoxCox^{-1}(y; \lambda_1, \lambda_2) = \begin{cases}
                                                (y \lambda_1 + 1)^{(1/\lambda_1)} - \lambda_2, \quad & \text{if }
                                                \lambda_1 \neq 0, \\
                                                \exp (y) - \lambda_2, \quad & \text{otherwise.}
                                               \end{cases}

    **Notes on numerical stability:**

    1.  For the forward transformation, :math:`\lambda_2` must always be chosen such that

        .. math::

            z + \lambda_2 > 0.

        To achieve this one needs to know a priori the lower bound on the observations.
        This is set in `BoxCoxTransformOutput`, since :math:`\lambda_2` is learnable.

    2.  Similarly for the inverse transformation to work reliably, a sufficient condition is

        .. math::

            y \lambda_1 + 1 \geq 0,

        where :math:`y` is the input to the inverse transformation.

        This cannot always be guaranteed especially when :math:`y` is a sample from a transformed distribution.
        Hence we always truncate :math:`y \lambda_1 + 1` at zero.

        An example showing why this could happen in our case:
        consider transforming observations from the unit interval (0, 1) with parameters

        .. math::

            \begin{align}
                \lambda_1 = &\ 1.1, \\
                \lambda_2 = &\ 0.
            \end{align}
        Then the range of the transformation is (-0.9090, 0.0).
        If Gaussian is fit to the transformed observations and a sample is drawn from it,
        then it is likely that the sample is outside this range, e.g., when the mean is close to -0.9.
        The subsequent inverse transformation of the sample is not a real number anymore.

        >>> y = -0.91
        >>> lambda_1 = 1.1
        >>> lambda_2 = 0.0
        >>> (y * lambda_1 + 1) ** (1 / lambda_1) + lambda_2
        (-0.0017979146510711471+0.0005279153735965289j)

    Parameters
    ----------
    lambda_1
    lambda_2
    tol_lambda_1
        For numerical stability, treat `lambda_1` as zero if it is less than
        `tol_lambda_1`
    F
    """
    arg_names = ['box_cox.lambda_1', 'box_cox.lambda_2']

    def __init__(
        self,
        lambda_1: Tensor,
        lambda_2: Tensor,
        tol_lambda_1: float = 1e-2,
        F=None,
    ) -> None:
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.tol_lambda_1 = tol_lambda_1
        self.F = F if F else getF(lambda_1)

        # Addressing mxnet madness
        self._power = self.F.power if self.F == mx.nd else self.F.pow

    @property
    def args(self) -> List:
        r"""
        List: current values of the parameters
        """
        return [self.lambda_1, self.lambda_2]

    @property
    def event_dim(self) -> int:
        return 0

    def f(self, z: Tensor) -> Tensor:
        r"""
        Forward transformation of observations `z`

        Parameters
        ----------
        z
            observations

        Returns
        -------
        Tensor
            Transformed observations
        """
        F = self.F
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        tol_lambda_1 = self.tol_lambda_1
        _power = self._power

        return F.where(
            condition=(F.abs(lambda_1).__ge__(tol_lambda_1).broadcast_like(z)),
            x=(_power(z + lambda_2, lambda_1) - 1.0) / lambda_1,
            y=F.log(z + lambda_2),
            name="Box_Cox_trans",
        )

    def f_inv(self, y: Tensor) -> Tensor:
        r"""Inverse of the Box-Cox Transform

        Parameters
        ----------
        y
            Transformed observations

        Returns
        -------
        Tensor
            Observations

        """
        F = self.F
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        tol_lambda_1 = self.tol_lambda_1
        _power = self._power

        # For numerical stability we truncate :math:`y * \lambda_1 + 1.0` at zero.
        base = F.relu(y * lambda_1 + 1.0)

        return F.where(
            condition=F.abs(lambda_1).__ge__(tol_lambda_1),
            x=_power(base, 1.0 / lambda_1) - lambda_2,
            y=F.exp(y) - lambda_2,
            name="Box_Cox_inverse_trans",
        )

    def log_abs_det_jac(self, z: Tensor, y: Tensor = None) -> Tensor:
        r"""
        Logarithm of the absolute value of the Jacobian determinant corresponding to the Box-Cox Transform
        is given by

        .. math::
            \log \frac{d}{dz} BoxCox(z; \lambda_1, \lambda_2) = \begin{cases}
                                \log (z + \lambda_2) (\lambda_1 - 1), \quad & \text{if } \lambda_1 \neq 0, \\
                                -\log (z + \lambda_2), \quad & \text{otherwise.}
                                \end{cases}

        Note that the derivative of the transformation is always non-negative.

        Parameters
        ----------
        z
            observations
        y
            not used

        Returns
        -------
        Tensor

        """
        F = self.F
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        tol_lambda_1 = self.tol_lambda_1

        return F.where(
            condition=F.abs(lambda_1).__ge__(tol_lambda_1),
            x=F.log(z + lambda_2) * (lambda_1 - 1.0),
            y=-F.log(z + lambda_2),
            name="Box_Cox_trans_log_det_jac",
        )


class BoxCoxTransformOutput(BijectionOutput):
    bij_cls: type = BoxCoxTranform
    args_dim: Dict[str, int] = dict(zip(BoxCoxTranform.arg_names, [1, 1]))

    @validated()
    def __init__(self, lb_obs: float = 0.0, fix_lambda_2: bool = True) -> None:
        super().__init__()
        self.lb_obs = lb_obs
        self.fix_lambda_2 = fix_lambda_2

    def domain_map(self, F, *args: Tensor) -> Tuple[Tensor, ...]:
        lambda_1, lambda_2 = args
        if self.fix_lambda_2:
            lambda_2 = self.lb_obs * F.ones_like(lambda_2)
        else:
            # This makes sure that :math:`z +  \lambda_2 > 0`, where :math:`z > lb_obs`
            lambda_2 = softplus(F, lambda_2) - self.lb_obs * F.ones_like(
                lambda_2
            )

        # we squeeze the output since event_shape is ()
        return lambda_1.squeeze(axis=-1), lambda_2.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()


class InverseBoxCoxTransform(InverseBijection):
    """
    Implements the inverse of Box-Cox transformation as a bijection.
    """

    arg_names = ['box_cox.lambda_1', 'box_cox.lambda_2']

    def __init__(
        self,
        lambda_1: Tensor,
        lambda_2: Tensor,
        tol_lambda_1: float = 1e-2,
        F=None,
    ) -> None:
        super().__init__(BoxCoxTranform(lambda_1, lambda_2, tol_lambda_1, F))

    @property
    def event_dim(self) -> int:
        return 0


class InverseBoxCoxTransformOutput(BoxCoxTransformOutput):
    bij_cls: type = InverseBoxCoxTransform

    args_dim: Dict[str, int] = dict(
        zip(InverseBoxCoxTransform.arg_names, [1, 1])
    )

    @property
    def event_shape(self) -> Tuple:
        return ()
