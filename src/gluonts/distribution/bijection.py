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
from typing import Optional

# Third-party imports
import numpy as np

# First-party imports
from gluonts.model.common import Tensor

# Relative imports
from .distribution import getF


class Bijection:
    """
    A bijective transformation.

    This is defined through the forward tranformation (computed by the
    `f` method) and the inverse transformation (`f_inv`).
    """

    def f(self, x: Tensor) -> Tensor:
        r"""
        Forward transformation x -> y
        """
        raise NotImplementedError

    def f_inv(self, y: Tensor) -> Tensor:
        r"""
        Inverse transformation y -> x
        """
        raise NotImplementedError

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Receives (x, y) and returns log of the absolute value of the Jacobian
        determinant

        .. math::
            \log |dy/dx|

        Note that this is the Jacobian determinant of the forward
        transformation x -> y.
        """
        raise NotImplementedError

    def inverse_bijection(self) -> 'Bijection':
        r"""
        Returns a Bijection instance that represents the inverse of this
        transformation.
        """
        return InverseBijection(self)

    @property
    def event_dim(self) -> int:
        raise NotImplementedError()

    @property
    def sign(self) -> Tensor:
        """
        Return the sign of the Jacobian's determinant.
        """
        raise NotImplementedError()


class InverseBijection(Bijection):
    """
    The inverse of a given transformation.

    This is a wrapper around bijective transformations, that inverts the
    role of `f` and `f_inv`, and modifies other related methods accordingly.

    Parameters
    ----------
    bijection
        The transformation to invert.
    """

    def __init__(self, bijection: Bijection) -> None:
        self._bijection = bijection

    def f(self, x: Tensor) -> Tensor:
        return self._bijection.f_inv(x)

    def f_inv(self, y: Tensor) -> Tensor:
        return self._bijection.f(y)

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        return -self._bijection.log_abs_det_jac(y, x)

    def inverse_bijection(self) -> Bijection:
        return self._bijection

    @property
    def event_dim(self) -> int:
        return self._bijection.event_dim

    @property
    def sign(self) -> Tensor:
        return self._bijection.sign


class _Exp(Bijection):
    def f(self, x: Tensor) -> Tensor:
        return x.clip(-np.inf, 30).exp()

    def f_inv(self, y: Tensor) -> Tensor:
        return y.clip(1.0e-20, np.inf).log()

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        return y.clip(1.0e-20, np.inf).log()

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def sign(self) -> Tensor:
        return 1.0


class _Log(Bijection):
    def f(self, x: Tensor) -> Tensor:
        return x.clip(1.0e-20, np.inf).log()

    def f_inv(self, y: Tensor) -> Tensor:
        return y.clip(-np.inf, 30).exp()

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        return -y

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def sign(self) -> Tensor:
        return 1.0


class _Softrelu(Bijection):
    def _log_expm1(self, F, y: Tensor) -> Tensor:
        r"""
        A numerically stable computation of :math:`x = \log(e^y - 1)`
        """
        thresh = F.zeros_like(y) + 20.0
        x = F.where(F.broadcast_greater(y, thresh), y, F.log(F.expm1(y)))
        return x

    def f(self, x: Tensor) -> Tensor:
        F = getF(x)
        return F.Activation(x.clip(-100.0, np.inf), act_type='softrelu')

    def f_inv(self, y: Tensor) -> Tensor:
        F = getF(y)
        return self._log_expm1(F, y)

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        F = getF(y)
        return self._log_expm1(F, y) - y

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def sign(self) -> Tensor:
        return 1.0


class AffineTransformation(Bijection):
    """
    An affine transformation consisting of a scaling and a translation.

    If translation is specified `loc`, and the scaling by `scale`, then
    this transformation computes `y = scale * x + loc`, where all operations
    are element-wise.

    Parameters
    ----------
    loc
        Translation parameter.
    scale
        Scaling parameter.
    """

    def __init__(
        self, loc: Optional[Tensor] = None, scale: Optional[Tensor] = None
    ) -> None:
        self.loc = loc
        self.scale = scale

    def f(self, x: Tensor) -> Tensor:
        F = getF(x)
        if self.scale is not None:
            x = F.broadcast_mul(x, self.scale)
        if self.loc is not None:
            x = F.broadcast_add(x, self.loc)
        return x

    def f_inv(self, y: Tensor) -> Tensor:
        F = getF(y)
        if self.loc is not None:
            y = F.broadcast_sub(y, self.loc)
        if self.scale is not None:
            y = F.broadcast_div(y, self.scale)
        return y

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        if self.scale is not None:
            return self.scale.broadcast_like(x).abs().log()
        else:
            raise RuntimeError("scale is of type None in log_abs_det_jac")

    @property
    def sign(self):
        return self.scale.sign()

    @property
    def event_dim(self) -> int:
        return 0


exp = _Exp()
log = _Log()
softrelu = _Softrelu()
