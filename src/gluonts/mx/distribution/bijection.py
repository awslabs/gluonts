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

from typing import Optional, Union, List

import numpy as np
from mxnet.gluon.nn import HybridBlock

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import getF


class Bijection:
    """
    A bijective transformation.

    This is defined through the forward transformation (computed by the
    `f` method) and the inverse transformation (`f_inv`).
    """

    @validated()
    def __init__(self):
        pass

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

    def inverse_bijection(self) -> "Bijection":
        r"""
        Returns a Bijection instance that represents the inverse of this
        transformation.
        """
        return InverseBijection(self)

    @property
    def event_dim(self) -> int:
        raise NotImplementedError()

    @property
    def sign(self) -> Union[float, Tensor]:
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

    @validated()
    def __init__(self, bijection: Bijection) -> None:
        super().__init__(self)
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
    def sign(self) -> Union[float, Tensor]:
        return self._bijection.sign


class ComposedBijection(Bijection):
    """
    Encapsulates a series of bijections and implements functions associated to their composition.
    """

    @validated()
    def __init__(self, bijections: Optional[List[Bijection]] = None) -> None:
        super().__init__(self)
        self._bijections: List[Bijection] = []
        if bijections is not None:
            self.__iadd__(bijections)

    @property
    def event_shape(self):
        return self._bijections[0].event_shape

    @property
    def event_dim(self):
        return self._bijections[0].event_dim

    def f(self, x: Tensor) -> Tensor:
        """
        Computes the forward transform of the composition of bijections.

        Parameters
        ----------
        x
            Input Tensor for the forward transform.
        Returns
        -------
        Tensor
            Transformation of x by the forward composition of bijections

        """
        y = x
        for t in self._bijections:
            y = t.f(y)
        return y

    def f_inv(self, y: Tensor) -> Tensor:
        """
        Computes the inverse transform of a composition of bijections.

        Parameters
        ----------
        y
            Input Tensor for the inverse function

        Returns
        -------
        Tensor
            Transformation of y by the inverse composition of bijections
        """
        x = y
        for t in reversed(self._bijections):
            x = t.f_inv(x)
        return x

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithm of the absolute value of the Jacobian determinant corresponding to the composed bijection

        Parameters
        ----------
        x
            input of the forward transformation or output of the inverse transform
        y
            output of the forward transform or input of the inverse transform

        Returns
        -------
        Tensor
            Jacobian evaluated for x as input or y as output

        """
        ladj = 0.0
        for t in reversed(self._bijections):
            x = t.f_inv(y)
            # TODO: eventually change for
            # ladj = ladj + sum_trailing_axes(getF(y), t.log_abs_det_jac(x, y),
            #                                 self.event_dim - t.event_dim)
            ladj = ladj + t.log_abs_det_jac(x, y)
            y = x
        return ladj

    def __getitem__(self, index: int):
        return self._bijections[index]

    def __len__(self):
        return len(self._bijections)

    def __iadd__(self, bijections: List[Bijection]):
        for b in bijections:
            if not isinstance(b, Bijection):
                raise TypeError(
                    f"Object is of type {type(b)}"
                    f" but should inherit from {Bijection}."
                )

            if len(self._bijections) > 0 and b.event_shape != self.event_shape:
                raise RuntimeError(
                    f"Bijection {b} has event_shape of '{b.event_shape}'"
                    f"but should be of '{self.event_shape}'"
                )

            self._bijections.append(b)

        return self

    def __add__(self, bijections: List[Bijection]):
        return ComposedBijection(self._bijections + bijections)


class BijectionHybridBlock(HybridBlock, Bijection):
    """Allows a Bijection to have parameters"""


class ComposedBijectionHybridBlock(BijectionHybridBlock, ComposedBijection):
    """
    Allows a ComposedBijection object to have parameters
    """

    @validated()
    def __init__(
        self,
        bij_blocks: Optional[List[Bijection]] = None,
        *args,
        **kwargs,
    ) -> None:
        HybridBlock.__init__(self, *args, **kwargs)
        ComposedBijection.__init__(self, bij_blocks)

    def __iadd__(self, bij_blocks: List[Bijection]):
        for b in bij_blocks:
            self.register_child(b)
        return super().__iadd__(bij_blocks)


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
    def sign(self) -> float:
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
    def sign(self) -> float:
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
        return F.Activation(x.clip(-100.0, np.inf), act_type="softrelu")

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
    def sign(self) -> float:
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
        Translation parameter. If unspecified or `None`, this will be zero.
    scale
        Scaling parameter. If unspecified or `None`, this will be one.
    """

    @validated()
    def __init__(
        self, loc: Optional[Tensor] = None, scale: Optional[Tensor] = None
    ) -> None:
        super().__init__(self)
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
        F = getF(x)
        return F.zeros_like(x)

    @property
    def sign(self) -> Union[float, Tensor]:
        return 1.0 if self.scale is None else self.scale.sign()

    @property
    def event_dim(self) -> int:
        return 0


exp = _Exp()
log = _Log()
softrelu = _Softrelu()
