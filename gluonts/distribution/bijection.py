# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np

# First-party imports
from gluonts.model.common import Tensor

# Relative imports
from .distribution import getF


class Bijection:
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


class InverseBijection(Bijection):
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


class AffineTransformation(Bijection):
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
    def event_dim(self) -> int:
        return 0


exp = _Exp()
log = _Log()
softrelu = _Softrelu()
