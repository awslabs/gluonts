import abc
from box import Box
from torch import nn
from torch.distributions import Distribution, OneHotCategorical, \
    MultivariateNormal

from utils.utils import make_inv_tril_parametrization
from torch_extensions.ops import cov_from_invcholesky_param


def prepend_batch_dims(tensor, shp: tuple):
    batched_tensor = tensor
    for _ in shp:
        batched_tensor = batched_tensor[None, ...]
    batched_tensor = batched_tensor.repeat(tuple(shp) + (1,) * tensor.ndim)
    return batched_tensor


class ParametrisedDistribution(nn.Module):
    """
    Modules holding the parameters of a distribution. This has several utilities:
    - Parameters are represented in a preferred form.
    - Since pytorch distributions are not subclasses of Module,
    modules using them will not see their learnable parameters.
    This class can serve as a container to avoid that problem.
    - when used as a function (forward method),
    it extracts the batch shape and adds/repeats those dims allow for broadcasting,
    and returns the a distribution object.
    """

    def forward(self, x_dummy: None, batch_shape_to_prepend: tuple):
        assert x_dummy is None
        batched_dist_params = {
            name: prepend_batch_dims(val, shp=batch_shape_to_prepend)
            for name, val in self.dist_params.items()
        }
        return self.dist_cls(**batched_dist_params)

    @property
    def distribution(self) -> Distribution:
        return self.dist_cls(**self.dist_params)

    @property
    @abc.abstractmethod
    def dist_cls(self):
        raise NotImplementedError("must be implemented by child class")

    @property
    @abc.abstractmethod
    def dist_params(self) -> dict:
        raise NotImplementedError("must be implemented by child class")


class ParametrisedMultivariateNormal(ParametrisedDistribution):
    def __init__(self,
                 m, LVinv_tril, LVinv_logdiag,
                 requires_grad_m=True,
                 requires_diag_LVinv_tril=True,
                 requires_diag_LVinv_logdiag=False,
                 ):
        super().__init__()
        self.m = nn.Parameter(m, requires_grad=requires_grad_m)
        self.LVinv_tril = nn.Parameter(
            LVinv_tril, requires_grad=requires_diag_LVinv_tril)
        self.LVinv_logdiag = nn.Parameter(
            LVinv_logdiag, requires_grad=requires_diag_LVinv_logdiag)

    @property
    def dist_params(self):
        return Box(
            loc=self.m,
            covariance_matrix=cov_from_invcholesky_param(
                Linv_tril=self.LVinv_tril,
                Linv_logdiag=self.LVinv_logdiag,
            )
        )

    @property
    def dist_cls(self):
        return MultivariateNormal

    @classmethod
    def from_dist_parametrisation(cls, loc, covariance_matrix, **kwargs):
        """ convenience constructor """
        LVinv_tril, LVinv_logdiag = make_inv_tril_parametrization(
            covariance_matrix)
        return cls(
            m=loc,
            LVinv_tril=LVinv_tril,
            LVinv_logdiag=LVinv_logdiag,
            **kwargs,
        )


class ParametrisedOneHotCategorical(ParametrisedDistribution):
    def __init__(self, logits, requires_grad=True):
        super().__init__()
        self.logits = nn.Parameter(logits, requires_grad=requires_grad)

    @property
    def dist_params(self):
        return Box(logits=self.logits)

    @property
    def dist_cls(self):
        return OneHotCategorical
