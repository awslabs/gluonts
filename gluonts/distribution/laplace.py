# Standard library imports
from typing import Dict, Tuple

# First-party imports
from gluonts.model.common import Tensor

# Relative imports
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class Laplace(Distribution):
    is_reparameterizable = True

    def __init__(self, mu: Tensor, b: Tensor, F=None) -> None:
        self.mu = mu
        self.b = b
        self.F = F if F else getF(mu)

    @property
    def batch_shape(self) -> Tuple:
        return self.mu.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        return -1.0 * (
            self.F.log(2.0 * self.b) + self.F.abs((x - self.mu) / self.b)
        )

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def stddev(self) -> Tensor:
        return 2.0 ** 0.5 * self.b

    def sample_rep(self, num_samples=None) -> Tensor:
        def s(mu: Tensor, b: Tensor) -> Tensor:
            ones = mu.ones_like()
            x = self.F.random.uniform(-0.5 * ones, 0.5 * ones)
            laplace_samples = mu - b * self.F.sign(x) * self.F.log(
                1.0 - 2.0 * self.F.abs(x)
            )
            return laplace_samples

        return _sample_multiple(
            s, mu=self.mu, b=self.b, num_samples=num_samples
        )


class LaplaceOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "b": 1}
    distr_cls: type = Laplace

    @classmethod
    def domain_map(cls, F, mu, b):
        b = softplus(F, b)
        return mu.squeeze(axis=-1), b.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
