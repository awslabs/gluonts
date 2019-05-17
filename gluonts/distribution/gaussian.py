# Standard library imports
import math
from typing import Dict, Optional, Tuple

# First-party imports
from gluonts.model.common import Tensor

# Relative imports
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class Gaussian(Distribution):
    is_reparameterizable = True

    def __init__(self, mu: Tensor, sigma: Tensor, F=None) -> None:
        self.mu = mu
        self.sigma = sigma
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
        F = self.F
        mu, sigma = self.mu, self.sigma
        return -1.0 * (
            F.log(sigma)
            + 0.5 * math.log(2 * math.pi)
            + 0.5 * F.square((x - mu) / sigma)
        )

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def stddev(self) -> Tensor:
        return self.sigma

    def sample(self, num_samples: Optional[int] = None) -> Tensor:
        return _sample_multiple(
            self.F.sample_normal,
            mu=self.mu,
            sigma=self.sigma,
            num_samples=num_samples,
        )

    def sample_rep(self, num_samples: Optional[int] = None) -> Tensor:
        def s(mu: Tensor, sigma: Tensor) -> Tensor:
            raw_samples = self.F.sample_normal(
                mu=mu.zeros_like(), sigma=sigma.ones_like()
            )
            return sigma * raw_samples + mu

        return _sample_multiple(
            s, mu=self.mu, sigma=self.sigma, num_samples=num_samples
        )


class GaussianOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "sigma": 1}
    distr_cls: type = Gaussian

    @classmethod
    def domain_map(cls, F, mu, sigma):
        """
        Maps raw tensors to valid arguments for constructing a Gaussian
        distribution.

        Parameters
        ----------
        F
        mu
            Tensor of shape `(*batch_shape, 1)`
        sigma
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]
            Two squeezed tensors, of shape `(*batch_shape)`: the first has the
            same entries as `mu` and the second has entries mapped to the
            positive orthant.
        """
        sigma = softplus(F, sigma)
        return mu.squeeze(axis=-1), sigma.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
