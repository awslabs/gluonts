import numpy as np
import mxnet as mx

from gluonts.mx.distribution.gaussian import Gaussian
from gluonts.mx.distribution.mixture import MixtureDistribution


def test_quantile() -> None:
    mu = mx.nd.array(
        [[1, 10, 100, 1000, 10000], [-1, -10, -100, -1000, -10000]]
    )
    sigma = mx.nd.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 2)

    gau = Gaussian(mu, sigma)

    mixture_probs = mx.nd.broadcast_like(
        mx.nd.array([[0.5, 0.5]]).expand_dims(axis=0),
        mu,
        lhs_axes=(0, 1),
        rhs_axes=(0, 1),
    )
    mix = MixtureDistribution(
        mixture_probs=mixture_probs,
        components=[Gaussian(mu, sigma), Gaussian(mu, sigma)],
    )

    quantiles = mx.nd.array([0.1, 0.5, 0.9])
    gau_quantiles = gau.quantile(quantiles)
    mix_quantiles = mix.quantile(quantiles)

    relative_error = np.max(
        np.abs(gau_quantiles.asnumpy() / mix_quantiles.asnumpy() - 1)
    )
    assert relative_error < 1e-4
