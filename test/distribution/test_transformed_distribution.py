# Third-party imports
import mxnet.ndarray as nd
import numpy as np

# First-party imports
from gluonts.distribution import Uniform
from gluonts.distribution.transformed_distribution import (
    TransformedDistribution,
)
from gluonts.distribution import bijection


def exp_cdf(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-x)


def test_transformed_distribution() -> None:
    zero = nd.zeros(1)
    one = nd.ones(1)

    # If Y = -log(U) with U ~ Uniform(0, 1), then Y ~ Exponential(1)
    exponential = TransformedDistribution(
        Uniform(zero, one),
        bijection.log,
        bijection.AffineTransformation(scale=-1 * one),
    )

    # For Y ~ Exponential(1), P(Y) = e^{-x) ==> log P(Y) = -x
    assert exponential.log_prob(1 * one).asscalar() == -1.0
    assert exponential.log_prob(2 * one).asscalar() == -2.0

    v = np.linspace(0, 5, 101)
    assert np.allclose(exponential.cdf(nd.array(v)).asnumpy(), exp_cdf(v))

    # If Y ~ Exponential(1), then U = 1 - e^{-Y} has Uniform(0, 1) distribution
    uniform = TransformedDistribution(
        exponential,
        bijection.AffineTransformation(scale=-1 * one),
        bijection.log.inverse_bijection(),  # == bijection.exp
        bijection.AffineTransformation(loc=one, scale=-1 * one),
    )
    # For U ~ Uniform(0, 1), log P(U) = 0
    assert uniform.log_prob(0.5 * one).asscalar() == 0
    assert uniform.log_prob(0.2 * one).asscalar() == 0

    v = np.linspace(0, 1, 101)
    assert np.allclose(uniform.cdf(nd.array(v)).asnumpy(), v)
