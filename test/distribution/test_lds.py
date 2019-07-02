# Standard library imports
import pytest
import json
import gzip
import os

# Third-party imports
import numpy as np
import mxnet as mx

# First-party imports
from gluonts.distribution.lds import LDS


def assert_shape_and_finite(x, shape):
    assert x.shape == shape
    assert not np.isnan(x.asnumpy()).any()
    assert not np.isinf(x.asnumpy()).any()


current_path = os.path.dirname(os.path.abspath(__file__))

# The following files contain different sets of LDS parameters
# (coefficients and noise terms) and observations, and the log-density
# of the observations that were computed using pykalman
# (https://pykalman.github.io/).
@pytest.mark.parametrize(
    "data_filename",
    [
        os.path.join(current_path, "test_lds_data/data_level_issm.json.gz"),
        os.path.join(
            current_path, "test_lds_data/data_level_trend_issm.json.gz"
        ),
        os.path.join(
            current_path,
            "test_lds_data/data_level_trend_weekly_seasonal_issm.json.gz",
        ),
    ],
)
def test_lds_likelihood(data_filename):
    """
    Test to check that likelihood is correctly computed for different
    innovation state space models (ISSM).
    Note that ISSM is a special case of LDS.
    """
    with gzip.GzipFile(data_filename, 'r') as fp:
        data = json.load(fp=fp)

    lds = LDS(
        mx.nd.array(data["emission_coeff"]),
        mx.nd.array(data["transition_coeff"]),
        mx.nd.array(data["innovation_coeff"]),
        mx.nd.array(data["noise_std"]),
        mx.nd.array(data["residuals"]),
        mx.nd.array(data["prior_mean"]),
        mx.nd.array(data["prior_covariance"]),
        data["latent_dim"],
        data["output_dim"],
        data["seq_length"],
    )

    targets = mx.nd.array(data["targets"])
    true_likelihood = mx.nd.array(data["true_likelihood"])

    batch_size = lds.emission_coeff[0].shape[0]
    time_length = len(lds.emission_coeff)
    output_dim = lds.emission_coeff[0].shape[1]
    latent_dim = lds.emission_coeff[0].shape[2]

    assert lds.batch_shape == (batch_size, time_length)
    assert lds.event_shape == (output_dim,)

    likelihood, final_mean, final_cov = lds.log_prob(targets)

    assert_shape_and_finite(likelihood, shape=lds.batch_shape)
    assert_shape_and_finite(final_mean, shape=(batch_size, latent_dim))
    assert_shape_and_finite(
        final_cov, shape=(batch_size, latent_dim, latent_dim)
    )

    likelihood_per_item = likelihood.sum(axis=1)

    np.testing.assert_almost_equal(
        likelihood_per_item.asnumpy(),
        true_likelihood.asnumpy(),
        decimal=2,
        err_msg=f"Likelihood did not match: \n "
        f"true likelihood = {true_likelihood},\n"
        f"obtained likelihood = {likelihood_per_item}",
    )

    samples = lds.sample_marginals(num_samples=100)

    assert_shape_and_finite(
        samples, shape=(100,) + lds.batch_shape + lds.event_shape
    )

    sample = lds.sample_marginals()

    assert_shape_and_finite(sample, shape=lds.batch_shape + lds.event_shape)

    samples = lds.sample(num_samples=100)

    assert_shape_and_finite(
        samples, shape=(100,) + lds.batch_shape + lds.event_shape
    )

    sample = lds.sample()

    assert_shape_and_finite(sample, shape=lds.batch_shape + lds.event_shape)

    ll, _, _ = lds.log_prob(sample)

    assert_shape_and_finite(ll, shape=lds.batch_shape)
