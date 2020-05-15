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

"""
Test that maximizing likelihood allows to correctly recover distribution parameters for all
distributions exposed to the user.
"""
# Standard library imports
from typing import Iterable, List, Tuple

# Third-party imports
import mxnet as mx
import numpy as np
import pytest
from pydantic import PositiveFloat, PositiveInt

# First-party imports
from gluonts.model.common import NPArrayLike
from gluonts.distribution.box_cox_transform import (
    InverseBoxCoxTransform,
    InverseBoxCoxTransformOutput,
)
from gluonts.distribution import (
    DistributionOutput,
    StudentT,
    StudentTOutput,
    Gamma,
    GammaOutput,
    Beta,
    BetaOutput,
    MultivariateGaussian,
    MultivariateGaussianOutput,
    LowrankMultivariateGaussian,
    LowrankMultivariateGaussianOutput,
    Dirichlet,
    DirichletOutput,
    DirichletMultinomial,
    DirichletMultinomialOutput,
    NegativeBinomial,
    NegativeBinomialOutput,
    Laplace,
    LaplaceOutput,
    Gaussian,
    GaussianOutput,
    Poisson,
    PoissonOutput,
    PiecewiseLinear,
    PiecewiseLinearOutput,
    Binned,
    BinnedOutput,
    Categorical,
    CategoricalOutput,
    LogitNormal,
    LogitNormalOutput,
)
from gluonts.distribution.transformed_distribution_output import (
    TransformedDistributionOutput,
)
from gluonts.distribution.transformed_distribution import (
    TransformedDistribution,
)


NUM_SAMPLES = 2000
BATCH_SIZE = 32
TOL = 0.3
START_TOL_MULTIPLE = 1

np.random.seed(1)
mx.random.seed(1)


def inv_softplus(y: NPArrayLike) -> np.ndarray:
    # y = log(1 + exp(x))  ==>  x = log(exp(y) - 1)
    return np.log(np.exp(y) - 1)


def maximum_likelihood_estimate_sgd(
    distr_output: DistributionOutput,
    samples: mx.ndarray,
    init_biases: List[mx.ndarray.NDArray] = None,
    num_epochs: PositiveInt = PositiveInt(5),
    learning_rate: PositiveFloat = PositiveFloat(1e-2),
    hybridize: bool = True,
) -> List[np.ndarray]:
    model_ctx = mx.cpu()

    arg_proj = distr_output.get_args_proj()
    arg_proj.initialize()

    if hybridize:
        arg_proj.hybridize()

    if init_biases is not None:
        for param, bias in zip(arg_proj.proj, init_biases):
            param.params[param.prefix + "bias"].initialize(
                mx.initializer.Constant(bias), force_reinit=True
            )

    trainer = mx.gluon.Trainer(
        arg_proj.collect_params(),
        "sgd",
        {"learning_rate": learning_rate, "clip_gradient": 10.0},
    )

    # The input data to our model is one-dimensional
    dummy_data = mx.nd.array(np.ones((len(samples), 1)))

    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(dummy_data, samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for e in range(num_epochs):
        cumulative_loss = 0
        num_batches = 0
        # inner loop
        for i, (data, sample_label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            sample_label = sample_label.as_in_context(model_ctx)
            with mx.autograd.record():
                distr_args = arg_proj(data)
                distr = distr_output.distribution(distr_args)
                loss = distr.loss(sample_label)
                if not hybridize:
                    assert loss.shape == distr.batch_shape
            loss.backward()
            trainer.step(BATCH_SIZE)
            num_batches += 1

            cumulative_loss += mx.nd.mean(loss).asscalar()

            assert not np.isnan(cumulative_loss)
        print("Epoch %s, loss: %s" % (e, cumulative_loss / num_batches))

    if len(distr_args[0].shape) == 1:
        return [
            param.asnumpy() for param in arg_proj(mx.nd.array(np.ones((1, 1))))
        ]

    return [
        param[0].asnumpy() for param in arg_proj(mx.nd.array(np.ones((1, 1))))
    ]


@pytest.mark.parametrize("alpha, beta", [(3.75, 1.25)])
@pytest.mark.parametrize("hybridize", [True, False])
def test_beta_likelihood(alpha: float, beta: float, hybridize: bool) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    alphas = mx.nd.zeros((NUM_SAMPLES,)) + alpha
    betas = mx.nd.zeros((NUM_SAMPLES,)) + beta

    distr = Beta(alphas, betas)
    samples = distr.sample()

    init_biases = [
        inv_softplus(alpha - START_TOL_MULTIPLE * TOL * alpha),
        inv_softplus(beta - START_TOL_MULTIPLE * TOL * beta),
    ]

    alpha_hat, beta_hat = maximum_likelihood_estimate_sgd(
        BetaOutput(),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(10),
    )

    print("ALPHA:", alpha_hat, "BETA:", beta_hat)
    assert (
        np.abs(alpha_hat - alpha) < TOL * alpha
    ), f"alpha did not match: alpha = {alpha}, alpha = {alpha_hat}"
    assert (
        np.abs(beta_hat - beta) < TOL * beta
    ), f"beta did not match: beta = {beta}, beta_hat = {beta_hat}"


@pytest.mark.parametrize("mu, sigma, nu", [(2.3, 0.7, 6.0)])
@pytest.mark.parametrize("hybridize", [True, False])
def test_studentT_likelihood(
    mu: float, sigma: float, nu: float, hybridize: bool
) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    mus = mx.nd.zeros((NUM_SAMPLES,)) + mu
    sigmas = mx.nd.zeros((NUM_SAMPLES,)) + sigma
    nus = mx.nd.zeros((NUM_SAMPLES,)) + nu

    distr = StudentT(mus, sigmas, nus)
    samples = distr.sample()

    # nu takes very long to learn, so we initialize it at the true value.
    # transform used is softplus(x) + 2
    init_bias = [
        mu - START_TOL_MULTIPLE * TOL * mu,
        inv_softplus(sigma - START_TOL_MULTIPLE * TOL * sigma),
        inv_softplus(nu - 2),
    ]

    mu_hat, sigma_hat, nu_hat = maximum_likelihood_estimate_sgd(
        StudentTOutput(),
        samples,
        init_biases=init_bias,
        hybridize=hybridize,
        num_epochs=PositiveInt(10),
        learning_rate=PositiveFloat(1e-2),
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(sigma_hat - sigma) < TOL * sigma
    ), f"sigma did not match: sigma = {sigma}, sigma_hat = {sigma_hat}"
    assert (
        np.abs(nu_hat - nu) < TOL * nu
    ), "nu0 did not match: nu0 = %s, nu_hat = %s" % (nu, nu_hat)


@pytest.mark.parametrize("alpha, beta", [(3.75, 1.25)])
@pytest.mark.parametrize("hybridize", [True, False])
def test_gamma_likelihood(alpha: float, beta: float, hybridize: bool) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    alphas = mx.nd.zeros((NUM_SAMPLES,)) + alpha
    betas = mx.nd.zeros((NUM_SAMPLES,)) + beta

    distr = Gamma(alphas, betas)
    samples = distr.sample()

    init_biases = [
        inv_softplus(alpha - START_TOL_MULTIPLE * TOL * alpha),
        inv_softplus(beta - START_TOL_MULTIPLE * TOL * beta),
    ]

    alpha_hat, beta_hat = maximum_likelihood_estimate_sgd(
        GammaOutput(),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(5),
    )

    assert (
        np.abs(alpha_hat - alpha) < TOL * alpha
    ), f"alpha did not match: alpha = {alpha}, alpha = {alpha_hat}"
    assert (
        np.abs(beta_hat - beta) < TOL * beta
    ), f"beta did not match: beta = {beta}, beta_hat = {beta_hat}"


@pytest.mark.parametrize("mu, sigma", [(1.0, 0.1)])
@pytest.mark.parametrize("hybridize", [True, False])
def test_gaussian_likelihood(mu: float, sigma: float, hybridize: bool):
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    mus = mx.nd.zeros((NUM_SAMPLES,)) + mu
    sigmas = mx.nd.zeros((NUM_SAMPLES,)) + sigma

    distr = Gaussian(mus, sigmas)
    samples = distr.sample()

    init_biases = [
        mu - START_TOL_MULTIPLE * TOL * mu,
        inv_softplus(sigma - START_TOL_MULTIPLE * TOL * sigma),
    ]

    mu_hat, sigma_hat = maximum_likelihood_estimate_sgd(
        GaussianOutput(),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.001),
        num_epochs=PositiveInt(5),
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(sigma_hat - sigma) < TOL * sigma
    ), f"alpha did not match: sigma = {sigma}, sigma_hat = {sigma_hat}"


@pytest.mark.parametrize("hybridize", [True, False])
def test_multivariate_gaussian(hybridize: bool) -> None:
    num_samples = 2000
    dim = 2

    mu = np.arange(0, dim) / float(dim)

    L_diag = np.ones((dim,))
    L_low = 0.1 * np.ones((dim, dim)) * np.tri(dim, k=-1)
    L = np.diag(L_diag) + L_low
    Sigma = L.dot(L.transpose())

    distr = MultivariateGaussian(mu=mx.nd.array(mu), L=mx.nd.array(L))

    samples = distr.sample(num_samples)

    mu_hat, L_hat = maximum_likelihood_estimate_sgd(
        MultivariateGaussianOutput(dim=dim),
        samples,
        init_biases=None,  # todo we would need to rework biases a bit to use it in the multivariate case
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.01),
        num_epochs=PositiveInt(10),
    )

    distr = MultivariateGaussian(
        mu=mx.nd.array([mu_hat]), L=mx.nd.array([L_hat])
    )

    Sigma_hat = distr.variance[0].asnumpy()

    assert np.allclose(
        mu_hat, mu, atol=0.1, rtol=0.1
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert np.allclose(
        Sigma_hat, Sigma, atol=0.1, rtol=0.1
    ), f"Sigma did not match: sigma = {Sigma}, sigma_hat = {Sigma_hat}"


@pytest.mark.parametrize("hybridize", [True, False])
def test_dirichlet(hybridize: bool) -> None:
    num_samples = 2000
    dim = 3

    alpha = np.array([1.0, 2.0, 3.0])

    distr = Dirichlet(alpha=mx.nd.array(alpha))
    cov = distr.variance.asnumpy()

    samples = distr.sample(num_samples)

    alpha_hat = maximum_likelihood_estimate_sgd(
        DirichletOutput(dim=dim),
        samples,
        init_biases=None,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(10),
    )

    distr = Dirichlet(alpha=mx.nd.array(alpha_hat))

    cov_hat = distr.variance.asnumpy()

    assert np.allclose(
        alpha_hat, alpha, atol=0.1, rtol=0.1
    ), f"alpha did not match: alpha = {alpha}, alpha_hat = {alpha_hat}"
    assert np.allclose(
        cov_hat, cov, atol=0.1, rtol=0.1
    ), f"Covariance did not match: cov = {cov}, cov_hat = {cov_hat}"


@pytest.mark.parametrize("hybridize", [True, False])
def test_dirichlet_multinomial(hybridize: bool) -> None:
    num_samples = 8000
    dim = 3
    n_trials = 500

    alpha = np.array([1.0, 2.0, 3.0])

    distr = DirichletMultinomial(
        dim=3, n_trials=n_trials, alpha=mx.nd.array(alpha)
    )
    cov = distr.variance.asnumpy()

    samples = distr.sample(num_samples)

    alpha_hat = maximum_likelihood_estimate_sgd(
        DirichletMultinomialOutput(dim=dim, n_trials=n_trials),
        samples,
        init_biases=None,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(10),
    )

    distr = DirichletMultinomial(
        dim=3, n_trials=n_trials, alpha=mx.nd.array(alpha_hat)
    )

    cov_hat = distr.variance.asnumpy()

    assert np.allclose(
        alpha_hat, alpha, atol=0.1, rtol=0.1
    ), f"alpha did not match: alpha = {alpha}, alpha_hat = {alpha_hat}"
    assert np.allclose(
        cov_hat, cov, atol=0.1, rtol=0.1
    ), f"Covariance did not match: cov = {cov}, cov_hat = {cov_hat}"


@pytest.mark.parametrize("hybridize", [True, False])
def test_lowrank_multivariate_gaussian(hybridize: bool) -> None:
    num_samples = 2000
    dim = 2
    rank = 1

    mu = np.arange(0, dim) / float(dim)
    D = np.eye(dim) * (np.arange(dim) / dim + 0.5)
    W = np.sqrt(np.ones((dim, rank)) * 0.2)
    Sigma = D + W.dot(W.transpose())

    distr = LowrankMultivariateGaussian(
        mu=mx.nd.array([mu]),
        D=mx.nd.array([np.diag(D)]),
        W=mx.nd.array([W]),
        dim=dim,
        rank=rank,
    )

    assert np.allclose(
        distr.variance[0].asnumpy(), Sigma, atol=0.1, rtol=0.1
    ), f"did not match: sigma = {Sigma}, sigma_hat = {distr.variance[0]}"

    samples = distr.sample(num_samples).squeeze().asnumpy()

    mu_hat, D_hat, W_hat = maximum_likelihood_estimate_sgd(
        LowrankMultivariateGaussianOutput(
            dim=dim, rank=rank, sigma_init=0.2, sigma_minimum=0.0
        ),
        samples,
        learning_rate=PositiveFloat(0.01),
        num_epochs=PositiveInt(25),
        init_biases=None,  # todo we would need to rework biases a bit to use it in the multivariate case
        hybridize=hybridize,
    )

    distr = LowrankMultivariateGaussian(
        dim=dim,
        rank=rank,
        mu=mx.nd.array([mu_hat]),
        D=mx.nd.array([D_hat]),
        W=mx.nd.array([W_hat]),
    )

    Sigma_hat = distr.variance.asnumpy()

    assert np.allclose(
        mu_hat, mu, atol=0.2, rtol=0.1
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"

    assert np.allclose(
        Sigma_hat, Sigma, atol=0.1, rtol=0.1
    ), f"sigma did not match: sigma = {Sigma}, sigma_hat = {Sigma_hat}"


@pytest.mark.parametrize("mu", [6.0])
@pytest.mark.parametrize("hybridize", [True, False])
def test_deterministic_l2(mu: float, hybridize: bool) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters.
    This tests uses the Gaussian distribution with fixed variance and sample mean.
    This essentially reduces to determistic L2.
    """
    # generate samples
    mu = mu
    mus = mx.nd.zeros(NUM_SAMPLES) + mu

    deterministic_distr = Gaussian(mu=mus, sigma=0.1 * mx.nd.ones_like(mus))
    samples = deterministic_distr.sample()

    class GaussianFixedVarianceOutput(GaussianOutput):
        @classmethod
        def domain_map(cls, F, mu, sigma):
            sigma = 0.1 * F.ones_like(sigma)
            return mu.squeeze(axis=-1), sigma.squeeze(axis=-1)

    mu_hat, _ = maximum_likelihood_estimate_sgd(
        GaussianFixedVarianceOutput(),
        samples,
        init_biases=[3 * mu, 0.1],
        hybridize=hybridize,
        num_epochs=PositiveInt(1),
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"


@pytest.mark.parametrize("mu", [1.0])
@pytest.mark.parametrize("hybridize", [True, False])
def test_deterministic_l1(mu: float, hybridize: bool) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters.
    This tests uses the Laplace distribution with fixed variance and sample mean.
    This essentially reduces to determistic L1.
    """
    # generate samples
    mu = mu
    mus = mx.nd.zeros(NUM_SAMPLES) + mu

    class LaplaceFixedVarianceOutput(LaplaceOutput):
        @classmethod
        def domain_map(cls, F, mu, b):
            b = 0.1 * F.ones_like(b)
            return mu.squeeze(axis=-1), b.squeeze(axis=-1)

    deterministic_distr = Laplace(mu=mus, b=0.1 * mx.nd.ones_like(mus))
    samples = deterministic_distr.sample()

    mu_hat, _ = maximum_likelihood_estimate_sgd(
        LaplaceFixedVarianceOutput(),
        samples,
        init_biases=[3 * mu, 0.1],
        learning_rate=PositiveFloat(1e-3),
        hybridize=hybridize,
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"


@pytest.mark.parametrize("mu_alpha", [(2.5, 0.7)])
@pytest.mark.parametrize("hybridize", [True, False])
def test_neg_binomial(mu_alpha: Tuple[float, float], hybridize: bool) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """
    # test instance
    mu, alpha = mu_alpha

    # generate samples
    mus = mx.nd.zeros((NUM_SAMPLES,)) + mu
    alphas = mx.nd.zeros((NUM_SAMPLES,)) + alpha

    neg_bin_distr = NegativeBinomial(mu=mus, alpha=alphas)
    samples = neg_bin_distr.sample()

    init_biases = [
        inv_softplus(mu - START_TOL_MULTIPLE * TOL * mu),
        inv_softplus(alpha + START_TOL_MULTIPLE * TOL * alpha),
    ]

    mu_hat, alpha_hat = maximum_likelihood_estimate_sgd(
        NegativeBinomialOutput(),
        samples,
        hybridize=hybridize,
        init_biases=init_biases,
        num_epochs=PositiveInt(15),
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(alpha_hat - alpha) < TOL * alpha
    ), f"alpha did not match: alpha = {alpha}, alpha_hat = {alpha_hat}"


@pytest.mark.parametrize("mu_b", [(3.3, 0.7)])
@pytest.mark.parametrize("hybridize", [True, False])
def test_laplace(mu_b: Tuple[float, float], hybridize: bool) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """
    # test instance
    mu, b = mu_b

    # generate samples
    mus = mx.nd.zeros((NUM_SAMPLES,)) + mu
    bs = mx.nd.zeros((NUM_SAMPLES,)) + b

    laplace_distr = Laplace(mu=mus, b=bs)
    samples = laplace_distr.sample()

    init_biases = [
        mu - START_TOL_MULTIPLE * TOL * mu,
        inv_softplus(b + START_TOL_MULTIPLE * TOL * b),
    ]

    mu_hat, b_hat = maximum_likelihood_estimate_sgd(
        LaplaceOutput(), samples, hybridize=hybridize, init_biases=init_biases
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(b_hat - b) < TOL * b
    ), f"b did not match: b = {b}, b_hat = {b_hat}"


@pytest.mark.parametrize(
    "gamma, slopes, knot_spacings",
    [(2.0, np.array([3, 1, 3, 4]), np.array([0.3, 0.2, 0.35, 0.15]))],
)
@pytest.mark.parametrize("hybridize", [True, False])
def test_piecewise_linear(
    gamma: float,
    slopes: np.ndarray,
    knot_spacings: np.ndarray,
    hybridize: bool,
) -> None:
    """
    Test to check that minimizing the CRPS recovers the quantile function
    """
    num_samples = 500  # use a few samples for timeout failure

    gammas = mx.nd.zeros((num_samples,)) + gamma
    slopess = mx.nd.zeros((num_samples, len(slopes))) + mx.nd.array(slopes)
    knot_spacingss = mx.nd.zeros(
        (num_samples, len(knot_spacings))
    ) + mx.nd.array(knot_spacings)

    pwl_sqf = PiecewiseLinear(gammas, slopess, knot_spacingss)

    samples = pwl_sqf.sample()

    # Parameter initialization
    gamma_init = gamma - START_TOL_MULTIPLE * TOL * gamma
    slopes_init = slopes - START_TOL_MULTIPLE * TOL * slopes
    knot_spacings_init = knot_spacings
    # We perturb knot spacings such that even after the perturbation they sum to 1.
    mid = len(slopes) // 2
    knot_spacings_init[:mid] = (
        knot_spacings[:mid] - START_TOL_MULTIPLE * TOL * knot_spacings[:mid]
    )
    knot_spacings_init[mid:] = (
        knot_spacings[mid:] + START_TOL_MULTIPLE * TOL * knot_spacings[mid:]
    )

    init_biases = [gamma_init, slopes_init, knot_spacings_init]

    # check if it returns original parameters of mapped
    gamma_hat, slopes_hat, knot_spacings_hat = maximum_likelihood_estimate_sgd(
        PiecewiseLinearOutput(len(slopes)),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.01),
        num_epochs=PositiveInt(20),
    )

    # Since the problem is highly non-convex we may not be able to recover the exact parameters
    # Here we check if the estimated parameters yield similar function evaluations at different quantile levels.
    quantile_levels = np.arange(0.1, 1.0, 0.1)

    # create a LinearSplines instance with the estimated parameters to have access to .quantile
    pwl_sqf_hat = PiecewiseLinear(
        mx.nd.array(gamma_hat),
        mx.nd.array(slopes_hat).expand_dims(axis=0),
        mx.nd.array(knot_spacings_hat).expand_dims(axis=0),
    )

    # Compute quantiles with the estimated parameters
    quantiles_hat = np.squeeze(
        pwl_sqf_hat.quantile_internal(
            mx.nd.array(quantile_levels).expand_dims(axis=0), axis=1
        ).asnumpy()
    )

    # Compute quantiles with the original parameters
    # Since params is replicated across samples we take only the first entry
    quantiles = np.squeeze(
        pwl_sqf.quantile_internal(
            mx.nd.array(quantile_levels)
            .expand_dims(axis=0)
            .repeat(axis=0, repeats=num_samples),
            axis=1,
        ).asnumpy()[0, :]
    )

    for ix, (quantile, quantile_hat) in enumerate(
        zip(quantiles, quantiles_hat)
    ):
        assert np.abs(quantile_hat - quantile) < TOL * quantile, (
            f"quantile level {quantile_levels[ix]} didn't match:"
            f" "
            f"q = {quantile}, q_hat = {quantile_hat}"
        )


@pytest.mark.skip("this test fails when run locally")
@pytest.mark.parametrize("lam_1, lam_2", [(0.1, 0.01)])
@pytest.mark.parametrize("mu, sigma", [(-1.5, 0.5)])
@pytest.mark.parametrize("hybridize", [True])
def test_box_cox_tranform(
    lam_1: float, lam_2: float, mu: float, sigma: float, hybridize: bool
):
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    lamdas_1 = mx.nd.zeros((NUM_SAMPLES,)) + lam_1
    lamdas_2 = mx.nd.zeros((NUM_SAMPLES,)) + lam_2
    transform = InverseBoxCoxTransform(lamdas_1, lamdas_2)

    mus = mx.nd.zeros((NUM_SAMPLES,)) + mu
    sigmas = mx.nd.zeros((NUM_SAMPLES,)) + sigma
    gausian_distr = Gaussian(mus, sigmas)

    # Here the base distribution is Guassian which is transformed to
    # non-Gaussian via the inverse Box-Cox transform.
    # Sampling from `trans_distr` gives non-Gaussian samples
    trans_distr = TransformedDistribution(gausian_distr, [transform])

    # Given the non-Gaussian samples find the true parameters
    # of the Box-Cox transformation as well as the underlying Gaussian distribution.
    samples = trans_distr.sample()

    init_biases = [
        mu - START_TOL_MULTIPLE * TOL * mu,
        inv_softplus(sigma - START_TOL_MULTIPLE * TOL * sigma),
        lam_1 - START_TOL_MULTIPLE * TOL * lam_1,
        inv_softplus(lam_2 - START_TOL_MULTIPLE * TOL * lam_2),
    ]

    mu_hat, sigma_hat, lam_1_hat, lam_2_hat = maximum_likelihood_estimate_sgd(
        TransformedDistributionOutput(
            GaussianOutput(),
            InverseBoxCoxTransformOutput(lb_obs=lam_2, fix_lambda_2=True),
        ),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.01),
        num_epochs=PositiveInt(18),
    )

    assert (
        np.abs(lam_1_hat - lam_1) < TOL * lam_1
    ), f"lam_1 did not match: lam_1 = {lam_1}, lam_1_hat = {lam_1_hat}"
    # assert (
    #     np.abs(lam_2_hat - lam_2) < TOL * lam_2
    # ), f"lam_2 did not match: lam_2 = {lam_2}, lam_2_hat = {lam_2_hat}"

    assert np.abs(mu_hat - mu) < TOL * np.abs(
        mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(sigma_hat - sigma) < TOL * sigma
    ), f"sigma did not match: sigma = {sigma}, sigma_hat = {sigma_hat}"


@pytest.mark.parametrize("num_bins", [6])
@pytest.mark.parametrize(
    "bin_probabilites", [np.array([0.3, 0.1, 0.05, 0.2, 0.1, 0.25])]
)
# some strange mxnet issue similar to this: https://github.com/apache/incubator-mxnet/issues/14228
# prevents hybridization for the maximum_likelihood_estimate_sgd test testing function.
# However, BinnedOutput does work with hybridize in a normal model.
# @pytest.mark.parametrize("hybridize", [True, False])
@pytest.mark.parametrize("hybridize", [False])
def test_binned_likelihood(
    num_bins: float, bin_probabilites: np.ndarray, hybridize: bool
):
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    bin_prob = mx.nd.array(bin_probabilites)
    bin_center = mx.nd.array(np.logspace(-1, 1, num_bins))

    # generate samples
    bin_probs = mx.nd.zeros((NUM_SAMPLES, num_bins)) + bin_prob
    bin_centers = mx.nd.zeros((NUM_SAMPLES, num_bins)) + bin_center

    distr = Binned(bin_probs.log(), bin_centers)
    samples = distr.sample()

    # add some jitter to the uniform initialization and normalize
    bin_prob_init = mx.nd.random_uniform(1 - TOL, 1 + TOL, num_bins) * bin_prob
    bin_prob_init = bin_prob_init / bin_prob_init.sum()

    init_biases = [bin_prob_init]

    bin_log_prob_hat, _ = maximum_likelihood_estimate_sgd(
        BinnedOutput(bin_center),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(25),
    )

    bin_prob_hat = np.exp(bin_log_prob_hat)

    assert all(
        mx.nd.abs(mx.nd.array(bin_prob_hat) - bin_prob) < TOL * bin_prob
    ), f"bin_prob did not match: bin_prob = {bin_prob}, bin_prob_hat = {bin_prob_hat}"


@pytest.mark.parametrize("num_cats", [6])
@pytest.mark.parametrize(
    "cat_probs", [np.array([0.3, 0.1, 0.05, 0.2, 0.1, 0.25])]
)
@pytest.mark.parametrize("hybridize", [True, False])
def test_categorical_likelihood(
    num_cats: int, cat_probs: np.ndarray, hybridize: bool
):
    """
    Test to check that maximizing the likelihood recovers the parameters
    """
    cat_prob = mx.nd.array(cat_probs)
    cat_probs = mx.nd.zeros((NUM_SAMPLES, num_cats)) + cat_prob

    distr = Categorical(cat_probs.log())
    samples = distr.sample()

    cat_prob_init = mx.nd.random_uniform(1 - TOL, 1 + TOL, num_cats) * cat_prob
    cat_prob_init = cat_prob_init / cat_prob_init.sum()

    init_biases = [cat_prob_init]

    cat_log_prob_hat = maximum_likelihood_estimate_sgd(
        CategoricalOutput(num_cats),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(25),
    )
    cat_prob_hat = np.exp(cat_log_prob_hat)

    prob_deviation = np.abs(cat_prob_hat - cat_prob.asnumpy()).flatten()
    tolerance = (TOL * cat_prob.asnumpy()).flatten()

    assert np.all(
        np.less(prob_deviation, tolerance)
    ), f"cat_prob did not match: cat_prob = {cat_prob}, cat_prob_hat = {cat_prob_hat}"


@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize("hybridize", [True, False])
def test_poisson_likelihood(rate: float, hybridize: bool) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    rates = mx.nd.zeros(NUM_SAMPLES) + rate

    distr = Poisson(rates)
    samples = distr.sample()

    init_biases = [inv_softplus(rate - START_TOL_MULTIPLE * TOL * rate)]

    rate_hat = maximum_likelihood_estimate_sgd(
        PoissonOutput(),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(20),
    )

    print("rate:", rate_hat)
    assert (
        np.abs(rate_hat[0] - rate) < TOL * rate
    ), f"mu did not match: rate = {rate}, rate_hat = {rate_hat}"


@pytest.mark.parametrize("mu, sigma", [(1.0, 0.1)])
@pytest.mark.parametrize("hybridize", [True, False])
def test_logit_normal_likelihood(mu: float, sigma: float, hybridize: bool):
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    mus = mx.nd.zeros((NUM_SAMPLES,)) + mu
    sigmas = mx.nd.zeros((NUM_SAMPLES,)) + sigma

    distr = LogitNormal(mus, sigmas)
    samples = distr.sample()

    init_biases = [
        mu - START_TOL_MULTIPLE * TOL * mu,
        inv_softplus(sigma - START_TOL_MULTIPLE * TOL * sigma),
    ]

    mu_hat, sigma_hat = maximum_likelihood_estimate_sgd(
        LogitNormalOutput(),
        samples,
        init_biases=init_biases,
        hybridize=hybridize,
        learning_rate=PositiveFloat(0.001),
        num_epochs=PositiveInt(5),
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(sigma_hat - sigma) < TOL * sigma
    ), f"sigma did not match: sigma = {sigma}, sigma_hat = {sigma_hat}"
