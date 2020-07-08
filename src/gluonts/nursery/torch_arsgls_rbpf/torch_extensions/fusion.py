from typing import Tuple, Dict
import torch
from torch.distributions import (
    MultivariateNormal,
    Bernoulli,
    Categorical,
    OneHotCategorical,
    Normal,
    Distribution,
)
from torch import nn
from torch_extensions.ops import matvec, batch_cholesky_inverse, cholesky

from utils.utils import flatten_iterable, one_hot


class ProbabilisticSensorFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *distributions):
        distributions = flatten_iterable([*distributions])
        self._check_inputs(*distributions)
        if len(distributions) == 1:
            return distributions[0]
        dist_cls = type(distributions[0])
        natural_params = self.get_natural_params(distributions)
        fused_natural_params = self.sum_natural_params(natural_params)
        fused_dist_params = self.natural_to_dist_params(
            fused_natural_params, dist_cls=dist_cls
        )
        fused_dist = dist_cls(**fused_dist_params)
        return fused_dist

    @staticmethod
    def get_natural_params(dists: Tuple[Distribution]):
        dist_cls = type(dists[0])
        if dist_cls is Normal:
            return tuple(
                {
                    "eta": dist.loc / dist.scale.pow(2),
                    "neg_precision": -0.5 * dist.scale.pow(2).reciprocal(),
                }
                for dist in dists
            )
        elif dist_cls is MultivariateNormal:
            return tuple(
                {
                    "eta": matvec(dist.precision_matrix, dist.loc),
                    "neg_precision": -0.5 * dist.precision_matrix,
                }
                for dist in dists
            )
        if dist_cls is Bernoulli:
            return tuple({"logits": dist.logits,} for dist in dists)
        elif dist_cls in (Categorical, OneHotCategorical):
            return tuple({"logits": dist.logits,} for dist in dists)
        else:
            raise NotImplementedError(f"Not implemented for {type(dists)}")

    @staticmethod
    def natural_to_dist_params(natural_params: dict, dist_cls):
        """
        Map the natural parameters to one of pytorch's accepted distribution parameters.
        This is not necessarily the so called canonical response function.
        E.g. the Bernoulli and Categoricals are mapped to logits, not to probs,
        so that we do not unnecessarily switch around between the two.
        """
        if dist_cls is Normal:
            eta, neg_precision = (
                natural_params["eta"],
                natural_params["neg_precision"],
            )
            return {
                "loc": -0.5 * eta / neg_precision,
                "scale": torch.sqrt(-0.5 / neg_precision),
            }
        elif dist_cls is MultivariateNormal:
            eta, neg_precision = (
                natural_params["eta"],
                natural_params["neg_precision"],
            )
            precision_matrix = -2 * neg_precision
            covariance_matrix = batch_cholesky_inverse(
                cholesky(precision_matrix)
            )
            return {
                "loc": matvec(covariance_matrix, eta),
                "covariance_matrix": covariance_matrix,
            }
        if dist_cls is Bernoulli:
            return natural_params
        elif dist_cls in (Categorical, OneHotCategorical):
            return natural_params
        else:
            raise NotImplementedError(f"Not implemented for {type(dist_cls)}")

    @staticmethod
    def sum_natural_params(natural_params: Tuple[Dict]):
        natural_param_names = natural_params[0].keys()
        n_dists = len(natural_params)
        return {
            np_name: sum(
                natural_params[idx_param][np_name]
                for idx_param in range(n_dists)
            )
            for np_name in natural_param_names
        }

    @staticmethod
    def _check_inputs(*distributions):
        assert len(distributions) >= 1
        assert isinstance(distributions[0], Distribution)
        assert all(
            type(distributions[0]) == type(dist) for dist in distributions
        )


def test_fusion_manually(distributions, threshold=1e-4):
    fused_dist = ProbabilisticSensorFusion()(distributions)
    if type(distributions[0]) in [Categorical]:
        test_points = torch.randint(
            low=0,
            high=distributions[0].probs.shape[-1],
            size=(100,)
            + tuple(fused_dist.batch_shape)
            + tuple(fused_dist.event_shape),
        )
    elif type(distributions[0]) in [OneHotCategorical]:
        test_points = one_hot(
            torch.randint(
                low=0,
                high=distributions[0].probs.shape[-1] - 1,
                size=(100,)
                + tuple(fused_dist.batch_shape)
                + tuple(fused_dist.event_shape),
            ),
            num_classes=distributions[0].probs.shape[-1],
        )
    elif type(distributions[0]) in [Bernoulli]:
        test_points = torch.randint(
            low=0,
            high=2,
            size=(100,)
            + tuple(fused_dist.batch_shape)
            + tuple(fused_dist.event_shape),
        )
        # what the fuck pytorch... float Bernoulli but integer categoricals?!
        test_points = test_points.to(distributions[0].logits.dtype)
    else:
        test_points = torch.randn(
            (100,)
            + tuple(fused_dist.batch_shape)
            + tuple(fused_dist.event_shape)
        )

    log_prob_fused = fused_dist.log_prob(test_points)
    log_probs = [dist.log_prob(test_points) for dist in distributions]
    manually_fused_unnormalized = torch.stack(log_probs, dim=-1).sum(dim=-1)

    diff_log_prob_fused = log_prob_fused[1:] - log_prob_fused[:-1]
    diff_log_prob_manual = (
        manually_fused_unnormalized[1:] - manually_fused_unnormalized[:-1]
    )
    assert torch.all((diff_log_prob_fused - diff_log_prob_manual) < threshold)
    return True


def plot_gaussian_fusion():
    import matplotlib.pyplot as plt

    distributions = [
        Normal(loc=0.5 * torch.ones(1), scale=0.5 * torch.ones(1)),
        Normal(loc=-0.5 * torch.ones(1), scale=0.5 * torch.ones(1)),
        Normal(loc=-torch.ones(1), scale=torch.ones(1)),
        Normal(loc=torch.ones(1), scale=torch.ones(1)),
    ]
    fused_dist = ProbabilisticSensorFusion()(distributions)
    x = torch.linspace(-3, 3, 1000)
    prob_fused = torch.exp(fused_dist.log_prob(x))
    probs = [torch.exp(dist.log_prob(x)) for dist in distributions]
    plt.figure()
    for prob in probs:
        plt.plot(x, prob)
    plt.plot(x, prob_fused, linestyle="--", color="black")
    plt.legend([f"#{idx}" for idx in range(len(distributions))] + ["fused"])
    plt.ylabel("p(x)")
    plt.xlabel("x")
    plt.grid()
    plt.show()


