import torch
from torch import nn
from torch.distributions import OneHotCategorical, MultivariateNormal

from torch_extensions.distributions.conditional_parametrised_distribution import (
    ParametrisedConditionalDistribution,
)
from torch_extensions.distributions.parametrised_distribution import (
    ParametrisedOneHotCategorical,
    ParametrisedMultivariateNormal,
)
from torch_extensions.mlp import MLP
from torch_extensions.layers_with_init import Linear
from utils.utils import make_inv_tril_parametrization
from torch_extensions.distributions.dist_param_rectifiers import (
    DefaultScaleTransform,
)


def _extract_dims_from_cfg(config):
    if config.dims.ctrl_switch is not None:
        dim_in = config.dims.ctrl_switch
    else:
        dim_in = None
    dim_out = config.dims.switch
    dims_stem = config.switch_prior_model_dims
    activations_stem = config.switch_prior_model_activations

    dim_in_dist_params = dims_stem[-1] if len(dims_stem) > 0 else dim_in
    return dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params


class SwitchPriorModelCategorical(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            dim_in,
            dim_out,
            dims_stem,
            activations_stem,
            dim_in_dist_params,
        ) = _extract_dims_from_cfg(config=config)
        if dim_in is None:
            self.dist = ParametrisedOneHotCategorical(
                logits=torch.zeros(dim_out), requires_grad=True,
            )
        else:
            self.dist = ParametrisedConditionalDistribution(
                stem=MLP(
                    dim_in=dim_in,
                    dims=dims_stem,
                    activations=activations_stem,
                ),
                dist_params=nn.ModuleDict(
                    {
                        "logits": nn.Sequential(
                            Linear(
                                in_features=dim_in_dist_params,
                                out_features=dim_out,
                            ),
                            # SigmoidLimiter(limits=[-10, 10]),
                        )
                    }
                ),
                dist_cls=OneHotCategorical,
            )

    def forward(self, *args, **kwargs):
        return self.dist(*args, **kwargs)


class SwitchPriorModelGaussian(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            dim_in,
            dim_out,
            dims_stem,
            activations_stem,
            dim_in_dist_params,
        ) = _extract_dims_from_cfg(config=config)
        if dim_in is None:
            covariance_matrix = (
                (config.switch_prior_scale ** 2) or 1.0
            ) * torch.eye(dim_out)
            LVinv_tril, LVinv_logdiag = make_inv_tril_parametrization(
                covariance_matrix
            )
            self.dist = ParametrisedMultivariateNormal(
                m=torch.ones(config.dims.switch) * config.switch_prior_loc,
                LVinv_tril=LVinv_tril,
                LVinv_logdiag=LVinv_logdiag,
                requires_grad_m=config.requires_grad_switch_prior,
                requires_diag_LVinv_tril=False,
                requires_diag_LVinv_logdiag=config.requires_grad_switch_prior,
            )
        else:
            self.dist = ParametrisedConditionalDistribution(
                stem=MLP(
                    dim_in=dim_in,
                    dims=dims_stem,
                    activations=activations_stem,
                ),
                dist_params=nn.ModuleDict(
                    {
                        "loc": nn.Sequential(
                            Linear(dim_in_dist_params, dim_out),
                        ),
                        "scale_tril": DefaultScaleTransform(
                            dim_in_dist_params, dim_out,
                        ),
                    }
                ),
                dist_cls=MultivariateNormal,
            )

    def forward(self, *args, **kwargs):
        return self.dist(*args, **kwargs)
