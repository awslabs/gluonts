import torch
from torch import nn
from torch.distributions import (
    OneHotCategorical,
    MultivariateNormal,
)
from torch_extensions.distributions.conditional_parametrised_distribution import (
    ParametrisedConditionalDistribution,
)
from torch_extensions.mlp import MLP
from torch_extensions.batch_diag_matrix import BatchDiagMatrix
from torch_extensions.affine import Bias
from torch_extensions.constant import Constant
from utils.utils import SigmoidLimiter


def _extract_dims_from_cfg(config):
    if config.dims.ctrl_switch is not None:
        dim_in = config.dims.switch + config.dims.ctrl_switch
    else:
        dim_in = config.dims.switch
    dim_out = config.dims.switch
    dims_stem = config.switch_transition_model_dims
    activations_stem = config.switch_transition_model_activations

    dim_in_dist_params = dims_stem[-1] if len(dims_stem) > 0 else dim_in
    return dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params


class SwitchTransitionBase(nn.Module):
    def forward(
            self, controls: torch.Tensor, switch: torch.Tensor
    ) -> torch.distributions.Distribution:
        raise NotImplementedError()


class SwitchTransitionModelCategorical(SwitchTransitionBase):
    def __init__(self, config):
        super().__init__()
        (
            dim_in,
            dim_out,
            dims_stem,
            activations_stem,
            dim_in_dist_params,
        ) = _extract_dims_from_cfg(config)
        dim_in_dist_params = dims_stem[-1] if len(dims_stem) > 0 else dim_in
        self.conditional_dist = ParametrisedConditionalDistribution(
            stem=MLP(
                dim_in=dim_in,
                dims=dims_stem,
                activations=activations_stem,
            ),
            dist_params=nn.ModuleDict(
                {
                    "logits": nn.Sequential(
                        nn.Linear(dim_in_dist_params, dim_out),
                        SigmoidLimiter(limits=[-10, 10]),
                    )
                }
            ),
            dist_cls=OneHotCategorical,
        )

    def forward(self, controls, switch):
        h = torch.cat((controls, switch), dim=-1) \
            if controls is not None \
            else switch
        switch_to_switch_dist = self.conditional_dist(h)
        return switch_to_switch_dist


class SwitchTransitionModelGaussian(SwitchTransitionBase):
    def __init__(self, config):
        super().__init__()
        (
            dim_in,
            dim_out,
            dims_stem,
            activations_stem,
            dim_in_dist_params,
        ) = _extract_dims_from_cfg(config)

        self.conditional_dist = ParametrisedConditionalDistribution(
            stem=MLP(
                dim_in=dim_in,
                dims=dims_stem,
                activations=activations_stem,
            ),
            dist_params=nn.ModuleDict(
                {
                    "loc": nn.Sequential(
                        nn.Linear(dim_in_dist_params, dim_out),
                    ),
                    "scale_tril": nn.Sequential(
                        nn.Linear(dim_in_dist_params, dim_out),
                        # TODO: hard-coded const for small initial scale
                        # Lambda(fn=lambda x: x - 4.0),
                        Bias(loc=-4.0),
                        nn.Softplus(),
                        Bias(loc=1e-6),  # FP64
                        BatchDiagMatrix(),
                    ),
                }
            ),
            dist_cls=MultivariateNormal,
        )

    def forward(self, controls, switch):
        h = torch.cat((controls, switch), dim=-1) if controls is not None else switch
        return self.conditional_dist(h)


class SwitchTransitionModelGaussianDirac(SwitchTransitionBase):
    """
    The output of this transition function is a Gaussian with zero variance.
    This is intended to be used with State-to-Switch recurrence,
    which already has a set of noise covariance base matrices
    and does not necessarily need a second noise source.
    """
    def __init__(self, config):
        super().__init__()
        (
            dim_in,
            dim_out,
            dims_stem,
            activations_stem,
            dim_in_dist_params,
        ) = _extract_dims_from_cfg(config)

        self.conditional_dist = ParametrisedConditionalDistribution(
            stem=MLP(
                dim_in=dim_in,
                dims=dims_stem,
                activations=activations_stem,
            ),
            dist_params=nn.ModuleDict(
                {
                    "loc": nn.Sequential(
                        nn.Linear(dim_in_dist_params, dim_out),
                    ),
                    "scale_tril": Constant(
                        val=0,
                        shp_append=(dim_out, dim_out),
                        n_dims_from_input=-1,  # x.shape[:-1]
                    ),
                }
            ),
            dist_cls=MultivariateNormal,
        )

    def forward(self, controls, switch):
        h = torch.cat((controls, switch), dim=-1) if controls is not None else switch
        return self.conditional_dist(h)
