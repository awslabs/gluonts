import torch
from torch import nn
from torch.distributions import OneHotCategorical, MultivariateNormal
from inference.analytical_gausian_linear.inference_step import (
    filter_forward_prediction_step,
)
from models_new_will_replace.gls_parameters.state_to_switch_parameters import (
    StateToSwitchParams,
)
from torch_extensions.distributions.conditional_parametrised_distribution import (
    ParametrisedConditionalDistribution,
)
from torch_extensions.mlp import MLP
from torch_extensions.ops import batch_diag_matrix, matvec
from utils.utils import SigmoidLimiter, Lambda
from models_new_will_replace.sgls_rbpf import GLSVariablesSGLS
from torch_extensions.ops import (
    symmetrize,
    matvec,
    cholesky,
    matmul,
)


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
                        Lambda(fn=lambda x: x - 4.0),
                        nn.Softplus(),
                        Lambda(fn=lambda x: x + 1e-6),  # FP64
                        Lambda(fn=batch_diag_matrix),
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
    Intended to be used with State-to-Switch recurrence, which already has a
    set of noise base matrices and does not necessarily a second noise source.
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
                    "scale_tril": Lambda(
                        fn=lambda h: torch.zeros(
                            h.shape[:-1] + (dim_out, dim_out,),
                            device=h.device, dtype=h.dtype,
                        )
                    ),
                }
            ),
            dist_cls=MultivariateNormal,
        )

    def forward(self, controls, switch):
        h = torch.cat((controls, switch), dim=-1) if controls is not None else switch
        return self.conditional_dist(h)


# class StateToSwitchBaseMat(nn.Module):
#     def __init__(self, config, switch_link=None):
#         super().__init__()
#
#         n_state, n_switch = config.dims.state, config.dims.switch
#         n_base_F, n_base_S = config.n_base_F, config.n_base_S
#         init_scale_S_diag = config.init_scale_S_diag
#
#         switch_link_type = (
#             config.recurrent_link_type if switch_link is None else None
#         )
#
#         self.base_parameters = StateToSwitchParams(
#             n_switch=n_switch,
#             n_state=n_state,
#             n_base_F=n_base_F,
#             n_base_S=n_base_S,
#             init_scale_S_diag=init_scale_S_diag,
#             switch_link=switch_link,
#             switch_link_type=switch_link_type,
#         )
#
#     def forward(self, lat_vars: GLSVariablesSGLS) -> MultivariateNormal:
#         if len({lat_vars.x is None, lat_vars.m is None}) != 2:
#             raise Exception("Provide samples XOR dist params (-> marginalize).")
#         marginalize_states = lat_vars.m is not None
#
#         if marginalize_states:
#             return self.marginalize(lat_vars=lat_vars)
#         else:
#             return self.conditional(lat_vars=lat_vars)
#
#     @staticmethod
#     def marginalize(
#             lat_vars: GLSVariablesSGLS,
#             transition_matrix: torch.Tensor,
#             covariance_matrix: torch.Tensor,
#     ) -> MultivariateNormal:
#         base_params = self.base_parameters(switch=lat_vars.switch)
#
#         m = matvec(transition_matrix, lat_vars.m)
#         V = matmul(transition_matrix,
#                    matmul(lat_vars.V, transition_matrix.transpose(-1, -2))) \
#             + covariance_matrix
#         return MultivariateNormal(loc=m, scale_tril=cholesky(symmetrize(V)))
#
#     @staticmethod
#     def conditional(
#             lat_vars: GLSVariablesSGLS,
#             transition_matrix: torch.Tensor,
#             covariance_matrix: torch.Tensor,
#     ) -> MultivariateNormal:
#         m = matvec(transition_matrix, lat_vars.x)
#         V = covariance_matrix
#         return MultivariateNormal(loc=m, scale_tril=cholesky(V))

