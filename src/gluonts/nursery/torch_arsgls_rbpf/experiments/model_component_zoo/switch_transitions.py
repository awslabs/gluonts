import torch
from torch import nn
from torch.distributions import OneHotCategorical, MultivariateNormal
from inference.analytical_gausian_linear.inference_step import \
    filter_forward_prediction_step
from models.gls_parameters.state_to_switch_parameters import StateToSwitchParams
from torch_extensions.distributions.conditional_parametrised_distribution import \
    ParametrisedConditionalDistribution
from torch_extensions.mlp import MLP
from torch_extensions.ops import batch_diag_matrix, matvec
from torch_extensions.recurrent_transition import GaussianRecurrentTransition
from utils.utils import SigmoidLimiter, Lambda


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


class SwitchTransitionModelCategorical(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params = \
            _extract_dims_from_cfg(config)
        dim_in_dist_params = dims_stem[-1] if len(dims_stem) > 0 else dim_in
        self.conditional_dist = ParametrisedConditionalDistribution(
            stem=MLP(
                dim_in=dim_in,
                dims_hidden=dims_stem,
                activations=activations_stem,
            ),
            dist_params=nn.ModuleDict({
                "logits": nn.Sequential(
                    nn.Linear(dim_in_dist_params, dim_out),
                    SigmoidLimiter(limits=[-10, 10]),
                )
            }),
            dist_cls=OneHotCategorical,
        )

    def forward(self, u, s, x=None, m=None, V=None):
        h = torch.cat((u, s), dim=-1) if u is not None else s
        switch_to_switch_dist = self.conditional_dist(h)
        return switch_to_switch_dist


class SwitchTransitionModelGaussian(nn.Module):
    def __init__(self, config):
        """ due to parameter sharing we cannot instantiate this one entirely from config file. """
        super().__init__()
        dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params = \
            _extract_dims_from_cfg(config)
        n_state, n_switch, is_recurrent = config.dims.state, config.dims.switch, config.is_recurrent
        self.conditional_dist = GaussianRecurrentTransition(
            conditional_dist_tranform=ParametrisedConditionalDistribution(
                stem=MLP(
                    dim_in=dim_in,
                    dims_hidden=dims_stem,
                    activations=activations_stem,
                ),
                dist_params=nn.ModuleDict({
                    "loc": nn.Sequential(
                        nn.Linear(dim_in_dist_params, dim_out),
                    ),
                    "scale_tril": nn.Sequential(
                        nn.Linear(dim_in_dist_params, dim_out),
                        Lambda(fn=lambda x: x - 4.0),
                        # start out with small variances.
                        nn.Softplus(),
                        Lambda(fn=lambda x: x + 1e-6),  # FP64
                        Lambda(fn=batch_diag_matrix),
                    ),
                }),
                dist_cls=MultivariateNormal,
            ),
            n_state=n_state,
            n_switch=n_switch,
            is_recurrent=is_recurrent,
        )

    def forward(self, u, s, x=None, m=None, V=None):
        return self.conditional_dist(u=u, s=s, x=x, m=m, V=V)


class SwitchTransitionModelGaussianRecurrentBaseMat(nn.Module):
    def __init__(self, config, switch_link=None):
        super().__init__()
        dim_in, dim_out, dims_stem, activations_stem, dim_in_dist_params = \
            _extract_dims_from_cfg(config)
        activations_stem = (activations_stem,) if isinstance(
            activations_stem, nn.Module) else activations_stem
        n_state, n_switch = config.dims.state, config.dims.switch
        self.is_recurrent = config.is_recurrent

        n_base_F, n_base_S = config.n_base_F, config.n_base_S
        init_scale_S_diag = config.init_scale_S_diag
        switch_link_type = config.recurrent_link_type if switch_link is None else None

        self.base_parameters = StateToSwitchParams(
            n_switch=n_switch,
            n_state=n_state,
            n_base_F=n_base_F,
            n_base_S=n_base_S,
            init_scale_S_diag=init_scale_S_diag,
            switch_link=switch_link,
            switch_link_type=switch_link_type,
        )
        self.transform = MLP(
            dim_in=dim_in,
            dims_hidden=dims_stem + (dim_out,),
            activations=activations_stem + (None,),
        )

    def forward(self, u, s, x=None, m=None, V=None):
        assert len({(x is None), (m is None and V is None)}) == 2
        base_params = self.base_parameters(switch=s)
        F, S = base_params.F, base_params.S
        if not self.is_recurrent:
            F *= 0
        h = torch.cat((u, s), dim=-1) if u is not None else s
        b = self.transform(h)

        if m is not None:  # marginalise
            mp, Vp = filter_forward_prediction_step(m=m, V=V, A=F, R=S, b=b)
        else:  # single sample fwd
            mp, Vp = matvec(F, x) + b, S
        return MultivariateNormal(loc=mp, covariance_matrix=Vp)
