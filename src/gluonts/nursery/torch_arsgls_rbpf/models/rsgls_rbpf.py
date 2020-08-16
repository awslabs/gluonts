import torch
from torch.distributions import MultivariateNormal

from models.sgls_rbpf import ControlInputsSGLS
from models.sgls_rbpf import SwitchingGaussianLinearSystemBaseRBSMC, \
    GLSVariablesSGLS
from torch_extensions.distributions.parametrised_distribution import \
    prepend_batch_dims
from models.gls_parameters.state_to_switch_parameters import (
    StateToSwitchParams,
)
from torch_extensions.fusion import gaussian_linear_combination
from inference.analytical_gausian_linear.inference_step import \
    filter_forward_prediction_step
from torch_extensions.ops import matvec


class RecurrentMixin:
    """
    Mixin to avoid diamond multiple inheritance.
    To be used for SGLS or ASGLS. Overrides switch transition distribution.
    """
    def __init__(self, recurrent_base_parameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recurrent_base_parameters = recurrent_base_parameters

    def _make_switch_transition_dist(
        self, lat_vars_tm1: GLSVariablesSGLS, ctrl_t: ControlInputsSGLS,
    ) -> torch.distributions.MultivariateNormal:
        """
        Compute p(s_t | s_{t-1}) = \int p(x_{t-1}, s_t | s_{t-1}) dx_{t-1}
        = \int p(s_t | s_{t-1}, x_{t-1}) N(x_{t-1} | s_{t-1}) dx_{t-1}.
        We use an additive structure, resulting in a convolution of PDFs, i.e.
        i) the conditional from the switch-to-switch transition and
        ii) the marginal from the state-switch-transition (state marginalised).
        The Gaussian is a stable distribution -> The sum of Gaussian variables
        (not PDFs!) is Gaussian, for which locs and covariances are the summed.
        """
        if len({lat_vars_tm1.x is None, lat_vars_tm1.m is None}) != 2:
            raise Exception("Provide samples XOR dist params (-> marginalize).")
        marginalize_states = lat_vars_tm1.m is not None

        # i) switch-to-switch conditional
        switch_to_switch_dist = super()._make_switch_transition_dist(
            lat_vars_tm1=lat_vars_tm1,
            ctrl_t=ctrl_t,
        )

        # ii) state-to-switch
        rec_base_params = self.recurrent_base_parameters(switch=lat_vars_tm1.switch)
        if marginalize_states:
            m, V = filter_forward_prediction_step(
                m=lat_vars_tm1.m,
                V=lat_vars_tm1.V,
                A=rec_base_params.F,
                R=rec_base_params.S,
                b=None,
            )
        else:
            m = matvec(rec_base_params.F, lat_vars_tm1.x)
            V = rec_base_params.S
        state_to_switch_dist = MultivariateNormal(loc=m, covariance_matrix=V)

        # combine i) & ii): sum variables (=convolve PDFs).
        switch_model_dist = gaussian_linear_combination({
            state_to_switch_dist: 1.0,
            switch_to_switch_dist: 1.0,
        })
        return switch_model_dist


class RecurrentSwitchingGaussianLinearSystemRBSMC(
    RecurrentMixin, SwitchingGaussianLinearSystemBaseRBSMC,
):
    pass
