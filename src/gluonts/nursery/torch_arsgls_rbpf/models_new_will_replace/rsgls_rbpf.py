import torch

from models_new_will_replace.dynamical_system import ControlInputs
from models_new_will_replace.sgls_rbpf import SwitchingGaussianLinearSystemRBSMC, \
    GLSVariablesSGLS
from torch_extensions.distributions.parametrised_distribution import \
    prepend_batch_dims


class RecurrentSwitchingGaussianLinearSystemRBSMC(
    SwitchingGaussianLinearSystemRBSMC
):
    def _make_switch_transition_dist(
        self, lat_vars_tm1: GLSVariablesSGLS, ctrl_t: ControlInputs,
    ) -> torch.distributions.MultivariateNormal:
        # TODO: currently switch_transition_model handles if it
        #  marginalizes or uses state sample.
        #  Ideally the transition model should *be able* to
        #  *let this class* marginalize it,
        #  but the logic should be in this class, prob. in filter_step.
        switch_model_dist = self.switch_transition_model(
            u=prepend_batch_dims(ctrl_t.switch, shp=(self.n_particle,))
            if ctrl_t.switch is not None
            else None,
            s=lat_vars_tm1.switch,
            x=lat_vars_tm1.x,
            m=lat_vars_tm1.m,
            V=lat_vars_tm1.V,
        )
        return switch_model_dist