from models.gls_parameters.state_to_switch_parameters import (
    StateToSwitchParams,
)


class StateToSwitchParamsDefault(StateToSwitchParams):
    def __init__(self, config, switch_link=None):
        n_state, n_switch = config.dims.state, config.dims.switch
        n_base_F, n_base_S = config.n_base_F, config.n_base_S
        init_scale_S_diag = config.init_scale_S_diag

        # switch_link may be provided (to be shared).
        if switch_link is None:
            switch_link_type = config.recurrent_link_type
            switch_link_dims_hidden = config.switch_link_dims_hidden
            switch_link_activations = config.switch_link_activations
        else:
            switch_link_type = None
            switch_link_dims_hidden = None
            switch_link_activations = None

        super().__init__(
            n_switch=n_switch,
            n_state=n_state,
            n_base_F=n_base_F,
            n_base_S=n_base_S,
            init_scale_S_diag=init_scale_S_diag,
            switch_link=switch_link,
            switch_link_type=switch_link_type,
            switch_link_dims_hidden=switch_link_dims_hidden,
            switch_link_activations=switch_link_activations,
            F_scaling=config.F_scaling,
            LSinv_logdiag_scaling=config.LSinv_logdiag_scaling,
            requires_grad_S=config.requires_grad_S,
        )
