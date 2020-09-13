from models.gls_parameters.gls_parameters import GLSParameters
from models.gls_parameters.issm_parameters import ISSMParameters
from models.gls_parameters.issm import CompositeISSM, SeasonalityISSM

from torch_extensions.mlp import MLP


class GlsParametersISSM(ISSMParameters):
    def __init__(self, config):
        super().__init__(
            issm=CompositeISSM.get_from_freq(
                freq=config.freq, add_trend=config.add_trend
            ),
            n_state=config.dims.state,
            n_obs=config.dims.target,
            n_ctrl_state=config.dims.ctrl_state,
            n_ctrl_obs=config.dims.ctrl_target,
            n_switch=config.dims.switch,
            n_base_B=config.n_base_B,
            n_base_D=config.n_base_D,
            n_base_R=config.n_base_R,
            n_base_Q=config.n_base_Q,
            switch_link_type=config.switch_link_type,
            switch_link_dims_hidden=config.switch_link_dims_hidden,
            switch_link_activations=config.switch_link_activations,
            make_cov_from_cholesky_avg=config.make_cov_from_cholesky_avg,
            b_fn=MLP(
                dim_in=config.dims.switch,  # f: b(s)
                dims=tuple(config.b_fn_dims) + (config.dims.state,),
                activations=config.b_fn_activations,
            )
            if config.b_fn_dims
            else None,
            d_fn=MLP(
                dim_in=config.dims.switch,  # f: d(s)
                dims=tuple(config.d_fn_dims) + (config.dims.target,),
                activations=config.d_fn_activations,
            )
            if config.d_fn_dims
            else None,
            init_scale_R_diag=config.init_scale_R_diag,
            init_scale_Q_diag=config.init_scale_Q_diag,
            init_scale_A=config.init_scale_A,
            requires_grad_R=True,
            requires_grad_Q=True,
        )


class GlsParametersSeasonalityISSM(ISSMParameters):
    def __init__(self, config):
        super().__init__(
            issm=SeasonalityISSM(n_seasons=config.dims.state),
            n_state=config.dims.state,
            n_obs=config.dims.target,
            n_ctrl_state=config.dims.ctrl_state,
            n_ctrl_obs=config.dims.ctrl_target,
            n_switch=config.dims.switch,
            n_base_B=config.n_base_B,
            n_base_D=config.n_base_D,
            n_base_R=config.n_base_R,
            n_base_Q=config.n_base_Q,
            switch_link_type=config.switch_link_type,
            switch_link_dims_hidden=config.switch_link_dims_hidden,
            switch_link_activations=config.switch_link_activations,
            make_cov_from_cholesky_avg=config.make_cov_from_cholesky_avg,
            b_fn=MLP(
                dim_in=config.dims.switch,  # f: b(s)
                dims=tuple(config.b_fn_dims) + (config.dims.state,),
                activations=config.b_fn_activations,
            )
            if config.b_fn_dims
            else None,
            d_fn=MLP(
                dim_in=config.dims.switch,  # f: d(s)
                dims=tuple(config.d_fn_dims) + (config.dims.target,),
                activations=config.d_fn_activations,
            )
            if config.d_fn_dims
            else None,
            init_scale_R_diag=config.init_scale_R_diag,
            init_scale_Q_diag=config.init_scale_Q_diag,
            init_scale_A=config.init_scale_A,
            requires_grad_R=True,
            requires_grad_Q=True,
        )


class GlsParametersUnrestricted(GLSParameters):
    def __init__(self, config):
        super().__init__(
            n_state=config.dims.state,
            n_obs=config.dims.target,
            n_switch=config.dims.switch,
            n_ctrl_state=config.dims.ctrl_state,
            n_ctrl_obs=config.dims.ctrl_target,
            n_base_A=config.n_base_A,
            n_base_B=config.n_base_B,
            n_base_C=config.n_base_C,
            n_base_D=config.n_base_D,
            n_base_Q=config.n_base_Q,
            n_base_R=config.n_base_R,
            switch_link_type=config.switch_link_type,
            switch_link_dims_hidden=config.switch_link_dims_hidden,
            switch_link_activations=config.switch_link_activations,
            b_fn=MLP(
                dim_in=config.dims.switch,  # f: b(s)
                dims=tuple(config.b_fn_dims) + (config.dims.state,),
                activations=config.b_fn_activations,
            )
            if config.b_fn_dims
            else None,
            d_fn=MLP(
                dim_in=config.dims.switch,  # f: d(s)
                dims=tuple(config.d_fn_dims) + (config.dims.target,),
                activations=config.d_fn_activations,
            )
            if config.d_fn_dims
            else None,
            init_scale_R_diag=config.init_scale_R_diag,
            init_scale_Q_diag=config.init_scale_Q_diag,
            init_scale_A=config.init_scale_A,
            full_cov_R=False,
            full_cov_Q=False,
            requires_grad_R=config.requires_grad_R,
            requires_grad_Q=config.requires_grad_Q,
        )


class GLSParametersKVAE(GLSParameters):
    def __init__(self, config):
        super().__init__(
            n_state=config.dims.state,
            n_obs=config.dims.auxiliary,
            # NOTE: SSM (pseudo) obs are Model auxiliary.
            n_switch=config.n_hidden_rnn,  # alpha in KVAE
            n_ctrl_state=config.dims.ctrl_state,
            n_ctrl_obs=config.dims.ctrl_target,
            n_base_A=config.n_base_A,
            n_base_B=config.n_base_B,
            n_base_C=config.n_base_C,
            n_base_D=None,  # KVAE does not have D
            n_base_Q=config.n_base_Q,  # 1 in KVAE - but consider using it > 1.
            n_base_R=config.n_base_R,  # 1 in KVAE
            switch_link_type=config.switch_link_type,
            switch_link_dims_hidden=config.switch_link_dims_hidden,
            switch_link_activations=config.switch_link_activations,
            # SharedLink in KVAE (except R, Q, but these are const)
            b_fn=None,  # KVAE does not have this
            d_fn=None,  # KVAE does not have this
            init_scale_A=config.init_scale_A,
            init_scale_B=config.init_scale_B,
            init_scale_C=config.init_scale_C,
            init_scale_D=None,  # KVAE does not have D
            init_scale_R_diag=config.init_scale_R_diag,
            init_scale_Q_diag=config.init_scale_Q_diag,
            full_cov_R=False,
            full_cov_Q=False,
            requires_grad_R=config.requires_grad_R,
            requires_grad_Q=config.requires_grad_Q,
        )


class GLSParametersASGLS(GLSParameters):
    def __init__(self, config):
        super().__init__(
            n_state=config.dims.state,
            n_obs=config.dims.auxiliary,
            # NOTE: SSM (pseudo) obs are Model auxiliary.
            n_switch=config.dims.switch,
            n_ctrl_state=config.dims.ctrl_state,
            n_ctrl_obs=config.dims.ctrl_target,
            n_base_A=config.n_base_A,
            n_base_B=config.n_base_B,
            n_base_C=config.n_base_C,
            n_base_D=config.n_base_D,
            n_base_Q=config.n_base_Q,
            n_base_R=config.n_base_R,
            switch_link_type=config.switch_link_type,
            switch_link_dims_hidden=config.switch_link_dims_hidden,
            switch_link_activations=config.switch_link_activations,
            b_fn=MLP(
                dim_in=config.dims.switch,  # f: b(s)
                dims=tuple(config.b_fn_dims) + (config.dims.state,),
                activations=config.b_fn_activations,
            )
            if config.b_fn_dims
            else None,
            d_fn=MLP(
                dim_in=config.dims.switch,  # f: d(s)
                dims=tuple(config.d_fn_dims) + (config.dims.target,),
                activations=config.d_fn_activations,
            )
            if config.d_fn_dims
            else None,
            init_scale_A=config.init_scale_A,
            init_scale_B=config.init_scale_B,
            init_scale_C=config.init_scale_C,
            init_scale_D=config.init_scale_D,
            init_scale_R_diag=config.init_scale_R_diag,
            init_scale_Q_diag=config.init_scale_Q_diag,
            full_cov_R=False,
            full_cov_Q=False,
            requires_grad_R=config.requires_grad_R,
            requires_grad_Q=config.requires_grad_Q,
        )
