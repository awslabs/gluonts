from dataclasses import dataclass
import torch
from torch import nn
from experiments.base_config import BaseConfig, SwitchLinkType
from utils.utils import TensorDims
from experiments.model_component_zoo import (
    gls_parameters,
    switch_transitions,
    switch_priors,
    state_priors,
    encoders,
)
from models.switching_gaussian_linear_system import (
    RecurrentSwitchingLinearDynamicalSystem,
)
from models_new_will_replace.dynamical_system import ControlInputs


@dataclass()
class PendulumSGLSConfig(BaseConfig):
    n_steps_forecast: int
    switch_prior_model_dims: tuple
    switch_prior_model_activations: (nn.Module, tuple)
    switch_transition_model_dims: tuple
    switch_transition_model_activations: (nn.Module, tuple)
    state_to_switch_encoder_dims: tuple
    state_to_switch_encoder_activations: nn.Module
    obs_to_switch_encoder_dims: tuple
    obs_to_switch_encoder_activations: nn.Module
    n_epochs_until_validate_loss: int
    recurrent_link_type: SwitchLinkType
    is_recurrent: bool

    obs_to_switch_encoder: bool
    state_to_switch_encoder: bool

    switch_prior_loc: float
    switch_prior_scale: float

    gpus: (tuple, list)
    dtype: torch.dtype

    b_fn_dims: tuple
    b_fn_activations: (nn.Module, tuple)
    d_fn_dims: tuple
    d_fn_activations: (nn.Module, tuple)

    n_epochs_no_resampling: int


dims = TensorDims(
    timesteps=50,
    particle=64,
    batch=100,
    state=3,
    obs=2,
    switch=5,
    auxiliary=None,
    ctrl_obs=None,
    ctrl_state=None,
)
config = PendulumSGLSConfig(
    dataset_name="pendulum",
    experiment_name="default",
    gpus=tuple(range(0, 4)),
    dtype=torch.float64,
    lr=1e-2,
    n_epochs=50,
    n_epochs_no_resampling=5,
    n_epochs_until_validate_loss=1,
    n_steps_forecast=100,
    dims=dims,
    init_scale_A=1.0,
    init_scale_B=None,
    init_scale_C=1.0,
    init_scale_D=None,
    init_scale_R_diag=[1e-4, 1e0],
    init_scale_Q_diag=[1e-4, 1e-1],
    init_scale_S_diag=[1e-4, 1e-1],
    #
    obs_to_switch_encoder=False,
    state_to_switch_encoder=True,
    #
    switch_transition_model_dims=tuple((32,)),
    state_to_switch_encoder_dims=(32,),
    obs_to_switch_encoder_dims=tuple(),
    switch_prior_model_dims=tuple(),
    switch_transition_model_activations=nn.LeakyReLU(0.1, inplace=True),
    state_to_switch_encoder_activations=nn.LeakyReLU(0.1, inplace=True),
    obs_to_switch_encoder_activations=nn.LeakyReLU(0.1, inplace=True),
    switch_prior_model_activations=nn.LeakyReLU(0.1, inplace=True),
    #
    switch_link_type=SwitchLinkType.individual,
    recurrent_link_type=SwitchLinkType.shared,
    is_recurrent=True,
    n_base_A=10,
    n_base_B=None,
    n_base_C=10,
    n_base_D=None,
    n_base_Q=10,
    n_base_R=10,
    n_base_F=10,
    n_base_S=10,  # 10,
    state_prior_scale=1.0,
    state_prior_loc=0.0,
    switch_prior_loc=0.0,
    switch_prior_scale=1.0,
    b_fn_dims=tuple(),
    b_fn_activations=nn.LeakyReLU(0.1, inplace=True),
    d_fn_dims=tuple(),
    d_fn_activations=nn.LeakyReLU(0.1, inplace=True),
    requires_grad_Q=True,
    requires_grad_R=True,
)


def make_model(config):
    dims = config.dims
    gls_base_parameters = gls_parameters.GlsParametersUnrestricted(
        config=config
    )
    input_transformer = lambda *args, **kwargs: ControlInputs(
        None, None, None, None
    )
    obs_to_switch_encoder = (
        encoders.ObsToSwitchEncoderGaussianMLP(config=config)
        if config.obs_to_switch_encoder
        else None
    )
    state_to_switch_encoder = (
        encoders.StateToSwitchEncoderGaussianMLP(config=config)
        if config.state_to_switch_encoder
        else None
    )
    state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
    switch_transition_model = switch_transitions.SwitchTransitionModelGaussianRecurrentBaseMat(
        config=config
    )
    switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
    model = RecurrentSwitchingLinearDynamicalSystem(
        n_state=dims.state,
        n_obs=dims.target,
        n_ctrl_state=dims.ctrl_state,
        n_particle=dims.particle,
        n_switch=dims.switch,
        gls_base_parameters=gls_base_parameters,
        input_transformer=input_transformer,
        obs_to_switch_encoder=obs_to_switch_encoder,
        state_to_switch_encoder=state_to_switch_encoder,
        switch_transition_model=switch_transition_model,
        state_prior_model=state_prior_model,
        switch_prior_model=switch_prior_model,
    )
    return model
