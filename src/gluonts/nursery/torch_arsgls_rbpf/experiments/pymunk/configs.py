import math
from typing import Tuple
from dataclasses import dataclass, asdict
import numpy as np
from torch import nn
from experiments.base_config import BaseConfig
from utils.utils import TensorDims
from experiments.model_component_zoo import gls_parameters, encoders, decoders, \
    state_priors, switch_transitions, switch_priors
from models.kalman_variational_autoencoder import KalmanVariationalAutoEncoder
from models.auxiliary_switching_gaussian_linear_system import \
    RecurrentAuxiliarySwitchingLinearDynamicalSystem
from experiments.base_config import SwitchLinkType
from torch_extensions.layers_with_init import LSTM, Linear


@dataclass()
class PymunkConfig(BaseConfig):
    dims_img: Tuple[int, int, int]
    dims_filter: Tuple[int]
    kernel_sizes: Tuple[int]
    strides: Tuple[int]
    paddings: Tuple[int]
    upscale_factor: int
    prediction_length: int
    decay_steps: int
    lr_decay_rate: float
    grad_clip_norm: float
    batch_size_test: int


@dataclass()
class PymunkKVAEConfig(PymunkConfig):
    n_hidden_rnn: int
    reconstruction_weight: float
    rao_blackwellized: bool


@dataclass()
class PymunkASGLSConfig(PymunkConfig):
    recurrent_link_type: SwitchLinkType
    b_fn_dims: tuple
    b_fn_activations: (nn.Module, tuple)
    d_fn_dims: tuple
    d_fn_activations: (nn.Module, tuple)
    switch_prior_model_dims: tuple
    switch_prior_model_activations: (nn.Module, tuple)
    switch_transition_model_dims: tuple
    switch_transition_model_activations: (nn.Module, tuple)
    dims_encoder: Tuple[tuple]
    activations_encoder: Tuple[tuple]
    is_recurrent: bool
    switch_prior_scale: float
    switch_prior_loc: float


dims_img = (1, 32, 32)
dims_kvae = TensorDims(
    timesteps=20,
    particle=1,
    batch=32,
    state=4,
    obs=int(np.prod(dims_img)),
    switch=None,  # --> n_hidden_rnn
    auxiliary=2,
    ctrl_obs=None,
    ctrl_state=None,
)
dims_asgls = TensorDims(
    timesteps=20,
    particle=32,
    batch=33,
    state=10,  # 10
    obs=int(np.prod(dims_img)),
    switch=5,
    auxiliary=2,
    ctrl_obs=None,
    ctrl_state=None,
)

base_config = PymunkConfig(
    dataset_name="box",  # "box_gravity" "polygon" "pong"
    experiment_name="arsgls",  # kvae
    dims=None,  # dummy
    batch_size_test=3,
    prediction_length=40,
    n_epochs=200,
    lr=7e-3,
    lr_decay_rate=0.85,
    decay_steps=20,
    grad_clip_norm=150.0,
    n_base_A=10,  # 10
    n_base_B=None,
    # Not used in image environments. But consider for other data.
    n_base_C=10,  # 10
    n_base_D=None,
    n_base_Q=10,  # 10
    n_base_R=10,  # 10
    n_base_S=10,  # 10
    n_base_F=10,  # 10
    requires_grad_R=False,
    requires_grad_Q=False,
    init_scale_A=1.0,
    init_scale_B=0.05,
    init_scale_C=0.05,
    init_scale_D=None,  # Note: KVAE does not have D.
    init_scale_R_diag=math.sqrt(0.08),
    init_scale_Q_diag=math.sqrt(0.03),
    init_scale_S_diag=(1e-4, 1e-1),
    # only ASGLS. but messed up base config using this...
    state_prior_loc=0.0,
    state_prior_scale=math.sqrt(20.0),
    #
    dims_img=dims_img,
    dims_filter=(32,) * 3,
    kernel_sizes=(3,) * 3,
    strides=(2,) * 3,
    paddings=(1,) * 3,
    upscale_factor=2,
    #
    switch_link_type=SwitchLinkType.shared,
)

kvae_config = PymunkKVAEConfig(
    **asdict(base_config),
    rao_blackwellized=False,
    reconstruction_weight=0.3,
    n_hidden_rnn=50,  # the state output corresponds to our switch.
)
kvae_config.dims = dims_kvae

asgls_config = PymunkASGLSConfig(
    **asdict(base_config),
    recurrent_link_type=SwitchLinkType.shared,
    b_fn_dims=tuple(),
    d_fn_dims=tuple(),
    b_fn_activations=None,
    d_fn_activations=None,
    switch_prior_model_dims=tuple(),
    switch_prior_model_activations=None,
    switch_transition_model_dims=(32,),
    switch_transition_model_activations=nn.ReLU(),
    dims_encoder=(None, (64,)),
    activations_encoder=((None,), (nn.ReLU(),)),
    is_recurrent=True,
    switch_prior_scale=1.0,
    switch_prior_loc=0.0,
)
asgls_config.dims = dims_asgls

# maybe change the following:
if True:
    asgls_config.requires_grad_R = True
    asgls_config.requires_grad_Q = True
    asgls_config.init_scale_R_diag = [1e-3, 1e-1]
    asgls_config.init_scale_Q_diag = [1e-3, 1e-1]
    asgls_config.init_scale_S_diag = [1e-3, 1e-1]
    # asgls_config.state_prior_loc = 0.0
    # asgls_config.state_prior_scale = math.sqrt(1.0)


def make_kvae(config):
    gls_base_params = gls_parameters.GLSParametersKVAE(config=config)
    decoder = decoders.AuxiliaryToObsDecoderConvBernoulli(config=config)
    encoder = encoders.ObsToAuxiliaryEncoderConvGaussian(config=config)
    state_prior_model = state_priors.StatePriorModeFixedlNoInputs(config=config)
    rnn = LSTM(input_size=config.dims.auxiliary,
               hidden_size=config.n_hidden_rnn)

    model = KalmanVariationalAutoEncoder(
        n_state=config.dims.state,
        n_obs=config.dims.obs,
        n_auxiliary=config.dims.auxiliary,
        n_ctrl_state=config.dims.ctrl_state,
        n_particle=config.dims.particle,
        gls_base_parameters=gls_base_params,
        measurement_model=decoder,
        obs_to_auxiliary_encoder=encoder,
        rnn_switch_model=rnn,
        state_prior_model=state_prior_model,
        reconstruction_weight=config.reconstruction_weight,
    )
    return model


def make_asgls(config):
    dims = config.dims
    input_transformer = None
    gls_base_parameters = gls_parameters.GLSParametersASGLS(config=config)
    switch_transition_model = switch_transitions.SwitchTransitionModelGaussianRecurrentBaseMat(
        config=config)
    state_prior_model = state_priors.StatePriorModeFixedlNoInputs(config=config)
    switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
    measurment_model = decoders.AuxiliaryToObsDecoderConvBernoulli(
        config=config)
    obs_encoder = encoders.ObsToAuxiliaryLadderEncoderConvMlpGaussian(
        config=config)

    model = RecurrentAuxiliarySwitchingLinearDynamicalSystem(
        n_state=dims.state,
        n_obs=dims.obs,
        n_ctrl_state=dims.ctrl_state,
        n_particle=dims.particle,
        n_switch=dims.switch,
        gls_base_parameters=gls_base_parameters,
        measurement_model=measurment_model,
        obs_encoder=obs_encoder,
        input_transformer=input_transformer,
        switch_transition_model=switch_transition_model,
        state_prior_model=state_prior_model,
        switch_prior_model=switch_prior_model,
    )
    return model
