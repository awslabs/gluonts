import math
from typing import Tuple
from dataclasses import dataclass, asdict
import numpy as np
from torch import nn

import consts
from experiments.base_config import BaseConfig
from utils.utils import TensorDims
from experiments.model_component_zoo import (
    encoders,
    decoders,
    state_priors,
    switch_priors,
)

from experiments_new_will_replace.model_component_zoo import (
    gls_parameters,
    switch_transitions,
)
from experiments_new_will_replace.model_component_zoo.recurrent_base_parameters \
    import StateToSwitchParamsDefault

from models_new_will_replace.kvae import KalmanVariationalAutoEncoder
from models_new_will_replace.arsgls_rbpf import \
    AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC

from experiments.base_config import SwitchLinkType
from torch_extensions.layers_with_init import LSTMCell
from experiments_new_will_replace.pymunk.pymunk_model import PymunkModel


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
    batch_size_eval: int
    weight_decay: float
    num_samples_eval: int
    n_epochs_no_resampling: int


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
    target=int(np.prod(dims_img)),
    switch=None,  # --> n_hidden_rnn
    auxiliary=2,
    ctrl_target=None,
    ctrl_state=None,
)
dims_asgls = TensorDims(
    timesteps=20,
    particle=20,
    batch=32,
    state=10,  # 10
    target=int(np.prod(dims_img)),
    switch=5,
    auxiliary=2,
    ctrl_target=None,
    ctrl_state=None,
)

base_config = PymunkConfig(
    dataset_name=consts.Datasets.box,
    experiment_name="arsgls",  # kvae
    dims=None,  # dummy
    batch_size_eval=10,
    num_samples_eval=100,
    prediction_length=40,
    n_epochs=200,
    lr=7e-3,
    lr_decay_rate=0.85,
    decay_steps=20,
    grad_clip_norm=150.0,
    weight_decay=0.0,
    n_epochs_no_resampling=0,
    n_base_A=10,
    n_base_B=None,
    # Not used in image environments. But consider for other data.
    n_base_C=10,
    n_base_D=None,
    n_base_Q=10,
    n_base_R=10,
    n_base_S=10,
    n_base_F=10,
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

arsgls_config = PymunkASGLSConfig(
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
arsgls_config.dims = dims_asgls

if True:  # well, well! this is just for quick testing :-)
    arsgls_config.requires_grad_R = True
    arsgls_config.requires_grad_Q = True
    arsgls_config.init_scale_R_diag = [1e-3, 1e-1]
    arsgls_config.init_scale_Q_diag = [1e-3, 1e-1]
    arsgls_config.init_scale_S_diag = [1e-3, 1e-1]
    # asgls_config.state_prior_loc = 0.0
    # asgls_config.state_prior_scale = math.sqrt(1.0)


def make_experiment_config(experiment_name, dataset_name):
    if experiment_name == "kvae":
        config = kvae_config
    elif experiment_name == "arsgls":
        config = arsgls_config
    else:
        raise Exception(f"unknown experiment/model: {experiment_name}")
    config.experiment_name = experiment_name
    config.dataset_name = dataset_name
    return config


def _make_kvae(config):
    gls_base_params = gls_parameters.GLSParametersKVAE(config=config)
    decoder = decoders.AuxiliaryToObsDecoderConvBernoulli(config=config)
    encoder = encoders.ObsToAuxiliaryEncoderConvGaussian(config=config)
    state_prior_model = state_priors.StatePriorModeFixedlNoInputs(
        config=config
    )
    rnn = LSTMCell(
        input_size=config.dims.auxiliary, hidden_size=config.n_hidden_rnn,
    )

    ssm = KalmanVariationalAutoEncoder(
        n_state=config.dims.state,
        n_target=config.dims.target,
        n_auxiliary=config.dims.auxiliary,
        n_ctrl_state=config.dims.ctrl_state,
        n_particle=config.dims.particle,
        gls_base_parameters=gls_base_params,
        measurement_model=decoder,
        encoder=encoder,
        rnn_switch_model=rnn,
        state_prior_model=state_prior_model,
        reconstruction_weight=config.reconstruction_weight,
    )
    return ssm


def _make_asgls(config):
    dims = config.dims
    gls_base_parameters = gls_parameters.GLSParametersASGLS(config=config)
    switch_transition_model = switch_transitions.SwitchTransitionModelGaussianDirac(
        config=config
    )
    state_prior_model = state_priors.StatePriorModeFixedlNoInputs(
        config=config
    )
    switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
    measurment_model = decoders.AuxiliaryToObsDecoderConvBernoulli(
        config=config
    )
    obs_encoder = encoders.ObsToAuxiliaryLadderEncoderConvMlpGaussian(
        config=config
    )
    recurrent_base_parameters = StateToSwitchParamsDefault(config=config)

    ssm = AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC(
        n_state=dims.state,
        n_target=dims.target,
        n_ctrl_state=dims.ctrl_state,
        n_ctrl_target=dims.ctrl_target,
        n_particle=dims.particle,
        n_switch=dims.switch,
        gls_base_parameters=gls_base_parameters,
        recurrent_base_parameters=recurrent_base_parameters,
        measurement_model=measurment_model,
        encoder=obs_encoder,
        switch_transition_model=switch_transition_model,
        state_prior_model=state_prior_model,
        switch_prior_model=switch_prior_model,
    )
    return ssm


def make_model(config):
    if config.experiment_name == "kvae":
        ssm = _make_kvae(config=config)
    elif config.experiment_name == "arsgls":
        ssm = _make_asgls(config=config)
    else:
        raise Exception("bad experiment name")

    model = PymunkModel(
        config=config,
        ssm=ssm,
        dataset_name=config.dataset_name,
        lr=config.lr,
        weight_decay=config.weight_decay,
        n_epochs=config.n_epochs,
        batch_sizes={
            "train": config.dims.batch,
            "val": config.batch_size_eval,
            "test": config.batch_size_eval,
        },
        past_length=config.dims.timesteps,
        n_particle_train=config.dims.particle,
        n_particle_eval=config.num_samples_eval,
        prediction_length=config.prediction_length,
        n_epochs_no_resampling=config.n_epochs_no_resampling,
        num_batches_per_epoch=50,
    )
    return model