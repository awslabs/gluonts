from dataclasses import dataclass
import torch
from torch import nn

import consts
from experiments.base_config import BaseConfig, SwitchLinkType
from utils.utils import TensorDims
from experiments.model_component_zoo import (
    switch_priors,
    state_priors,
    encoders,
)
from models_new_will_replace.rsgls_rbpf import (
    RecurrentSwitchingGaussianLinearSystemRBSMC,
)
from models_new_will_replace.sgls_rbpf import (
    SwitchingGaussianLinearSystemBaseRBSMC,
)
from experiments_new_will_replace.model_component_zoo import (
    switch_transitions,
    gls_parameters,
)
from experiments_new_will_replace.model_component_zoo\
    .recurrent_base_parameters import StateToSwitchParamsDefault
from experiments_new_will_replace.pendulum.pendulum_rbsmc_model \
    import PendulumModel


@dataclass()
class PendulumSGLSConfig(BaseConfig):
    prediction_length: int
    batch_size_eval: int
    num_samples_eval: int
    switch_prior_model_dims: tuple
    switch_prior_model_activations: (nn.Module, tuple)
    switch_transition_model_dims: tuple
    switch_transition_model_activations: (nn.Module, tuple)
    # state_to_switch_encoder_dims: tuple
    # state_to_switch_encoder_activations: nn.Module
    obs_to_switch_encoder_dims: tuple
    obs_to_switch_encoder_activations: nn.Module
    n_epochs_until_validate_loss: int
    recurrent_link_type: SwitchLinkType
    is_recurrent: bool

    # obs_to_switch_encoder: bool
    # state_to_switch_encoder: bool

    switch_prior_loc: float
    switch_prior_scale: float

    gpus: (tuple, list)
    dtype: torch.dtype

    b_fn_dims: tuple
    b_fn_activations: (nn.Module, tuple)
    d_fn_dims: tuple
    d_fn_activations: (nn.Module, tuple)

    n_epochs_no_resampling: int
    weight_decay: float
    grad_clip_norm: float


dims = TensorDims(
    timesteps=50,
    particle=64,
    batch=100,
    state=3,
    target=2,
    switch=5,
    auxiliary=None,
    ctrl_target=None,
    ctrl_state=None,
)
config = PendulumSGLSConfig(
    dataset_name=consts.Datasets.pendulum_3D_coord,
    experiment_name="default",
    gpus=tuple(range(0, 4)),
    dtype=torch.float64,
    batch_size_eval=1000,
    num_samples_eval=100,
    lr=1e-2,
    weight_decay=1e-5,
    grad_clip_norm=10.0,
    n_epochs=50,
    n_epochs_no_resampling=5,
    n_epochs_until_validate_loss=1,
    prediction_length=100,
    dims=dims,
    init_scale_A=1.0,
    init_scale_B=None,
    init_scale_C=1.0,
    init_scale_D=None,
    init_scale_R_diag=[1e-4, 1e0],
    init_scale_Q_diag=[1e-4, 1e-1],
    init_scale_S_diag=[1e-4, 1e-1],
    #
    switch_transition_model_dims=tuple((32,)),
    # state_to_switch_encoder_dims=(32,),
    obs_to_switch_encoder_dims=tuple(),
    switch_prior_model_dims=tuple(),
    switch_transition_model_activations=nn.LeakyReLU(0.1, inplace=True),
    # state_to_switch_encoder_activations=nn.LeakyReLU(0.1, inplace=True),
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

    # obs_to_switch_encoder = (
    #     encoders.ObsToSwitchEncoderGaussianMLP(config=config)
    #     if config.obs_to_switch_encoder
    #     else None
    # )
    encoder = encoders.ObsToSwitchEncoderGaussianMLP(config=config)
    state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
    switch_transition_model = switch_transitions.SwitchTransitionModelGaussianDirac(
        config=config,
    )
    switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
    recurrent_base_parameters = StateToSwitchParamsDefault(config=config)

    ssm = RecurrentSwitchingGaussianLinearSystemRBSMC(
        n_state=dims.state,
        n_target=dims.target,
        n_ctrl_state=dims.ctrl_state,
        n_ctrl_target=dims.ctrl_target,
        n_particle=dims.particle,
        n_switch=dims.switch,
        gls_base_parameters=gls_base_parameters,
        recurrent_base_parameters=recurrent_base_parameters,
        encoder=encoder,
        switch_transition_model=switch_transition_model,
        state_prior_model=state_prior_model,
        switch_prior_model=switch_prior_model,
    )
    model = PendulumModel(
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

