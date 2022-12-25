from typing import Tuple
from dataclasses import dataclass
import math
import numpy as np
from torch import nn

import consts
from utils.utils import TensorDims
from experiments.base_config import SwitchLinkType
from experiments.pymunk.configs.configs_base import PymunkConfig

from experiments.model_component_zoo import (
    encoders,
    decoders,
    state_priors,
    switch_priors,
    gls_parameters,
    switch_transitions,
)

from experiments.model_component_zoo.recurrent_base_parameters import (
    StateToSwitchParamsDefault,
)

from models.arsgls_rbpf import (
    AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC,
)



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
    requires_grad_switch_prior: bool


dims_img = (1, 32, 32)
config = PymunkASGLSConfig(
    dataset_name=consts.Datasets.box,
    experiment_name="arsgls",
    dims_img=dims_img,
    dims=TensorDims(
        timesteps=20,
        particle=32,
        batch=34,  # pytorch-magma bug with 32.
        state=10,
        target=int(np.prod(dims_img)),
        switch=8,
        auxiliary=2,
        ctrl_target=None,
        ctrl_state=None,
    ),
    #
    batch_size_eval=8,
    num_samples_eval=256,
    prediction_length=40,
    n_epochs=400,
    lr=2e-3,
    lr_decay_rate=0.85,
    lr_decay_steps=20,
    grad_clip_norm=1500.0,
    weight_decay=0,
    n_epochs_no_resampling=0,
    n_epochs_freeze_gls_params=20,
    #
    state_prior_loc=0.0,
    state_prior_scale=math.sqrt(20.0),
    #
    n_base_A=10,
    n_base_B=None,
    n_base_C=10,
    n_base_D=None,
    n_base_Q=10,
    n_base_R=10,
    n_base_S=10,
    n_base_F=10,
    requires_grad_R=True,
    requires_grad_Q=True,
    requires_grad_S=True,
    init_scale_A=1.0,
    init_scale_B=None,  # no controls
    init_scale_C=None,  # variance-scaling init.
    init_scale_D=None,  # no controls
    init_scale_R_diag=[1e-5, 1e-1],
    init_scale_Q_diag=[1e-5, 1e-1],
    init_scale_S_diag=[1e-5, 1e-1],
    eye_init_A=True,
    #
    LRinv_logdiag_scaling=5.0,
    LQinv_logdiag_scaling=5.0,
    A_scaling=5.0,
    B_scaling=5.0,
    C_scaling=5.0,
    D_scaling=5.0,
    LSinv_logdiag_scaling=5.0,
    F_scaling=5.0,
    #
    dims_filter=(32,) * 3,
    kernel_sizes=(3,) * 3,
    strides=(2,) * 3,
    paddings=(1,) * 3,
    upscale_factor=2,
    #
    switch_link_type=SwitchLinkType.individual,
    switch_link_dims_hidden=(64,),
    switch_link_activations=nn.ReLU(),
    #
    # ARSGLS specific
    #
    recurrent_link_type=SwitchLinkType.individual,
    b_fn_dims=tuple(),
    d_fn_dims=tuple(),
    b_fn_activations=None,
    d_fn_activations=None,
    switch_prior_model_dims=tuple(),
    switch_prior_model_activations=None,
    switch_transition_model_dims=(64,),
    switch_transition_model_activations=nn.ReLU(),
    dims_encoder=(None, (64,)),
    activations_encoder=((None,), (nn.ReLU(),)),
    is_recurrent=True,
    switch_prior_scale=1.0,
    switch_prior_loc=0.0,
    requires_grad_switch_prior=False,
)


def _make_asgls(config):
    dims = config.dims
    gls_base_parameters = gls_parameters.GLSParametersASGLS(config=config)
    switch_transition_model = switch_transitions.SwitchTransitionModelGaussianDirac(
        config=config,
    )
    state_prior_model = state_priors.StatePriorModeFixedNoInputs(
        config=config,
    )
    switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
    measurment_model = decoders.AuxiliaryToObsDecoderConvBernoulli(
        config=config,
    )
    encoder = encoders.ObsToAuxiliaryLadderEncoderConvMlpGaussian(
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
        encoder=encoder,
        switch_transition_model=switch_transition_model,
        state_prior_model=state_prior_model,
        switch_prior_model=switch_prior_model,
    )
    return ssm


def make_arsgls_config(dataset_name):
    from copy import deepcopy
    config_cp = deepcopy(config)
    config_cp.dataset_name = dataset_name
    return config_cp
