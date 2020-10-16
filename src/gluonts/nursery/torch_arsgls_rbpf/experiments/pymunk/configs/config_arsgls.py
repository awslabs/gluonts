import math
import numpy as np
from torch import nn

import consts
from utils.utils import TensorDims
from experiments.base_config import SwitchLinkType
from experiments.pymunk.configs.configs_base import PymunkASGLSConfig


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
        auxiliary=6,
        ctrl_target=None,
        ctrl_state=None,
    ),
    #
    batch_size_eval=8,
    num_samples_eval=256,
    prediction_length=40,
    n_epochs=600,
    lr=2e-3,
    lr_decay_rate=0.85,
    lr_decay_steps=20,
    grad_clip_norm=1500.0,
    weight_decay=0,
    n_epochs_no_resampling=0,
    n_epochs_freeze_gls_params=40,
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
    LRinv_logdiag_scaling=10.0,
    LQinv_logdiag_scaling=10.0,
    A_scaling=10.0,
    B_scaling=10.0,
    C_scaling=10.0,
    D_scaling=10.0,
    LSinv_logdiag_scaling=10.0,
    F_scaling=10.0,
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
