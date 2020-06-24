from dataclasses import dataclass
import torch
from torch.nn.modules.activation import ReLU
from torch import nn
from experiments.base_config import BaseConfig, SwitchLinkType, TimeFeatType
from utils.utils import TensorDims
from src.data.synthetic_issm import (
    generate_synthetic_issm,
    generate_synthetic_issm_changepoint,
    generate_synthetic_issm_random_changepoint,
)


@dataclass()
class SyntheticIssmSGLSConfig(BaseConfig):
    n_steps_forecast: int
    noise_scale_obs: float
    input_transform_dims: tuple
    input_transform_activations: (nn.Module, tuple)
    switch_prior_model_dims: tuple
    switch_prior_model_activations: (nn.Module, tuple)
    switch_transition_model_dims: tuple
    switch_transition_model_activations: (nn.Module, tuple)
    state_to_switch_encoder_dims: tuple
    state_to_switch_encoder_activations: (nn.Module, tuple)
    obs_to_switch_encoder_dims: tuple
    obs_to_switch_encoder_activations: (nn.Module, tuple)
    b_fn_dims: tuple
    b_fn_activations: (nn.Module, tuple)
    d_fn_dims: tuple
    d_fn_activations: (nn.Module, tuple)
    time_feat: TimeFeatType
    recurrent_link_type: SwitchLinkType
    freq: str
    n_epochs_until_validate_loss: int
    gpus: (list, tuple)
    dtype: torch.dtype
    make_cov_from_cholesky_avg: bool
    n_data_per_group: int
    is_recurrent: bool


experiment_name = "synthetic_issm_changepoint"  # TODO: plots for random changepoint bad groundtruth.
timefeat = TimeFeatType.seasonal_indicator
n_data_per_group = 100

change_point_experiment = True if "changepoint" in experiment_name else False
if experiment_name == "synthetic_issm":
    n_steps_forecast = generate_synthetic_issm.n_steps_forecast
    timesteps = generate_synthetic_issm.T
elif experiment_name == "synthetic_issm_changepoint":
    n_steps_forecast = generate_synthetic_issm_changepoint.n_steps_forecast
    timesteps = generate_synthetic_issm_changepoint.T
elif experiment_name == "synthetic_issm_random_changepoint":
    n_steps_forecast = (
        generate_synthetic_issm_random_changepoint.n_steps_forecast
    )
    timesteps = generate_synthetic_issm_random_changepoint.T
else:
    raise Exception(f"unknown experiment {experiment_name}")

n_timefeat = (
    7
    if timefeat.value == TimeFeatType.seasonal_indicator.value
    else 4
    if timefeat.value == TimeFeatType.timefeat.value
    else 0
    if timefeat.value == TimeFeatType.none.value
    else None
)
assert n_timefeat is not None
n_group = 5

config = SyntheticIssmSGLSConfig(
    dataset_name=experiment_name,
    experiment_name=experiment_name,
    n_epochs=1000,
    lr=1e-2,
    n_epochs_until_validate_loss=1,
    freq="D",
    gpus=tuple(range(0, 4)),  # this and prob
    dtype=torch.float32,
    make_cov_from_cholesky_avg=True,
    dims=TensorDims(
        timesteps=timesteps,
        particle=100,
        batch=n_group * n_data_per_group,
        state=7,
        obs=1,
        switch=10,
        ctrl_switch=n_group + n_timefeat,
        ctrl_obs=n_group + n_timefeat,
        timefeat=n_timefeat,
        staticfeat=n_group,
        cat_embedding=None,
        auxiliary=None,
        ctrl_state=None,
    ),
    n_steps_forecast=n_steps_forecast,
    time_feat=timefeat,
    init_scale_A=None,
    init_scale_B=None,
    init_scale_C=None,
    init_scale_D=1.0,
    # init_scale_R_diag=math.sqrt(10.0),
    init_scale_R_diag=[1e-4, 1e0],
    init_scale_Q_diag=[1e-3, 1e0],
    init_scale_S_diag=[1e-4, 1e0],
    state_prior_scale=0.1,
    state_prior_loc=10.0 / 100.0,
    noise_scale_obs=None,
    #
    n_data_per_group=n_data_per_group,
    #
    switch_prior_model_dims=tuple(),
    input_transform_dims=(64, n_group + n_timefeat,),
    switch_transition_model_dims=tuple(),
    state_to_switch_encoder_dims=tuple(),
    obs_to_switch_encoder_dims=tuple(),
    b_fn_dims=tuple(),
    d_fn_dims=tuple(),  # (64,),
    switch_prior_model_activations=ReLU(),
    switch_transition_model_activations=ReLU(),
    state_to_switch_encoder_activations=ReLU(),
    obs_to_switch_encoder_activations=ReLU(),
    input_transform_activations=(ReLU(), None,),
    b_fn_activations=ReLU(),
    d_fn_activations=ReLU(),
    #
    switch_link_type=SwitchLinkType.shared,
    recurrent_link_type=SwitchLinkType.shared,
    is_recurrent=True,
    n_base_A=10,
    n_base_B=None,
    n_base_C=10,
    n_base_D=None,
    n_base_Q=10,
    n_base_R=10,
    n_base_F=10,
    n_base_S=10,
    requires_grad_R=True,
    requires_grad_Q=True,
)
