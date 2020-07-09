from copy import deepcopy
from dataclasses import dataclass
from torch.nn.modules.activation import LeakyReLU
from torch import nn
from experiments.base_config import BaseConfig, SwitchLinkType, TimeFeatType
from utils.utils import TensorDims
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    get_cardinalities,
    get_dataset,
)
from experiments.model_component_zoo import (
    gls_parameters,
    switch_transitions,
    switch_priors,
    state_priors,
    encoders,
    input_transforms,
)
from models.switching_gaussian_linear_system import (
    RecurrentSwitchingLinearDynamicalSystem,
)


@dataclass()
class GtsExpConfig(BaseConfig):
    prediction_length_rolling: int
    prediction_length_full: int
    switch_prior_model_dims: tuple
    switch_prior_model_activations: (nn.Module, tuple)
    input_transform_dims: tuple
    input_transform_activations: (nn.Module, tuple)
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
    add_trend: bool
    normalisation_params: list
    extract_tail_chunks_for_train: bool
    n_epochs_until_validate_loss: int
    # gpus: (list, tuple)
    num_samples_eval: int
    batch_size_val: int
    # dtype: torch.dtype
    make_cov_from_cholesky_avg: bool
    is_recurrent: bool
    n_epochs_no_resampling: int
    obs_to_switch_encoder: bool
    state_to_switch_encoder: bool
    grad_clip_norm: float
    n_epochs_freeze_gls_params: int
    weight_decay: float


# TODO: maybe add trend.
add_trend_map = {
    "exchange_rate_nips": False,
    "electricity_nips": False,
    "traffic_nips": False,
    "solar_nips": False,
    "wiki2000_nips": False,
}
normalisation_params = {
    "exchange_rate_nips": [0.68114537, 0.4800387],
    "electricity_nips": [611.15765, 3574.196],
    "traffic_nips": [0.05299745, 0.045804948],
    "solar_nips": [40.349354, 61.589676],
    "wiki2000_nips": [3720.5366, 10840.078],
}
past_lengths = {
    "exchange_rate_nips": 4 * 31,
    "electricity_nips": 2 * 168,
    "traffic_nips": 2 * 168,
    "solar_nips": 2 * 168,
    "wiki2000_nips": 4 * 31,
}


def get_n_feat_and_freq(dataset_name, timefeat):
    dataset = get_dataset(
        dataset_name
    )  # bad to load this here, but must extract metadata.
    freq = dataset.metadata.freq
    cardinalities = get_cardinalities(
        dataset=dataset, add_trend=add_trend_map[dataset_name]
    )
    n_staticfeat = sum(cardinalities["cardinalities_feat_static_cat"])
    if timefeat.value == TimeFeatType.seasonal_indicator.value:
        n_timefeat = sum(cardinalities["cardinalities_season_indicators"])
    elif timefeat.value == TimeFeatType.timefeat.value:
        n_timefeat = len(cardinalities["cardinalities_season_indicators"]) + 3
    elif timefeat.value == TimeFeatType.both.value:
        n_timefeat = (
            sum(cardinalities["cardinalities_season_indicators"])
            + len(cardinalities["cardinalities_season_indicators"])
            + 3
        )
    elif timefeat.value == TimeFeatType.timefeat.none:
        n_timefeat = 0
    else:
        raise Exception("unexpected")
    n_latent = sum(cardinalities["cardinalities_season_indicators"]) + (
        2 if add_trend_map[dataset_name] else 1
    )

    prediction_length_rolling = dataset.metadata.prediction_length
    if dataset.metadata.freq == "H":
        prediction_length_full = 7 * prediction_length_rolling
    elif dataset.metadata.freq in ["B", "D"]:
        prediction_length_full = 5 * prediction_length_rolling
    else:
        raise Exception("unexpected freq")

    return (
        n_timefeat,
        n_staticfeat,
        n_latent,
        freq,
        cardinalities,
        prediction_length_rolling,
        prediction_length_full,
    )


def make_default_config(dataset_name):
    timefeat = TimeFeatType.timefeat
    (
        n_timefeat,
        n_staticfeat,
        n_latent,
        freq,
        cardinalities,
        prediction_length_rolling,
        prediction_length_full,
    ) = get_n_feat_and_freq(dataset_name=dataset_name, timefeat=timefeat)
    assert len(cardinalities["cardinalities_feat_static_cat"]) == 1
    n_static_embedding = min(
        50, (cardinalities["cardinalities_feat_static_cat"][0] + 1) // 2
    )
    n_ctrl = 64

    dims = TensorDims(
        timesteps=past_lengths[dataset_name],
        particle=20,
        batch=10,
        state=n_latent,
        target=1,
        switch=10,
        # ctrl_state=None,
        # ctrl_switch=n_staticfeat + n_timefeat,
        # ctrl_obs=n_staticfeat + n_timefeat,
        ctrl_state=n_ctrl,
        ctrl_switch=n_ctrl,
        ctrl_target=n_ctrl,
        timefeat=n_timefeat,
        staticfeat=n_staticfeat,
        cat_embedding=n_static_embedding,
        auxiliary=None,
    )

    config = GtsExpConfig(
        experiment_name="default",
        dataset_name=dataset_name,
        #
        n_epochs=40,
        n_epochs_no_resampling=5,
        n_epochs_freeze_gls_params=1,
        n_epochs_until_validate_loss=1,
        lr=1e-3
        if dataset_name in ["electricity_nips"]
        else 1e-2
        if dataset_name in ["solar_nips"]
        else 5e-3,
        weight_decay=1e-5,
        grad_clip_norm=10.0,
        num_samples_eval=100,
        batch_size_val=15,  # 10
        # gpus=tuple(range(3, 4)),
        # dtype=torch.float64,
        # architecture, prior, etc.
        state_prior_scale=1.0,
        state_prior_loc=0.0,
        make_cov_from_cholesky_avg=True,
        extract_tail_chunks_for_train=False,
        switch_link_type=SwitchLinkType.individual,
        recurrent_link_type=SwitchLinkType.shared,
        is_recurrent=True,
        n_base_A=20,
        n_base_B=None,
        n_base_C=20,
        n_base_D=20,
        n_base_Q=20,
        n_base_R=20,
        n_base_F=20,
        n_base_S=20,
        requires_grad_R=True,
        requires_grad_Q=True,
        obs_to_switch_encoder=True,
        state_to_switch_encoder=False,
        switch_prior_model_dims=tuple(),
        input_transform_dims=tuple() + (n_ctrl,),
        switch_transition_model_dims=(64,),
        state_to_switch_encoder_dims=(64,),
        obs_to_switch_encoder_dims=(64,),
        b_fn_dims=tuple(),
        d_fn_dims=tuple(),  # (64,),
        switch_prior_model_activations=LeakyReLU(0.1, inplace=True),
        input_transform_activations=LeakyReLU(0.1, inplace=True),
        switch_transition_model_activations=LeakyReLU(0.1, inplace=True),
        state_to_switch_encoder_activations=LeakyReLU(0.1, inplace=True),
        obs_to_switch_encoder_activations=LeakyReLU(0.1, inplace=True),
        b_fn_activations=LeakyReLU(0.1, inplace=True),
        d_fn_activations=LeakyReLU(0.1, inplace=True),
        # initialisation
        init_scale_A=None,
        init_scale_B=None,
        init_scale_C=None,
        init_scale_D=1e-4,
        init_scale_R_diag=[1e-5, 1e-1],
        init_scale_Q_diag=[1e-4, 1e0],
        init_scale_S_diag=[1e-5, 1e0],
        # set from outside due to dependencies.
        dims=dims,
        freq=freq,
        time_feat=timefeat,
        add_trend=add_trend_map[dataset_name],
        prediction_length_rolling=prediction_length_rolling,
        prediction_length_full=prediction_length_full,
        normalisation_params=normalisation_params[dataset_name],
    )
    return config


def make_model(config):
    dims = config.dims
    input_transformer = input_transforms.InputTransformEmbeddingAndMLP(
        config=config
    )
    gls_base_parameters = gls_parameters.GlsParametersISSM(config=config)
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
    switch_transition_model = switch_transitions.SwitchTransitionModelGaussianRecurrentBaseMat(
        config=config
    )
    state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
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


def switch_mod(config):
    cfgs = {}
    for n_switch in [1, 3, 5, 10]:
        cfg = deepcopy(config)
        dims_dict = cfg.dims._asdict()
        dims_dict.switch = n_switch
        cfg.dims = cfg.dims.__class__(**dims_dict)
        cfgs[str(n_switch)] = cfg
    return cfgs


def batchsize_mod(config):
    cfgs = {}
    for n_batch in [50, 100, 200]:
        cfg = deepcopy(config)
        dims_dict = cfg.dims._asdict()
        dims_dict.batch = n_batch
        cfg.dims = cfg.dims.__class__(**dims_dict)
        cfgs[str(n_batch)] = cfg
    return cfgs


def particle_mod(config):
    cfgs = {}
    for n_particle in [1, 10, 100]:
        cfg = deepcopy(config)
        dims_dict = cfg.dims._asdict()
        dims_dict.particle = n_particle
        cfg.dims = cfg.dims.__class__(**dims_dict)
        cfgs[str(n_particle)] = cfg
    return cfgs


def basemat_mod(config):
    cfgs = {}
    for n_base in [10, 15, 20, 25]:
        cfg = deepcopy(config)
        cfg.n_base_A = (n_base,)
        cfg.n_base_B = (None,)
        cfg.n_base_C = (n_base,)
        cfg.n_base_D = (n_base,)
        cfg.n_base_Q = (n_base,)
        cfg.n_base_R = (n_base,)
        cfg.n_base_F = (n_base,)
        cfg.n_base_S = (n_base,)
        cfgs[str(n_base)] = cfg
    return cfgs


def encoder_mod(config):
    cfgs = {}
    for which in ["none", "obs", "state", "both"]:
        cfg = deepcopy(config)
        if which == "none":
            cfg.obs_to_switch_encoder = False
            cfg.state_to_switch_encoder = False
        elif which == "obs":
            cfg.obs_to_switch_encoder = True
            cfg.state_to_switch_encoder = False
        elif which == "state":
            cfg.obs_to_switch_encoder = False
            cfg.state_to_switch_encoder = True
        elif which == "both":
            cfg.obs_to_switch_encoder = True
            cfg.state_to_switch_encoder = True
        else:
            raise Exception()
        cfgs[which] = cfg
    return cfgs


# This fn must come after the *_mod functions as it uses locals
def make_experiment_config(dataset_name, experiment_name):
    config = make_default_config(dataset_name=dataset_name)
    if experiment_name is not None and experiment_name != "default":
        if not f"{experiment_name}" in locals():
            raise Exception(
                f"config file must have function {experiment_name}_mod"
            )
        mod_fn = locals()[f"{experiment_name}_mod"]
        print(f"modifying config for experiment {experiment_name}")
        config = mod_fn(config)
    return config
