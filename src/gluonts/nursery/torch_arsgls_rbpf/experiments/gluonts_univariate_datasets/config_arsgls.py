from typing import Tuple
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.modules.activation import LeakyReLU
from experiments.base_config import BaseConfig, SwitchLinkType, TimeFeatType
from utils.utils import TensorDims
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    get_cardinalities,
    get_dataset,
)
from experiments.model_component_zoo import (
    switch_priors,
    state_priors,
    encoders,
    decoders,
    gls_parameters,
    input_transforms,
    switch_transitions,
)

from models.arsgls_rbpf import (
    AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC,
)
from experiments.gluonts_univariate_datasets.gts_rbsmc_model import (
    GluontsUnivariateDataModel,
)
from experiments.model_component_zoo.recurrent_base_parameters import (
    StateToSwitchParamsDefault,
)


@dataclass()
class ArsglsGtsExpConfigGtsExpConfig(BaseConfig):
    prediction_length_rolling: int
    prediction_length_full: int
    switch_prior_model_dims: tuple
    switch_prior_model_activations: (nn.Module, tuple)
    input_transform_dims: tuple
    input_transform_activations: (nn.Module, tuple)
    switch_transition_model_dims: tuple
    switch_transition_model_activations: (nn.Module, tuple)
    dims_encoder: Tuple[Tuple[int]]
    dims_decoder: tuple
    activations_encoders: Tuple[(tuple, nn.Module)]
    activations_decoder: (tuple, nn.Module)
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
    # use_encoder: bool
    grad_clip_norm: float
    n_epochs_freeze_gls_params: int
    weight_decay: float


# TODO: maybe add trend.
add_trend_map = {
    "exchange_rate_nips": False,
    "electricity_nips": False,
    "traffic_nips": False,
    "solar_nips": False,
    "wiki-rolling_nips": False,  # not used in paper
    "wiki2000_nips": False,
}
normalisation_params = {
    "exchange_rate_nips": [0.68114537, 0.4800387],
    "electricity_nips": [611.15765, 3574.196],
    "traffic_nips": [0.05299745, 0.045804948],
    "solar_nips": [40.349354, 61.589676],
    "wiki-rolling_nips": [3720.5366, 10840.078],  # not used in paper
    "wiki2000_nips": [3720.5366, 10840.078],
}

# because of GPU memory issues and performance, our models use only 2 weeks.
past_lengths = {
    "exchange_rate_nips": 4 * 31,
    "electricity_nips": 2 * 168,
    "traffic_nips": 2 * 168,
    "solar_nips": 2 * 168,
    "wiki-rolling_nips": 4 * 31,  # not used in paper
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
        particle=10,
        batch=64,
        state=16,
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
        auxiliary=10,
    )

    config = ArsglsGtsExpConfigGtsExpConfig(
        experiment_name="auxiliary",
        dataset_name=dataset_name,
        #
        n_epochs=50,
        n_epochs_no_resampling=5,
        n_epochs_freeze_gls_params=0,
        n_epochs_until_validate_loss=1,
        lr=1e-2 if dataset_name in ["solar_nips"] else 5e-3,
        weight_decay=1e-5,
        grad_clip_norm=10.0,
        num_samples_eval=100,
        batch_size_val=50,  # 10
        # gpus=tuple(range(3, 4)),
        # dtype=torch.float64,
        # architecture, prior, etc.
        state_prior_scale=1.0,
        state_prior_loc=0.0,
        make_cov_from_cholesky_avg=True,
        extract_tail_chunks_for_train=False,
        switch_link_type=SwitchLinkType.individual,
        switch_link_dims_hidden=(64,),
        switch_link_activations=nn.LeakyReLU(0.1, inplace=True),
        recurrent_link_type=SwitchLinkType.shared,
        is_recurrent=True,
        n_base_A=10,
        n_base_B=10,
        n_base_C=10,
        n_base_D=10,
        n_base_Q=10,
        n_base_R=10,
        n_base_F=10,
        n_base_S=10,
        requires_grad_R=True,
        requires_grad_Q=True,
        # use_encoder: True,
        switch_prior_model_dims=tuple(),
        input_transform_dims=tuple() + (n_ctrl,),
        switch_transition_model_dims=(64,),
        dims_encoder=((64, 64), (64,)),
        dims_decoder=(64, 64),
        b_fn_dims=tuple(),
        d_fn_dims=tuple(),  # (64,),
        switch_prior_model_activations=LeakyReLU(0.1, inplace=True),
        input_transform_activations=LeakyReLU(0.1, inplace=True),
        switch_transition_model_activations=LeakyReLU(0.1, inplace=True),
        activations_encoders=(
            LeakyReLU(0.1, inplace=True),
            LeakyReLU(0.1, inplace=True),
        ),
        activations_decoder=LeakyReLU(0.1, inplace=True),
        b_fn_activations=LeakyReLU(0.1, inplace=True),
        d_fn_activations=LeakyReLU(0.1, inplace=True),
        # initialisation
        init_scale_A=None,  # 0.95,
        init_scale_B=1e-4,
        init_scale_C=None,
        init_scale_D=None,
        init_scale_R_diag=[1e-4, 1e-1],
        init_scale_Q_diag=[1e-4, 1e-1],
        init_scale_S_diag=[1e-4, 1e-1],
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
    gls_base_parameters = gls_parameters.GLSParametersASGLS(config=config)
    switch_transition_model = switch_transitions.SwitchTransitionModelGaussianDirac(
        config=config
    )
    state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
    switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
    measurment_model = decoders.AuxiliaryToObsDecoderMlpGaussian(config=config)
    encoder = encoders.ObsToAuxiliaryLadderEncoderMlpGaussian(config=config)
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
    model = GluontsUnivariateDataModel(
        config=config,
        ssm=ssm,
        ctrl_transformer=input_transformer,
        tar_transformer=torch.distributions.AffineTransform(
            loc=config.normalisation_params[0],
            scale=config.normalisation_params[1],
        ),
        dataset_name=config.dataset_name,
        lr=config.lr,
        weight_decay=config.weight_decay,
        n_epochs=config.n_epochs,
        batch_sizes={
            "train": config.dims.batch,
            "val": config.batch_size_val,
            "test_rolling": config.batch_size_val,
            "test_full": config.batch_size_val,
        },
        past_length=config.dims.timesteps,
        n_particle_train=config.dims.particle,
        n_particle_eval=config.num_samples_eval,
        prediction_length_full=config.prediction_length_full,
        prediction_length_rolling=config.prediction_length_rolling,
        n_epochs_no_resampling=config.n_epochs_no_resampling,
        n_epochs_freeze_gls_params=config.n_epochs_freeze_gls_params,
        num_batches_per_epoch=50,
        extract_tail_chunks_for_train=config.extract_tail_chunks_for_train,
    )
    return model


# This fn must come after the *_mod functions as it uses locals
# TODO: this is obsolete
def make_experiment_config(dataset_name, experiment_name):
    config = make_default_config(dataset_name=dataset_name)
    if experiment_name is not None and experiment_name != "default":
        if experiment_name == "auxiliary":
            return config
        else:
            raise NotImplementedError("")
            if not f"{experiment_name}" in locals():
                raise Exception(
                    f"config file must have function {experiment_name}_mod"
                )
            mod_fn = locals()[f"{experiment_name}_mod"]
            print(f"modifying config for experiment {experiment_name}")
            config = mod_fn(config)
    return config
