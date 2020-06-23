from dataclasses import dataclass
from torch.nn.modules.activation import LeakyReLU
from torch import nn
from experiments.base_config import BaseConfig, SwitchLinkType, TimeFeatType
from utils.utils import TensorDims
from data.gluonts_nips_datasets.gluonts_nips_datasets import get_cardinalities, \
    get_dataset
from experiments.model_component_zoo import gls_parameters, state_priors, \
    encoders, decoders, input_transforms
from models.kalman_variational_autoencoder import KalmanVariationalAutoEncoder


@dataclass()
class KVAEGtsExpConfig(BaseConfig):
    prediction_length_rolling: int
    prediction_length_full: int
    # switch_prior_model_dims: tuple
    # switch_prior_model_activations: (nn.Module, tuple)
    input_transform_dims: tuple
    input_transform_activations: (nn.Module, tuple)
    # switch_transition_model_dims: tuple
    # switch_transition_model_activations: (nn.Module, tuple)
    # state_to_switch_encoder_dims: tuple
    # state_to_switch_encoder_activations: (nn.Module, tuple)
    # obs_to_switch_encoder_dims: tuple
    # obs_to_switch_encoder_activations: (nn.Module, tuple)
    # b_fn_dims: tuple
    # b_fn_activations: (nn.Module, tuple)
    # d_fn_dims: tuple
    # d_fn_activations: (nn.Module, tuple)
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
    # is_recurrent: bool
    # n_epochs_no_resampling: int
    # obs_to_switch_encoder: bool
    # state_to_switch_encoder: bool

    n_hidden_rnn: int
    rao_blackwellized: bool
    reconstruction_weight: float
    dims_encoder: tuple
    dims_decoder: tuple
    activations_encoder: (tuple, nn.Module)
    activations_decoder: (tuple, nn.Module)
    grad_clip_norm: float
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
    "electricity_nips": 4 * 168,
    "traffic_nips": 4 * 168,
    "solar_nips": 4 * 168,
    "wiki2000_nips": 4 * 31,
}


def get_n_feat_and_freq(dataset_name, timefeat):
    dataset = get_dataset(
        dataset_name)  # bad to load this here, but must extract metadata.
    freq = dataset.metadata.freq
    cardinalities = get_cardinalities(dataset=dataset,
                                      add_trend=add_trend_map[dataset_name])
    n_staticfeat = sum(cardinalities["cardinalities_feat_static_cat"])
    if timefeat.value == TimeFeatType.seasonal_indicator.value:
        n_timefeat = sum(cardinalities["cardinalities_season_indicators"])
    elif timefeat.value == TimeFeatType.timefeat.value:
        n_timefeat = len(cardinalities["cardinalities_season_indicators"]) + 3
    elif timefeat.value == TimeFeatType.both.value:
        n_timefeat = sum(cardinalities["cardinalities_season_indicators"]) \
                     + len(cardinalities["cardinalities_season_indicators"]) + 3
    elif timefeat.value == TimeFeatType.timefeat.none:
        n_timefeat = 0
    else:
        raise Exception("unexpected")
    n_latent = sum(cardinalities["cardinalities_season_indicators"]) \
               + (2 if add_trend_map[dataset_name] else 1)

    prediction_length_rolling = dataset.metadata.prediction_length
    if dataset.metadata.freq == "H":
        prediction_length_full = 7 * prediction_length_rolling
    elif dataset.metadata.freq in ["B", "D"]:
        prediction_length_full = 5 * prediction_length_rolling
    else:
        raise Exception("unexpected freq")

    return n_timefeat, n_staticfeat, n_latent, freq, cardinalities, prediction_length_rolling, prediction_length_full


def make_default_config(dataset_name):
    timefeat = TimeFeatType.seasonal_indicator
    (n_timefeat, n_staticfeat, n_latent, freq, cardinalities,
     prediction_length_rolling, prediction_length_full) = get_n_feat_and_freq(
        dataset_name=dataset_name, timefeat=timefeat)
    assert len(cardinalities["cardinalities_feat_static_cat"]) == 1
    n_static_embedding = min(50, (
            cardinalities["cardinalities_feat_static_cat"][0] + 1) // 2)
    n_ctrl = n_timefeat + n_static_embedding

    dims = TensorDims(
        timesteps=past_lengths[dataset_name],
        particle=1,
        batch=50,
        state=16,  # n_latent,
        obs=1,
        switch=None,
        ctrl_state=n_ctrl,
        ctrl_switch=n_ctrl,
        ctrl_obs=n_ctrl,
        timefeat=n_timefeat,
        staticfeat=n_staticfeat,
        cat_embedding=n_static_embedding,
        auxiliary=10,
    )

    config = KVAEGtsExpConfig(
        experiment_name="kvae",
        dataset_name=dataset_name,
        #
        n_epochs=50,
        # n_epochs_no_resampling=5,
        n_epochs_until_validate_loss=1,
        lr=1e-2 if dataset_name in ["solar_nips"] else 5e-3,
        grad_clip_norm=5.0,
        weight_decay=1e-5,
        num_samples_eval=100,
        # Note: These batch sizes barely fit on the GPU. for Multivariate must be reduced.
        batch_size_val=10 if dataset_name in ["exchange_rate_nips",
                                              "wiki2000_nips"] else 2,
        # architecture, prior, etc.
        state_prior_scale=1.0,
        state_prior_loc=0.0,
        make_cov_from_cholesky_avg=True,
        extract_tail_chunks_for_train=False,
        switch_link_type=SwitchLinkType.shared,
        # they have 1 Dense layer after LSTM.
        recurrent_link_type=SwitchLinkType.shared,
        # is_recurrent=False,
        n_hidden_rnn=50,
        rao_blackwellized=False,
        reconstruction_weight=1.0,  # They use 0.3 w/o rao-BW.
        dims_encoder=(64, 64),
        dims_decoder=(64, 64),
        activations_encoder=LeakyReLU(0.1, inplace=True),
        activations_decoder=LeakyReLU(0.1, inplace=True),
        n_base_A=20,
        n_base_B=20,
        n_base_C=20,
        n_base_D=None,  # they dont have D
        n_base_Q=20,
        n_base_R=20,
        n_base_F=None,
        n_base_S=None,
        requires_grad_R=True,
        requires_grad_Q=True,
        # obs_to_switch_encoder=True,
        # state_to_switch_encoder=False,
        # switch_prior_model_dims=tuple(),
        input_transform_dims=tuple() + (n_ctrl,),
        # switch_transition_model_dims=(64,),
        # state_to_switch_encoder_dims=(64,),
        # obs_to_switch_encoder_dims=(64,),
        # b_fn_dims=tuple(),
        # d_fn_dims=tuple(),  #(64,),
        # switch_prior_model_activations=LeakyReLU(0.1, inplace=True),
        input_transform_activations=LeakyReLU(0.1, inplace=True),
        # switch_transition_model_activations=LeakyReLU(0.1, inplace=True),
        # state_to_switch_encoder_activations=LeakyReLU(0.1, inplace=True),
        # obs_to_switch_encoder_activations=LeakyReLU(0.1, inplace=True),
        # b_fn_activations=LeakyReLU(0.1, inplace=True),
        # d_fn_activations=LeakyReLU(0.1, inplace=True),
        # initialisation
        init_scale_A=0.95,
        init_scale_B=1e-4,
        init_scale_C=None,
        init_scale_D=None,
        init_scale_R_diag=[1e-4, 1e-1],
        init_scale_Q_diag=[1e-4, 1e-1],
        init_scale_S_diag=None,
        # init_scale_S_diag=[1e-5, 1e0],
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
    # dims = config.dims
    # gls_base_parameters = gls_parameters.GlsParametersISSM(config=config)
    # obs_to_switch_encoder = encoders.ObsToSwitchEncoderGaussianMLP(config=config) \
    #     if config.obs_to_switch_encoder else None
    # state_to_switch_encoder = encoders.StateToSwitchEncoderGaussianMLP(config=config) \
    #     if config.state_to_switch_encoder else None
    # switch_transition_model = switch_transitions.SwitchTransitionModelGaussian(config=config)
    # state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
    # switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)

    input_transformer = input_transforms.InputTransformEmbeddingAndMLPKVAE(
        config=config)
    state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
    gls_base_params = gls_parameters.GLSParametersKVAE(config=config)
    decoder = decoders.AuxiliaryToObsDecoderMlpGaussian(config=config)
    encoder = encoders.ObsToAuxiliaryEncoderMlpGaussian(config=config)
    rnn = nn.LSTM(input_size=config.dims.auxiliary,
                  hidden_size=config.n_hidden_rnn)

    model = KalmanVariationalAutoEncoder(
        n_state=config.dims.state,
        n_obs=config.dims.obs,
        n_auxiliary=config.dims.auxiliary,
        n_ctrl_state=config.dims.ctrl_state,
        n_particle=config.dims.particle,
        gls_base_parameters=gls_base_params,
        input_transformer=input_transformer,
        measurement_model=decoder,
        obs_to_auxiliary_encoder=encoder,
        rnn_switch_model=rnn,
        state_prior_model=state_prior_model,
        reconstruction_weight=config.reconstruction_weight,
    )
    return model


# This fn must come after the *_mod functions as it uses locals
def make_experiment_config(dataset_name, experiment_name):
    config = make_default_config(dataset_name=dataset_name)
    if experiment_name is not None and experiment_name != "default":
        if experiment_name in ["kvae", "kvae_mc", "kvae_rb"]:
            return config
        else:
            raise NotImplementedError("")
            if not f"{experiment_name}" in locals():
                raise Exception(
                    f"config file must have function {experiment_name}_mod")
            mod_fn = locals()[f"{experiment_name}_mod"]
            print(f"modifying config for experiment {experiment_name}")
            config = mod_fn(config)
    return config
