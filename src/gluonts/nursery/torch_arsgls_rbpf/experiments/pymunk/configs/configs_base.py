from typing import Tuple
from dataclasses import dataclass

from torch import nn
from experiments.base_config import BaseConfig
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

from models.kvae import KalmanVariationalAutoEncoder
from models.arsgls_rbpf import (
    AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC,
)

from experiments.base_config import SwitchLinkType
from torch_extensions.layers_with_init import LSTMCell
from experiments.pymunk.pymunk_model import PymunkModel


@dataclass()
class PymunkConfig(BaseConfig):
    dims_img: Tuple[int, int, int]
    dims_filter: Tuple[int]
    kernel_sizes: Tuple[int]
    strides: Tuple[int]
    paddings: Tuple[int]
    upscale_factor: int
    prediction_length: int
    lr_decay_steps: int
    lr_decay_rate: float
    grad_clip_norm: float
    batch_size_eval: int
    weight_decay: float
    num_samples_eval: int
    n_epochs_no_resampling: int
    n_epochs_freeze_gls_params: int


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
    requires_grad_switch_prior: bool


def make_experiment_config(experiment_name, dataset_name):
    from experiments.pymunk.configs import (
        config_arsgls,
        config_kvae_rb_learn,
        config_kvae_rb_fix,
        config_kvae_mc_learn,
        config_kvae_mc_fix,
    )
    configs = {
        "arsgls": config_arsgls,
        "kvae_rb_learn": config_kvae_rb_learn,
        "kvae_rb_fix": config_kvae_rb_fix,
        "kvae_mc_learn": config_kvae_mc_learn,
        "kvae_mc_fix": config_kvae_mc_fix,
    }

    if experiment_name in configs:
        config = configs[experiment_name].config
        assert config.experiment_name == experiment_name
        config.dataset_name = dataset_name
        return config
    else:
        raise Exception(f"unknown experiment/model: {experiment_name}")


def _make_kvae(config):
    gls_base_params = gls_parameters.GLSParametersKVAE(config=config)
    decoder = decoders.AuxiliaryToObsDecoderConvBernoulli(config=config)
    encoder = encoders.ObsToAuxiliaryEncoderConvGaussian(config=config)
    state_prior_model = state_priors.StatePriorModeFixedNoInputs(
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
        rao_blackwellized=config.rao_blackwellized,
    )
    return ssm


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


def make_model(config):
    if config.experiment_name.startswith("kvae"):
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
        lr_decay_rate=config.lr_decay_rate,
        lr_decay_steps=config.lr_decay_steps,
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
        num_batches_per_epoch=None,
        log_param_norms=False,
        n_epochs_freeze_gls_params=config.n_epochs_freeze_gls_params,
    )
    return model
