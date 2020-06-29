import torch
from pytorch_lightning import LightningModule, Trainer

from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    create_loaders,
    get_dataset,
    transform_gluonts_to_pytorch,
    get_cardinalities,
)
from experiments.gluonts_nips_experiments.auxiliary_config import (
    make_default_config,
)
from experiments.model_component_zoo import (
    gls_parameters,
    switch_transitions,
    switch_priors,
    state_priors,
    encoders,
    input_transforms,
    decoders,
)
from models_new_will_replace.sgls_rbpf import SwitchingGaussianLinearSystemRBSMC
from models_new_will_replace.rsgls_rbpf import RecurrentSwitchingGaussianLinearSystemRBSMC
from models_new_will_replace.asgls_rbpf import AuxiliarySwitchingGaussianLinearSystemRBSMC
from models_new_will_replace.arsgls_rbpf import (
    AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC,
)
from experiments_new_will_replace.gluonts_univariate_datasets.gts_rbsmc_model \
    import GluontsUnivariateDataModel, GluontsUnivariateDataLoaderWrapper

from models.switching_gaussian_linear_system import (
    SwitchingLinearDynamicalSystem,
)
from models.switching_gaussian_linear_system import (
    RecurrentSwitchingLinearDynamicalSystem,
)
from models.auxiliary_switching_gaussian_linear_system import (
    AuxiliarySwitchingLinearDynamicalSystem,
)
from models.auxiliary_switching_gaussian_linear_system import (
    RecurrentAuxiliarySwitchingLinearDynamicalSystem,
)


dataset_name = "exchange_rate_nips"

config = make_default_config(dataset_name=dataset_name)
config.obs_to_switch_encoder_dims = config.dims_encoder[0]
config.obs_to_switch_encoder_activations = config.activations_encoders[0]


dataset = get_dataset(dataset_name)

(
    train_loader,
    val_loader,
    test_full_loader,
    test_rolling_loader,
    input_transformations,
) = create_loaders(
    dataset=dataset,
    batch_sizes={
        "train": config.dims.batch,
        "val": config.batch_size_val,
        "test_rolling": config.batch_size_val,
        "test_full": config.batch_size_val,
    },
    past_length=config.dims.timesteps,
    prediction_length_full=config.prediction_length_full,
    prediction_length_rolling=config.prediction_length_rolling,
    num_workers=0,
    extract_tail_chunks_for_train=config.extract_tail_chunks_for_train,
)


dims = config.dims
input_transformer = input_transforms.InputTransformEmbeddingAndMLP(
    config=config,
)
gls_base_parameters = gls_parameters.GLSParametersASGLS(config=config)
switch_transition_model = switch_transitions.SwitchTransitionModelGaussianRecurrentBaseMat(
    config=config
)
state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
measurment_model = decoders.AuxiliaryToObsDecoderMlpGaussian(config=config)
encoder = encoders.ObsToAuxiliaryLadderEncoderMlpGaussian(config=config)
obs_encoder = encoders.ObsToSwitchEncoderGaussianMLP(config=config)

# ***** SGLS *****
sgls = SwitchingGaussianLinearSystemRBSMC(
    n_state=dims.state,
    n_obs=dims.obs,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_obs=dims.ctrl_obs,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    obs_encoder=encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to("cuda").to(torch.float64)

sgls_old = SwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.obs,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    obs_to_switch_encoder=obs_encoder,
    state_to_switch_encoder=None,
    input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to("cuda").to(torch.float64)

# ***** RSGLS *****
rsgls = RecurrentSwitchingGaussianLinearSystemRBSMC(
    n_state=dims.state,
    n_obs=dims.obs,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_obs=dims.ctrl_obs,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    obs_encoder=encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to("cuda").to(torch.float64)
rsgls_old = RecurrentSwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.obs,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    obs_to_switch_encoder=obs_encoder,
    state_to_switch_encoder=None,
    input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to("cuda").to(torch.float64)

# ***** ASGLS *****
asgls = AuxiliarySwitchingGaussianLinearSystemRBSMC(
    n_state=dims.state,
    n_obs=dims.obs,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_obs=dims.ctrl_obs,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    measurement_model=measurment_model,
    obs_encoder=encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to("cuda").to(torch.float64)
asgls_old = AuxiliarySwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.obs,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    measurement_model=measurment_model,
    obs_encoder=encoder,
    input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to("cuda").to(torch.float64)

# ***** ARSGLS *****
arsgls = AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC(
    n_state=dims.state,
    n_obs=dims.obs,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_obs=dims.ctrl_obs,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    measurement_model=measurment_model,
    obs_encoder=encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to("cuda").to(torch.float64)
arsgls_old = RecurrentAuxiliarySwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.obs,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    measurement_model=measurment_model,
    obs_encoder=encoder,
    input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to("cuda").to(torch.float64)

cardinalities = get_cardinalities(
    dataset=dataset, add_trend=config.add_trend
)

b = next(iter(train_loader))
batch = transform_gluonts_to_pytorch(
    batch=b,
    bias_y=config.normalisation_params[0],
    factor_y=config.normalisation_params[1],
    device="cuda",
    dtype=getattr(torch, "float64"),
    time_features=config.time_feat,
    **cardinalities,
    normalize_targets=False
)
batch_old = transform_gluonts_to_pytorch(
    batch=b,
    bias_y=config.normalisation_params[0],
    factor_y=config.normalisation_params[1],
    device="cuda",
    dtype=getattr(torch, "float64"),
    time_features=config.time_feat,
    **cardinalities,
    normalize_targets=True,
)


model = GluontsUnivariateDataModel(
    ssm=sgls,
    ctrl_transformer=input_transformer,
    tar_transformer=torch.distributions.AffineTransform(
        loc=config.normalisation_params[0],
        scale=config.normalisation_params[1],
    ),
    dataset_name=dataset_name,
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
    prediction_length_full=config.prediction_length_full,
    prediction_length_rolling=config.prediction_length_rolling,
    num_batches_per_epoch=50,
    extract_tail_chunks_for_train=config.extract_tail_chunks_for_train,
)


trainer = Trainer(gpus=[0])
trainer.fit(model)