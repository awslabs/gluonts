import torch
from pytorch_lightning import LightningModule, Trainer

from inference.smc.resampling import EffectiveSampleSizeResampleCriterion
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
import experiments_new_will_replace.model_component_zoo.input_transforms
import experiments_new_will_replace.model_component_zoo.gls_parameters
from experiments_new_will_replace.model_component_zoo\
    .recurrent_base_parameters import StateToSwitchParamsDefault

from models_new_will_replace.sgls_rbpf import SwitchingGaussianLinearSystemBaseRBSMC
from models_new_will_replace.rsgls_rbpf import RecurrentSwitchingGaussianLinearSystemRBSMC
from models_new_will_replace.asgls_rbpf import AuxiliarySwitchingGaussianLinearSystemRBSMC
from models_new_will_replace.arsgls_rbpf import (
    AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC,
)
from models_new_will_replace.categorical_sgls_rbpf import \
    CategoricalSwitchingGaussianLinearSystemRBSMC
from experiments_new_will_replace.gluonts_univariate_datasets.gts_rbsmc_model \
    import GluontsUnivariateDataModel, GluontsUnivariateDataLoaderWrapper

from models.switching_gaussian_linear_system import (
    SwitchingLinearDynamicalSystem, CategoricalSwitchingLinearDynamicalSystem,
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
# config.obs_to_switch_encoder_dims = config.dims_encoder[0]
# config.obs_to_switch_encoder_activations = config.activations_encoders[0]


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
input_transformer_old = input_transforms.InputTransformEmbeddingAndMLP(
    config=config,
)
input_transformer = experiments_new_will_replace.model_component_zoo\
    .input_transforms.InputTransformEmbeddingAndMLP(
    config=config,
)


def deep_getattr(cls, *args):
    if len(args) == 0:
        raise ValueError("no arguments provided")
    elif len(args) == 1:
        return getattr(cls, args[0])
    else:
        return deep_getattr(getattr(cls, args[0]), *args[1:])


def deep_setattr(cls, *args, val):
    if len(args) == 0:
        raise ValueError("no arguments provided")
    elif len(args) == 1:
        return setattr(cls, args[0], val)
    else:
        return deep_setattr(getattr(cls, args[0]), *args[1:], val=val)


gls_base_parameters_old = gls_parameters.GLSParametersASGLS(config=config)
gls_base_parameters = experiments_new_will_replace.model_component_zoo\
    .gls_parameters.GLSParametersASGLS(config=config)
recurrent_base_parameters = StateToSwitchParamsDefault(config=config)

for name in dict(gls_base_parameters_old.named_parameters()).keys():
    val = deep_getattr(gls_base_parameters_old, *name.split("."))
    deep_setattr(gls_base_parameters, *name.split("."), "data", val=val.data)

switch_transition_model_recurrent = switch_transitions\
    .SwitchTransitionModelGaussianRecurrentBaseMat(config=config)
from copy import deepcopy
config_nonrec = deepcopy(config)
config_nonrec.is_recurrent = False
config_nonrec.obs_to_switch_encoder_dims = (64,)
config_nonrec.obs_to_switch_encoder_activations = (torch.nn.ReLU(),)

switch_transition_model_old = switch_transitions.SwitchTransitionModelGaussian(
    config=config_nonrec,
)
switch_transition_model_cat_old = switch_transitions.SwitchTransitionModelCategorical(config=config_nonrec)
from experiments_new_will_replace.model_component_zoo.switch_transitions import SwitchTransitionModelGaussian, SwitchTransitionModelCategorical
switch_transition_model = SwitchTransitionModelGaussian(config=config)
switch_transition_model_cat = SwitchTransitionModelCategorical(config=config)

from experiments_new_will_replace.model_component_zoo.switch_transitions import SwitchTransitionModelGaussianDirac
switch_transition_model_dirac = SwitchTransitionModelGaussianDirac(config)
#
# switch_transition_model_dirac.conditional_dist.stem = switch_transition_model.conditional_dist.stem
# switch_transition_model_dirac.conditional_dist.dist_params['loc'] = switch_transition_model.conditional_dist.dist_params['loc']
# switch_transition_model_recurrent

state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
switch_prior_model_cat = switch_priors.SwitchPriorModelCategorical(config=config)
measurment_model = decoders.AuxiliaryToObsDecoderMlpGaussian(config=config)
encoder = encoders.ObsToAuxiliaryLadderEncoderMlpGaussian(config=config)
# obs_encoder = encoders.ObsToSwitchEncoderGaussianMLP(config=config)
obs_encoder = lambda x: encoder(x)[1]  # hack
from box import  Box
obs_encoder_auxiliary_old = lambda x: Box(auxiliary=encoder(x)[0], switch=encoder(x)[1])

obs_encoder_cat = encoders.ObsToSwitchEncoderCategoricalMLP(config=config_nonrec)


device = "cuda"
dtype = torch.float32
# ***** SGLS *****
sgls = SwitchingGaussianLinearSystemBaseRBSMC(
    n_state=dims.state,
    n_target=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_target=dims.ctrl_target,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    encoder=obs_encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
    resampling_criterion_fn=EffectiveSampleSizeResampleCriterion(0.5),
).to(dtype)

csgls = CategoricalSwitchingGaussianLinearSystemRBSMC(
    n_state=dims.state,
    n_target=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_target=dims.ctrl_target,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    obs_encoder=obs_encoder_cat,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model_cat,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model_cat,
    resampling_criterion_fn=EffectiveSampleSizeResampleCriterion(0.5),
    temperature=torch.Tensor([1.0]),
).to(dtype)

sgls_old = SwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters_old,
    obs_to_switch_encoder=obs_encoder,
    state_to_switch_encoder=None,
    input_transformer=input_transformer_old,
    switch_transition_model=switch_transition_model_old,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
    min_ess_ratio=0.0,
).to(device).to(dtype)

csgls_old = CategoricalSwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters_old,
    obs_to_switch_encoder=obs_encoder_cat,
    state_to_switch_encoder=None,
    input_transformer=input_transformer_old,
    switch_transition_model=switch_transition_model_cat_old,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model_cat,
    min_ess_ratio=0.0,
    temperature=torch.Tensor([1.0]),
).to(device).to(dtype)
# ***** RSGLS *****
rsgls = RecurrentSwitchingGaussianLinearSystemRBSMC(
    n_state=dims.state,
    n_target=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_target=dims.ctrl_target,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    recurrent_base_parameters=recurrent_base_parameters,
    obs_encoder=obs_encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model_dirac,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to(dtype)
rsgls_old = RecurrentSwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters_old,
    obs_to_switch_encoder=obs_encoder,
    state_to_switch_encoder=None,
    input_transformer=input_transformer_old,
    switch_transition_model=switch_transition_model_recurrent,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to(device).to(dtype)

# ***** ASGLS *****
asgls = AuxiliarySwitchingGaussianLinearSystemRBSMC(
    n_state=dims.state,
    n_target=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_target=dims.ctrl_target,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    measurement_model=measurment_model,
    encoder=encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to(dtype)
asgls_old = AuxiliarySwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters_old,
    measurement_model=measurment_model,
    obs_encoder=obs_encoder_auxiliary_old,
    input_transformer=input_transformer_old,
    switch_transition_model=switch_transition_model,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to(device).to(dtype)

# ***** ARSGLS *****
arsgls = AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC(
    n_state=dims.state,
    n_target=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_ctrl_target=dims.ctrl_target,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters,
    recurrent_base_parameters=recurrent_base_parameters,
    measurement_model=measurment_model,
    obs_encoder=encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model_dirac,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to(dtype)
arsgls_old = RecurrentAuxiliarySwitchingLinearDynamicalSystem(
    n_state=dims.state,
    n_obs=dims.target,
    n_ctrl_state=dims.ctrl_state,
    n_particle=dims.particle,
    n_switch=dims.switch,
    gls_base_parameters=gls_base_parameters_old,
    measurement_model=measurment_model,
    obs_encoder=obs_encoder_auxiliary_old,
    input_transformer=input_transformer_old,
    switch_transition_model=switch_transition_model_recurrent,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to(device).to(dtype)

cardinalities = get_cardinalities(
    dataset=dataset, add_trend=config.add_trend
)

batch = next(iter(GluontsUnivariateDataLoaderWrapper(val_loader)))
batch = {k: v.to(device) for k, v in batch.items()}
batch_test = next(iter(GluontsUnivariateDataLoaderWrapper(test_full_loader)))
batch_test = {k: v.to(device) for k, v in batch_test.items()}

batch_old = transform_gluonts_to_pytorch(
    batch=next(iter(val_loader)),
    bias_y=config.normalisation_params[0],
    factor_y=config.normalisation_params[1],
    device=device,
    dtype=dtype,
    time_features=config.time_feat,
    **cardinalities,
)


models = {}
# for ssm in [sgls, rsgls, asgls, arsgls]:
for name, ssm in {
    "sgls": sgls,
    "rsgls": rsgls,
    "asgls": asgls,
    "arsgls": arsgls,
    "csgls": csgls,
}.items():
    models[name] = GluontsUnivariateDataModel(
        ssm=ssm,
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
    ).to(device).to(dtype=dtype)

    # trainer = Trainer(gpus=[0], max_epochs=1)
    # trainer.fit(model)

models_old = {
    "sgls": sgls_old.to(device),
    "rsgls": rsgls_old.to(device),
    "asgls": asgls_old.to(device),
    "arsgls": arsgls_old.to(device),
    "csgls": csgls_old.to(device),
}

for name in ["sgls", "rsgls", "asgls", "arsgls"]:
    models_old[name].input_transformer.embedding.weight.data \
        = models[name].ctrl_transformer.embedding.weight.data
    models_old[name].input_transformer.mlp.linear_0.weight.data \
        = models[name].ctrl_transformer.mlp.linear_0.weight.data
    models_old[name].input_transformer.mlp.linear_0.bias.data \
        = models[name].ctrl_transformer.mlp.linear_0.bias.data

    models[name].ssm.resampling_criterion_fn = \
        EffectiveSampleSizeResampleCriterion(1.0)
    models_old[name].resampling_criterion_fn = \
        EffectiveSampleSizeResampleCriterion(1.0)

import numpy as np
TB = np.prod(batch_old['y'].shape[:2])

for name in ["csgls", "sgls", "rsgls", "asgls", "arsgls"]:
    print(f"next is: {name}")
    # print(models_old[name].loss_forward(**batch_old).sum() / TB)
    print(models[name].loss(**batch).sum())

for name in ["csgls", "sgls", "rsgls", "asgls", "arsgls"]:
    print(f"next is: {name}")
    _ = models[name].ssm.sample_generative(
        n_steps_forecast=len(batch_test["future_time_feat"]),
        n_batch=batch_test["future_time_feat"].shape[1],
        n_particle=config.dims.particle,
        future_controls=models[name].ctrl_transformer(
            time_feat=batch_test["future_time_feat"],
            feat_static_cat=batch_test["feat_static_cat"],
        ),
        deterministic=False,
    )

print("foo")
trainer = Trainer(gpus=[0])
trainer.fit(models['sgls'])