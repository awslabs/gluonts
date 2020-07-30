import torch
from pytorch_lightning import LightningModule, Trainer

from inference.smc.resampling import make_criterion_fn_with_ess_threshold
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    create_loaders,
    get_dataset,
    transform_gluonts_to_pytorch,
    get_cardinalities,
)
from experiments.gluonts_nips_experiments.config import (
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

from models_new_will_replace.sgls_rbpf import SwitchingGaussianLinearSystemBaseRBSMC
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
from copy import deepcopy
import experiments_new_will_replace.model_component_zoo.switch_transitions as st

from experiments_new_will_replace.model_component_zoo\
    .recurrent_base_parameters import StateToSwitchParamsDefault
from box import Box
import torch
import numpy as np
import mxnet as mx

seed = 2
mx.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


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


dataset_name = "wiki-rolling_nips"
device = "cuda"
dtype = torch.float64

config = make_default_config(dataset_name=dataset_name)
dims = config.dims

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



input_transformer_old = input_transforms.InputTransformEmbeddingAndMLP(
    config=config,
)
input_transformer = experiments_new_will_replace.model_component_zoo\
    .input_transforms.InputTransformEmbeddingAndMLP(
    config=config,
)


#  ************** GLS PARAMS *******************
gls_base_parameters_old = gls_parameters.GlsParametersISSM(config=config)
gls_base_parameters = experiments_new_will_replace.model_component_zoo.gls_parameters.GlsParametersISSM(config=config)


#  *************** TRANSITION *******************
config_nonrec = deepcopy(config)
config_nonrec.is_recurrent = False

recurrent_base_parameters = StateToSwitchParamsDefault(config=config)

switch_transition_model_recurrent_old = switch_transitions\
    .SwitchTransitionModelGaussianRecurrentBaseMat(
        config=config,
)
switch_transition_model_recurrent = st.SwitchTransitionModelGaussianDirac(
    config=config,
)
switch_transition_model_old = switch_transitions.SwitchTransitionModelGaussian(
    config=config_nonrec,
)
switch_transition_model = st.SwitchTransitionModelGaussian(
    config=config,
)


state_prior_model = state_priors.StatePriorModelNoInputs(config=config)
switch_prior_model = switch_priors.SwitchPriorModelGaussian(config=config)
encoder = encoders.ObsToSwitchEncoderGaussianMLP(config=config)
obs_encoder = encoder
obs_encoder_auxiliary_old = lambda x: Box(auxiliary=encoder(x)[0], switch=encoder(x)[1])


from experiments.gluonts_nips_experiments.config import (
    make_model,
    make_experiment_config,
)
rsgls_old = make_model(config=config).to(device).to(dtype)
switch_transition_model_recurrent_old = rsgls_old.switch_transition_model
switch_prior_model = rsgls_old.switch_prior_model
state_prior_model = rsgls_old.state_prior_model
input_transformer_old = rsgls_old.input_transformer
gls_base_parameters_old = rsgls_old.gls_base_parameters
# rsgls_old.obs_to_switch_encoder = obs_encoder
obs_encoder = rsgls_old.obs_to_switch_encoder

# ****************** SET OLD -> NEW ********************
for name in dict(gls_base_parameters_old.named_parameters()).keys():
    val = deep_getattr(gls_base_parameters_old, *name.split("."))
    deep_setattr(gls_base_parameters, *name.split("."), "data", val=val.data)


switch_transition_model.conditional_dist.stem.linear_0.weight.data = switch_transition_model_old.conditional_dist.conditional_dist_transform.stem.linear_0.weight.data
switch_transition_model.conditional_dist.stem.linear_0.bias.data = switch_transition_model_old.conditional_dist.conditional_dist_transform.stem.linear_0.bias.data
switch_transition_model.conditional_dist.dist_params.loc[0].weight.data = switch_transition_model_old.conditional_dist.conditional_dist_transform.dist_params.loc[0].weight.data
switch_transition_model.conditional_dist.dist_params.loc[0].bias.data = switch_transition_model_old.conditional_dist.conditional_dist_transform.dist_params.loc[0].bias.data
switch_transition_model.conditional_dist.dist_params.scale_tril[0].weight.data = switch_transition_model_old.conditional_dist.conditional_dist_transform.dist_params.scale_tril[0].weight.data
switch_transition_model.conditional_dist.dist_params.scale_tril[0].bias.data = switch_transition_model_old.conditional_dist.conditional_dist_transform.dist_params.scale_tril[0].bias.data


switch_transition_model_recurrent.conditional_dist.stem.linear_0.weight.data = switch_transition_model_recurrent_old.transform.linear_0.weight.data
switch_transition_model_recurrent.conditional_dist.stem.linear_0.bias.data = switch_transition_model_recurrent_old.transform.linear_0.bias.data
switch_transition_model_recurrent.conditional_dist.dist_params.loc[0].weight.data = switch_transition_model_recurrent_old.transform.linear_1.weight.data
switch_transition_model_recurrent.conditional_dist.dist_params.loc[0].bias.data = switch_transition_model_recurrent_old.transform.linear_1.bias.data

recurrent_base_parameters.F.data = switch_transition_model_recurrent_old.base_parameters.F.data
recurrent_base_parameters.LSinv_logdiag.data = switch_transition_model_recurrent_old.base_parameters.LSinv_logdiag.data
recurrent_base_parameters.link_transformers.link.linear_0.weight.data = switch_transition_model_recurrent_old.base_parameters.link_transformers.link.linear_0.weight.data
recurrent_base_parameters.link_transformers.link.linear_0.bias.data = switch_transition_model_recurrent_old.base_parameters.link_transformers.link.linear_0.bias.data

input_transformer.embedding.weight.data = input_transformer_old.embedding.weight.data
input_transformer.mlp.linear_0.weight.data = input_transformer_old.mlp.linear_0.weight.data
input_transformer.mlp.linear_0.bias.data = input_transformer_old.mlp.linear_0.bias.data


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
    resampling_criterion_fn=make_criterion_fn_with_ess_threshold(0.5),
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
    encoder=obs_encoder,
    # input_transformer=input_transformer,
    switch_transition_model=switch_transition_model_recurrent,
    state_prior_model=state_prior_model,
    switch_prior_model=switch_prior_model,
).to(dtype)
# rsgls_old = RecurrentSwitchingLinearDynamicalSystem(
#     n_state=dims.state,
#     n_obs=dims.target,
#     n_ctrl_state=dims.ctrl_state,
#     n_particle=dims.particle,
#     n_switch=dims.switch,
#     gls_base_parameters=gls_base_parameters_old,
#     obs_to_switch_encoder=obs_encoder,
#     state_to_switch_encoder=None,
#     input_transformer=input_transformer_old,
#     switch_transition_model=switch_transition_model_recurrent_old,
#     state_prior_model=state_prior_model,
#     switch_prior_model=switch_prior_model,
# ).to(device).to(dtype)


cardinalities = get_cardinalities(
    dataset=dataset, add_trend=config.add_trend
)

batch = next(iter(GluontsUnivariateDataLoaderWrapper(val_loader)))
batch = {k: v.to(device).to(dtype if v.dtype == torch.float32 else v.dtype) for k, v in batch.items()}
batch_test = next(iter(GluontsUnivariateDataLoaderWrapper(test_full_loader)))
batch_test = {k: v.to(device).to(dtype if v.dtype == torch.float32 else v.dtype) for k, v in batch_test.items()}

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
        n_particle_train=config.dims.particle,
        n_particle_eval=config.num_samples_eval,  # TODO
        prediction_length_full=config.prediction_length_full,
        prediction_length_rolling=config.prediction_length_rolling,
        num_batches_per_epoch=50,
        extract_tail_chunks_for_train=config.extract_tail_chunks_for_train,
    ).to(device).to(dtype=dtype)

models_old = {
    "sgls": sgls_old.to(dtype).to(device),
    "rsgls": rsgls_old.to(dtype).to(device),
    # "asgls": asgls_old.to(device),
    # "arsgls": arsgls_old.to(device),
}




TB = np.prod(batch_old['y'].shape[:2])
for name in ["sgls", "rsgls"]:
    print(f"next is: {name}")
    losses_old = []
    losses_new = []
    with torch.no_grad():
        for _ in range(3):
            seed = 2
            mx.random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            losses_old.append(models_old[name].loss_forward(**batch_old).sum().detach().cpu().numpy() / TB)
            mx.random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            losses_new.append(models[name].loss(**batch).sum().detach().cpu().numpy())
    print(f"{np.mean(losses_old):.3f} +- {np.std(losses_old):.3f}")
    print(f"{np.mean(losses_new):.3f} +- {np.std(losses_new):.3f}")

for name in ["sgls", "rsgls"]:
    print(f"next is: {name}")
    _ = models[name].ssm.sample_generative(
        n_steps_forecast=len(batch_test["future_time_feat"]),
        n_batch=batch_test["future_time_feat"].shape[1],
        n_particle=config.dims.particle,
        future_controls=models[name].ctrl_transformer(
            time_feat=batch_test["future_time_feat"],
            feat_static_cat=batch_test["feat_static_cat"],
            seasonal_indicators=batch_test['future_seasonal_indicators'],
        ),
        deterministic=False,
    )

print("training")
from utils.utils import prepare_logging
import consts
log_paths = prepare_logging(
    config=config, consts=consts, run_nr=None,
)

trainer = Trainer(
    gpus=[0],
    default_root_dir=log_paths.root,
    gradient_clip_val=config.grad_clip_norm,
    limit_val_batches=1,
    max_epochs=config.n_epochs,
)
trainer.log_paths = log_paths

batch_train = next(iter(train_loader))
np.save("/home/richardk/Desktop/tmp.npy", batch_train, allow_pickle=True)




# trainer.fit(models['rsgls'])
# trainer.test(models['rsgls'])


from experiments.gluonts_nips_experiments.run import train

model, validator = train(
    config=config,
    log_paths=log_paths,
    model=models_old['rsgls'],
    dataset=dataset,
    val_loader=val_loader,
    n_epochs=config.n_epochs,
    lr=config.lr,
    train_loader=train_loader,
    gpus=[0],
    dtype=getattr(torch, "float64"),
    input_transforms=input_transformations,
    cardinalities=cardinalities,
    num_samples_eval=config.num_samples_eval,
)

# from experiments.validator import Validator
#
# validator = Validator(
#     log_paths=log_paths,
#     dataset=dataset.train,
#     forecaster=models['rsgls'].predictors['full'],
#     num_samples=config.num_samples_eval,
#     num_series=min(len(dataset.train), 100),
# )
# agg_metrics = validator(epoch=0, save=False)
#
# print(agg_metrics["mean_wQuantileLoss"])