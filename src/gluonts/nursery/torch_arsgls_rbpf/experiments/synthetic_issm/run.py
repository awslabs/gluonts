import os
import torch
from torch.optim.adam import Adam

from models.switching_gaussian_linear_system import (
    SwitchingLinearDynamicalSystem,
    CategoricalSwitchingLinearDynamicalSystem,
    RecurrentSwitchingLinearDynamicalSystem,
)
import consts
from data.synthetic_issm.synthetic_issm_loader import (
    create_trainset_loader,
    create_inference_loader,
    gluonts_batch_to_train_pytorch,
    gluonts_batch_to_forecast_pytorch,
)
from experiments.synthetic_issm.sgls_components import (
    SyntheticGLSParameters,
    SyntheticStateToSwitchEncoder,
    SyntheticObsToSwitchEncoder,
    SyntheticStatePriorModel,
    SyntheticSwitchTransitionModel,
    SyntheticSwitchPriorModel,
    SyntheticInputTransform,
)
from utils.utils import prepare_logging
from experiments.synthetic_issm.config import config, change_point_experiment
from inference.smc.resampling import make_criterion_fn_with_ess_threshold

if change_point_experiment:
    from experiments.synthetic_issm.plots_changepoint import (
        make_params_over_training_plots,
        make_state_prior_plots,
        make_forecast_plots,
    )
else:
    from experiments.synthetic_issm.plots import (
        make_params_over_training_plots,
        make_state_prior_plots,
        make_forecast_plots,
    )


def train(
    log_paths,
    model,
    train_loader,
    val_loader,
    device="cuda",
    dtype=torch.float32,
    seed=42,
    n_epochs_until_validate_loss=10,
    n_epochs_until_validate_plots=10000,
    show_plots=True,
):
    print(f"Training on device: {device}; dtype: {dtype}; seed: {seed}")
    torch.manual_seed(seed)
    model = model.to(dtype).to(device)

    # ***** Optimizer *****
    all_params = list(model.parameters())
    warmup_params = [model.state_prior_model.m]
    large_params = [
        model.state_prior_model.m,
        model.state_prior_model.LVinv_tril,
        model.state_prior_model.LVinv_logdiag,
    ] + list(model.gls_base_parameters.parameters())
    other_params = [
        param
        for param in all_params
        if not any([param is lparam for lparam in large_params])
    ]
    assert len(all_params) == len(large_params) + len(other_params)

    # # ********** some configs that we want to change quickly # **********
    # warmup and 2x learning rate params are too big messy for a config file...
    n_epochs = 500
    n_epochs_no_resampling = 50
    n_epochs_warmup = 0
    n_iter_decay_one_order_of_magnitude = 500
    optimizer_warmup = Adam(
        warmup_params, lr=1e-2, betas=(0.95, 0.99), amsgrad=False
    )
    optimizer = Adam(
        [{"params": large_params, "lr": 1e-2}, {"params": other_params}],
        lr=1e-2,
        betas=(0.9, 0.95),
        amsgrad=False,
    )
    # **********
    decay_rate = (1 / 10) ** (1 / n_iter_decay_one_order_of_magnitude)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=decay_rate
    )

    val_data = gluonts_batch_to_forecast_pytorch(
        batch=next(iter(val_loader)),
        device=model.state_prior_model.m.device,
        dtype=model.state_prior_model.m.dtype,
        dims=dims,
        time_features=config.time_feat,
        n_steps_forecast=config.n_steps_forecast,
    )
    for idx_epoch in range(n_epochs):
        # save model
        torch.save(
            model.state_dict(),
            os.path.join(log_paths.model, f"{idx_epoch}.pt"),
        )

        # choose optimizer & apply lr decay
        optimizer = (
            optimizer_warmup if idx_epoch < n_epochs_warmup else optimizer
        )

        # validation: loss
        if idx_epoch % n_epochs_until_validate_loss == 0:
            loss_val = (
                model(**val_data).mean(dim=(0, 1)).detach().cpu().numpy()
            )
            print(f"epoch: {idx_epoch}; loss {loss_val:.3f}")
        # validation: plots
        if (idx_epoch + 1) % n_epochs_until_validate_plots == 0:
            make_params_over_training_plots(
                epoch=idx_epoch,
                model=model,
                log_paths=log_paths,
                show=show_plots,
            )
            make_forecast_plots(
                epoch=idx_epoch,
                model=model,
                log_paths=log_paths,
                dims=dims,
                n_steps_forecast=config.n_steps_forecast,
                data=val_data,
                show=show_plots,
            )

        # annealing
        # set re-sampling criterion to ESS = 0.5 after given number of epochs.
        # Start with never re-sampling.
        if idx_epoch == 0:
            print(
                f"\nEpoch: {idx_epoch}: setting resampling criterion fn to ESS = 0.0"
            )
            model.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
                min_ess_ratio=0.0
            )
        elif idx_epoch == n_epochs_no_resampling:
            print(
                f"\nEpoch: {idx_epoch}: setting resampling criterion fn to ESS = 0.5"
            )
            model.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
                min_ess_ratio=0.5
            )
        else:
            pass
        # training
        for idx_batch, batch in enumerate(train_loader):
            batch = gluonts_batch_to_train_pytorch(
                batch=batch,
                device=model.state_prior_model.m.device,
                dtype=model.state_prior_model.m.dtype,
                dims=dims,
                time_features=config.time_feat,
            )
            optimizer.zero_grad()
            loss = model(**batch).sum(dim=(0, 1))
            loss.backward()
            optimizer.step()

        model.trained_epochs = idx_epoch
        if idx_epoch > n_epochs_warmup:
            scheduler.step()

    print("done training.")
    # save model
    torch.save(
        model.state_dict(), os.path.join(log_paths.model, f"{n_epochs}.pt")
    )
    return model


if __name__ == "__main__":
    dims = config.dims
    log_paths = prepare_logging(config=config, consts=consts)
    input_transformer = SyntheticInputTransform(config=config)
    model = RecurrentSwitchingLinearDynamicalSystem(
        # temperature=torch.tensor((1.0 / dims.switch)),
        n_state=dims.state,
        n_obs=dims.target,
        n_ctrl_state=dims.ctrl_state,
        n_particle=dims.particle,
        n_switch=dims.switch,
        input_transformer=input_transformer,
        gls_base_parameters=SyntheticGLSParameters(config=config),
        obs_to_switch_encoder=SyntheticObsToSwitchEncoder(config=config),
        state_to_switch_encoder=SyntheticStateToSwitchEncoder(config=config),
        switch_transition_model=SyntheticSwitchTransitionModel(config=config),
        state_prior_model=SyntheticStatePriorModel(config=config),
        switch_prior_model=SyntheticSwitchPriorModel(config=config),
    )

    train_loader = create_trainset_loader(
        n_data_per_group=config.n_data_per_group,
        batch_size=config.dims.batch,
        dataset_name=config.dataset_name,
        past_length=config.dims.timesteps,
        prediction_length=0,
    )
    inference_loader = create_inference_loader(
        n_data_per_group=config.n_data_per_group,
        batch_size=10000,
        dataset_name=config.dataset_name,
        past_length=config.dims.timesteps + config.n_steps_forecast,
        prediction_length=0,
    )
    val_loader = inference_loader

    model = train(
        log_paths=log_paths,
        model=model,
        train_loader=train_loader,
        val_loader=inference_loader,
    )
    print("generating plots")
    data = gluonts_batch_to_forecast_pytorch(
        batch=next(iter(inference_loader)),
        device=model.state_prior_model.m.device,
        dtype=model.state_prior_model.m.dtype,
        dims=dims,
        time_features=config.time_feat,
        n_steps_forecast=config.n_steps_forecast,
    )
    show_plots = True
    make_state_prior_plots(
        model=model, log_paths=log_paths, show=show_plots,
    )
    make_forecast_plots(
        epoch=model.trained_epochs,
        model=model,
        log_paths=log_paths,
        dims=dims,
        n_steps_forecast=config.n_steps_forecast,
        data=data,
        show=show_plots,
    )
    make_params_over_training_plots(
        epoch=model.trained_epochs,
        model=model,
        log_paths=log_paths,
        show=show_plots,
    )
    print("done")
