import os
import argparse
import tqdm
import numpy as np
import mxnet as mx
import torch
from torch.optim.adam import Adam

import consts
from experiments.validator import Validator
from models.auxiliary_switching_gaussian_linear_system import SGLSPredictor
from data.gluonts_nips_datasets.gluonts_nips_datasets import create_loaders, \
    transform_gluonts_to_pytorch, get_cardinalities, get_dataset
from utils.utils import prepare_logging
from experiments.gluonts_nips_experiments.auxiliary_config import make_model, \
    make_experiment_config
from inference.smc.resampling import make_criterion_fn_with_ess_threshold
from visualization.plot_forecasts import \
    make_val_plots_auxiliary as make_val_plots
from experiments.gluonts_nips_experiments.auxiliary_evaluate import test


def model_to(model, gpus, dtype):
    if len(gpus) == 0:
        device = "cpu"
    elif len(gpus) == 1:
        device = f"cuda:{gpus[0]}"
    else:
        device = f"cuda:{gpus[0]}"  # first of the available GPUs to store params on
    print(f"device: {device}")
    model = model.to(dtype).to(device)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, gpus, dim=1)  # batch dim is 1.

    return model


def train(config, log_paths, model, n_epochs, lr,
          train_loader, input_transforms, cardinalities, gpus, num_samples_eval,
          dtype=torch.float32,
          ):
    model = model_to(model=model, gpus=gpus, dtype=dtype)
    m = model.module if hasattr(model, "module") else model
    device = m.state_prior_model.m.device

    print(f"Training on device: {device} [{gpus}]; dtype: {dtype}")
    forecaster = SGLSPredictor(
        model=model,
        input_transform=input_transforms["test_full"],
        batch_size=config.batch_size_val,
        prediction_length=config.prediction_length_full,
        freq=dataset.metadata.freq,
        lead_time=0,
        cardinalities=cardinalities,
        dims=config.dims,
        bias_y=config.normalisation_params[0],
        factor_y=config.normalisation_params[1],
        time_feat=config.time_feat,
        keep_filtered_predictions=True,
        yield_forecast_only=False,
    )

    validator = Validator(
        log_paths=log_paths,
        dataset=dataset.train,
        forecaster=forecaster,
        num_samples=num_samples_eval,
        num_series=min(len(dataset.train), 100),
    )

    # ***** Optimizer *****
    n_iter_decay_one_order_of_magnitude = max(int(n_epochs / 2), 1)

    param_names_except_gls = [
        name for name in dict(model.named_parameters()).keys()
        if (not "gls_base_parameters" in name) or ("link_transformers" in name)
    ]
    params_except_gls = tuple(param for name, param in model.named_parameters()
                              if name in param_names_except_gls)
    assert len(params_except_gls) < len(tuple(model.parameters()))

    optimizer_except_gls = Adam(
        params=params_except_gls, lr=lr, betas=(0.9, 0.95), amsgrad=False,
        weight_decay=config.weight_decay)
    optimizer_all = Adam(
        params=model.parameters(), lr=lr, betas=(0.9, 0.95), amsgrad=False,
        weight_decay=config.weight_decay)
    decay_rate = (1 / 10) ** (1 / n_iter_decay_one_order_of_magnitude)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_all, gamma=decay_rate)

    torch.save(m.state_dict(), os.path.join(log_paths.model, f"initial.pt"))
    epoch_iter = tqdm.tqdm(range(n_epochs), desc='Epoch', position=0)
    best_metric = np.inf
    for idx_epoch in epoch_iter:
        torch.save(m.state_dict(),
                   os.path.join(log_paths.model, f"{idx_epoch}.pt"))
        # Start training without base matrices -> the link should learn to pick matrices first.
        if idx_epoch < config.n_epochs_freeze_gls_params:
            optimizer = optimizer_except_gls
        else:
            optimizer = optimizer_all
            if idx_epoch > config.n_epochs_freeze_gls_params:
                scheduler.step()
        # Start with never re-sampling, set re-sampling criterion to ESS = 0.5 later.
        if idx_epoch < config.n_epochs_no_resampling:
            m.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
                min_ess_ratio=0.0)
        else:
            m.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
                min_ess_ratio=0.5)

        # training
        for idx_batch, batch in enumerate(train_loader):
            epoch_iter.set_postfix(
                batch=f"{idx_batch}/{len(train_loader)}", refresh=True)
            batch = transform_gluonts_to_pytorch(
                batch=batch,
                bias_y=config.normalisation_params[0],
                factor_y=config.normalisation_params[1],
                device=device,
                dtype=dtype,
                time_features=config.time_feat,
                **cardinalities,
            )
            optimizer.zero_grad()
            loss = model(**batch).mean(dim=(0, 1))  # sum over time and batch
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  config.grad_clip_norm)
            # print(f"norm {norm}")
            optimizer.step()

        if idx_epoch >= 0:
            # validation
            prev_n_particles = m.n_particle
            m.n_particle = config.num_samples_eval
            agg_metrics = validator(
                epoch=idx_epoch, save=idx_epoch % 100 == 0 and idx_epoch > 0)
            m.n_particle = prev_n_particles

            T, B = config.dims.timesteps, validator.num_series
            epoch_iter.set_description_str(
                f"Epoch: {idx_epoch}, "
                f"loss: {agg_metrics['loss'] / T / B:.4f}, "
                f"ma(loss): {agg_metrics['loss_ma'] / T / B:.4f}, "
                f"mQuantile: {agg_metrics['mean_wQuantileLoss']:.4f}, "
                f"min(loss): {min(validator.agg_metrics['loss']) / T / B:.4f}, "
                f"min(mQuantile): {min(validator.agg_metrics['mean_wQuantileLoss']):.4f}, "
            )
            print(agg_metrics['mean_wQuantileLoss'])

            val_data = transform_gluonts_to_pytorch(
                batch=next(iter(val_loader)),
                bias_y=config.normalisation_params[0],
                factor_y=config.normalisation_params[1],
                device=device,
                dtype=dtype,
                time_features=config.time_feat,
                **cardinalities,
            )
            if config.dataset_name == "electricity_nips":
                idxs_time_series = [0, 8, 9]
            else:
                idxs_time_series = [0]
            for idx_timeseries in idxs_time_series:
                make_val_plots(
                    model=model,
                    data=val_data,
                    idx_particle=None,
                    n_steps_forecast=config.prediction_length_full,
                    idx_timeseries=idx_timeseries,
                    show=False,
                    savepath=os.path.join(
                        log_paths.plot,
                        f"forecast_b{idx_timeseries}_ep{idx_epoch}.pdf")
                )

            metric = agg_metrics["mean_wQuantileLoss"]
            if best_metric / metric - 1 > 1e-2:
                torch.save(m.state_dict(),
                           os.path.join(log_paths.model, f"best.pt"))
                if metric < best_metric:
                    best_metric = metric

    print('done training.')
    torch.save(m.state_dict(), os.path.join(log_paths.model, f"final.pt"))
    return model, validator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_log_path', type=str, default="/home/ubuntu/logs")
    parser.add_argument('-dataset_name', type=str, default="exchange_rate_nips")
    parser.add_argument('-experiment_name', type=str)
    parser.add_argument('-run_nr', type=int)
    parser.add_argument('-gpus', '--gpus', nargs="*", default=[0],
                        help='"-gpus 0 1 2 3". or "-gpus ".')
    parser.add_argument('-dtype', type=str, default="float64")
    args = parser.parse_args()

    # random seeds
    seed = args.run_nr if args.run_nr is not None else 0
    mx.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # config
    config = make_experiment_config(
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
    )

    if args.experiment_name is None:
        log_paths = prepare_logging(
            config=config, consts=consts, run_nr=args.run_nr)
    else:
        log_paths = prepare_logging(
            config=config, consts=consts, run_nr=args.run_nr)

    dataset = get_dataset(config.dataset_name)
    train_loader, val_loader, test_full_loader, test_rolling_loader, input_transforms = \
        create_loaders(
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
    cardinalities = get_cardinalities(
        dataset=dataset, add_trend=config.add_trend)

    model = make_model(config=config)
    model, validator = train(
        config=config,
        log_paths=log_paths,
        model=model,
        n_epochs=config.n_epochs,
        lr=config.lr,
        train_loader=train_loader,
        gpus=[int(gpu) for gpu in args.gpus],
        dtype=getattr(torch, args.dtype),
        input_transforms=input_transforms,
        cardinalities=cardinalities,
        num_samples_eval=config.num_samples_eval,
    )
    m = model.module if hasattr(model, "module") else model

    test(model=model, dataset=dataset, log_paths=log_paths,
         input_transforms=input_transforms, cardinalities=cardinalities,
         config=config,
         which="best")
    test(model=model, dataset=dataset, log_paths=log_paths,
         input_transforms=input_transforms, cardinalities=cardinalities,
         config=config,
         which="final")
