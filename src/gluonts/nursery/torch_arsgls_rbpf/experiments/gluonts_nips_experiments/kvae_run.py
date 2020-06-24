import os
import argparse
import tqdm
from box import Box
import numpy as np
import mxnet as mx
import torch
from torch.optim.adam import Adam

import consts
from experiments.kvae_validator import KVAEValidator as Validator
from models.kalman_variational_autoencoder import KVAEPredictor
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    create_loaders,
    transform_gluonts_to_pytorch,
    get_cardinalities,
    get_dataset,
)
from utils.utils import prepare_logging
from experiments.gluonts_nips_experiments.kvae_config import (
    make_model,
    make_experiment_config,
)
from experiments.gluonts_nips_experiments.kvae_evaluate import test
from visualization.plot_forecasts import plot_predictive_distribution


def model_to(model, gpus, dtype):
    if len(gpus) == 0:
        device = "cpu"
    elif len(gpus) == 1:
        device = f"cuda:{gpus[0]}"
    else:
        device = (
            f"cuda:{gpus[0]}"  # first of the available GPUs to store params on
        )
    print(f"device: {device}")
    model = model.to(dtype).to(device)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, gpus, dim=1)  # batch dim is 1.

    return model


def make_plots(model, data, config, idxs_time_series, idx_epoch):
    for idx in idxs_time_series:
        data_single_series = {
            name: val[:, idx : idx + 1] for name, val in data.items()
        }
        n_steps_filter = config.dims.timesteps
        data_filter = {
            name: d[:n_steps_filter] for name, d in data_single_series.items()
        }
        data_forecast = {
            name: d[n_steps_filter:]
            for name, d in data_single_series.items()
            if name is not "y"
        }
        # 1) filter
        n_particle = model.n_particle
        model.n_particle = config.num_samples_eval
        (
            x_filter_dist,
            z_filter_dist,
            gls_params,
            last_rnn_state,
            inv_measurement_dist,
            z,
        ) = model.filter_forward(**data_filter)
        # 2) forecast --> sample

        (
            x_forecast,
            z_forecast,
            gls_params_forecast,
            rnn_state_forecast,
        ) = model.sample(
            **data_forecast,
            x=x_filter_dist.sample()[-1],
            z=z_filter_dist.sample()[-1],
            gls_params=Box(
                {
                    key: val[-1] if val is not None else None
                    for key, val in gls_params.items()
                }
            ),
            rnn_state=last_rnn_state,
            n_timesteps=config.prediction_length_rolling,
            n_batch=x_filter_dist.loc.shape[2],
        )
        model.n_particle = n_particle
        y_predictive_dist = model.measurement_model(z_filter_dist.sample())
        y_forecast_dist = model.measurement_model(z_forecast)
        mpy = torch.cat([y_predictive_dist.loc, y_forecast_dist.loc,], dim=0)
        Vpy = torch.cat(
            [
                y_predictive_dist.covariance_matrix,
                y_forecast_dist.covariance_matrix,
            ],
            dim=0,
        )

        plot_predictive_distribution(
            y=data["y"].detach(),
            mpy=mpy.detach(),
            Vpy=Vpy.detach(),
            norm_weights=torch.ones(
                mpy.shape[:3], device=mpy.device, dtype=mpy.dtype
            )
            / config.num_samples_eval,
            n_steps_forecast=config.prediction_length_rolling,
            idx_timeseries=0,
            # we already provide only data for series with idx_timeseries.
            idx_particle=None,
            show=False,
            savepath=os.path.join(
                log_paths.plot, f"forecast_b{idx}_ep{idx_epoch}.pdf"
            ),
        )


def train(
    config,
    log_paths,
    model,
    n_epochs,
    lr,
    train_loader,
    input_transforms,
    cardinalities,
    gpus,
    num_samples_eval,
    dtype=torch.float32,
):
    model = model_to(model=model, gpus=gpus, dtype=dtype)
    m = model.module if hasattr(model, "module") else model
    device = m.state_prior_model.m.device

    print(f"Training on device: {device} [{gpus}]; dtype: {dtype}")
    forecaster = KVAEPredictor(
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
    optimizer = Adam(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        amsgrad=False,
        weight_decay=config.weight_decay,
    )
    decay_rate = (1 / 10) ** (1 / n_iter_decay_one_order_of_magnitude)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=decay_rate
    )

    torch.save(m.state_dict(), os.path.join(log_paths.model, f"initial.pt"))
    epoch_iter = tqdm.tqdm(range(n_epochs), desc="Epoch", position=0)
    best_metric = np.inf
    for idx_epoch in epoch_iter:
        torch.save(
            m.state_dict(), os.path.join(log_paths.model, f"{idx_epoch}.pt")
        )
        # training
        for idx_batch, batch in enumerate(train_loader):
            epoch_iter.set_postfix(
                batch=f"{idx_batch}/{len(train_loader)}", refresh=True
            )
            batch = transform_gluonts_to_pytorch(
                batch=batch,
                bias_y=config.normalisation_params[0],
                factor_y=config.normalisation_params[1],
                device=device,
                dtype=dtype,
                time_features=config.time_feat,
                **cardinalities,
            )
            batch.pop("seasonal_indicators")
            optimizer.zero_grad()
            loss = model.loss_em(
                **batch, rao_blackwellized=config.rao_blackwellized
            ).sum(dim=(0,)) / np.prod(
                batch["y"].shape[:2]
            )  # regularisation is tuned for mean
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm
            )
            # print(norm)
            optimizer.step()
        scheduler.step()

        # validation
        if idx_epoch == 0 or idx_epoch > 25:
            prev_n_particles = m.n_particle
            m.n_particle = config.num_samples_eval
            agg_metrics = validator(
                epoch=idx_epoch, save=idx_epoch % 100 == 0 and idx_epoch > 0
            )
            m.n_particle = prev_n_particles

            metric = agg_metrics["mean_wQuantileLoss"]
            if best_metric / metric - 1 > 1e-2:
                torch.save(
                    m.state_dict(), os.path.join(log_paths.model, f"best.pt")
                )
                if metric < best_metric:
                    best_metric = metric

            val_data = transform_gluonts_to_pytorch(
                batch=next(iter(val_loader)),
                bias_y=config.normalisation_params[0],
                factor_y=config.normalisation_params[1],
                device=device,
                dtype=dtype,
                time_features=config.time_feat,
                **cardinalities,
            )
            val_data.pop("seasonal_indicators")
            loss_val = model.loss_em(**val_data).sum(dim=0)
            loss_val_norm = loss_val / np.prod(val_data["y"].shape[:2])
            print(loss_val_norm)

            epoch_iter.set_description_str(
                f"Epoch: {idx_epoch}, "
                f"loss: {loss_val_norm.detach().cpu().numpy():.4f}, "
                f"mQuantile: {agg_metrics['mean_wQuantileLoss']:.4f}, "
                f"min(mQuantile): {min(validator.agg_metrics['mean_wQuantileLoss']):.4f}, "
            )

            if config.dataset_name == "electricity_nips":
                idxs_time_series = (
                    [0, 8, 9] if config.batch_size_val >= 3 else [0]
                )
            else:
                idxs_time_series = (
                    [0, 1, 2] if config.batch_size_val >= 3 else [0]
                )
            make_plots(
                model=model,
                data=val_data,
                config=config,
                idxs_time_series=idxs_time_series,
                idx_epoch=idx_epoch,
            )

    print("done training.")
    torch.save(m.state_dict(), os.path.join(log_paths.model, f"final.pt"))
    return model, validator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-root_log_path", type=str, default="/home/ubuntu/logs"
    )
    parser.add_argument(
        "-dataset_name", type=str, default="exchange_rate_nips"
    )
    parser.add_argument("-experiment_name", type=str)
    parser.add_argument("-run_nr", type=int)
    parser.add_argument(
        "-gpus",
        "--gpus",
        nargs="*",
        default=[0],
        help='"-gpus 0 1 2 3". or "-gpus ".',
    )
    parser.add_argument("-dtype", type=str, default="float64")
    args = parser.parse_args()
    assert len(args.gpus) <= 1, "do not support multi-GPU for this model atm."

    # random seeds
    seed = args.run_nr if args.run_nr is not None else 42
    mx.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # config
    config = make_experiment_config(
        dataset_name=args.dataset_name, experiment_name=args.experiment_name,
    )

    if args.experiment_name is None:
        log_paths = prepare_logging(
            config=config, consts=consts, run_nr=args.run_nr
        )
    else:
        log_paths = prepare_logging(
            config=config, consts=consts, run_nr=args.run_nr
        )

    dataset = get_dataset(config.dataset_name)
    (
        train_loader,
        val_loader,
        test_full_loader,
        test_rolling_loader,
        input_transforms,
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
        val_full_length=False,
    )
    cardinalities = get_cardinalities(
        dataset=dataset, add_trend=config.add_trend
    )

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

    test(
        model=model,
        dataset=dataset,
        log_paths=log_paths,
        input_transforms=input_transforms,
        cardinalities=cardinalities,
        config=config,
        which="best",
    )
    test(
        model=model,
        dataset=dataset,
        log_paths=log_paths,
        input_transforms=input_transforms,
        cardinalities=cardinalities,
        config=config,
        which="final",
    )
