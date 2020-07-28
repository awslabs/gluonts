import os
import argparse
import tqdm
import numpy as np
import mxnet as mx
import torch
from torch.optim.adam import Adam
from pytorch_lightning import Trainer

import consts
from experiments.validator import Validator
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    create_loaders,
    transform_gluonts_to_pytorch,
    get_cardinalities,
    get_dataset,
)
from utils.utils import prepare_logging
from inference.smc.resampling import make_criterion_fn_with_ess_threshold
from visualization.plot_forecasts import (
    make_val_plots_auxiliary as make_val_plots,
)

from experiments_new_will_replace.gluonts_univariate_datasets.config_arsgls import (
    make_model,
    make_experiment_config,
)
from experiments.gluonts_nips_experiments.auxiliary_evaluate import test


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

    # random seeds
    # seed = 11
    seed = args.run_nr if args.run_nr is not None else 0
    mx.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # TODO: config -> hydra or something like that
    config = make_experiment_config(
        dataset_name=args.dataset_name, experiment_name=args.experiment_name,
    )

    # TODO: Should not need anymore
    log_paths = prepare_logging(
        config=config, consts=consts, run_nr=args.run_nr,
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
    )
    cardinalities = get_cardinalities(
        dataset=dataset, add_trend=config.add_trend
    )

    model = make_model(config=config)

    # ***** Optimizer *****

    trainer = Trainer(gpus=args.gpus, gradient_clip_val=config.grad_clip_norm)
    trainer.fit(model)


    # model, validator = train(
    #     config=config,
    #     log_paths=log_paths,
    #     model=model,
    #     n_epochs=config.n_epochs,
    #     lr=config.lr,
    #     train_loader=train_loader,
    #     gpus=[int(gpu) for gpu in args.gpus],
    #     dtype=getattr(torch, args.dtype),
    #     input_transforms=input_transforms,
    #     cardinalities=cardinalities,
    #     num_samples_eval=config.num_samples_eval,
    # )
    # m = model.module if hasattr(model, "module") else model

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
    # TODO: val plots and others
    # TODO: eval all metrics
    # TODO: saving based on metrics

# TODO: annealing kind of stuff. Add standard annealing later also.
# # Start training without base matrices -> the link should learn to pick matrices first.
# if idx_epoch < config.n_epochs_freeze_gls_params:
#     optimizer = optimizer_except_gls
# else:
#     optimizer = optimizer_all
#     if idx_epoch > config.n_epochs_freeze_gls_params:
#         scheduler.step()
# # Start with never re-sampling, set re-sampling criterion to ESS = 0.5 later.
# if idx_epoch < config.n_epochs_no_resampling:
#     m.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
#         min_ess_ratio=0.0
#     )
# else:
#     m.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
#         min_ess_ratio=0.5
#     )