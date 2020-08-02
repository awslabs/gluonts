import os
import argparse
import tqdm
import numpy as np
import mxnet as mx
import torch
from torch.optim.adam import Adam
from pytorch_lightning import Trainer
import pytorch_lightning as pl

import consts

from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    create_loaders,
    transform_gluonts_to_pytorch,
    get_cardinalities,
    get_dataset,
)
from utils.utils import prepare_logging
from inference.smc.resampling import make_criterion_fn_with_ess_threshold

from experiments_new_will_replace.gluonts_univariate_datasets.config_arsgls import (
    make_model,
    make_experiment_config,
)
from experiments.gluonts_nips_experiments.auxiliary_evaluate import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(  # TODO: unused?
        "-root_log_path", type=str, default="/home/ubuntu/logs"
    )
    parser.add_argument(
        "-dataset_name", type=str, default="wiki-rolling_nips"
    )
    parser.add_argument("-experiment_name", type=str)
    parser.add_argument("-run_nr", type=int, default=None)
    parser.add_argument(
        "-gpus",
        "--gpus",
        nargs="*",
        default=[0],
        help='"-gpus 0 1 2 3". or "-gpus ".',
    )
    parser.add_argument("-dtype", type=str, default="float64")
    args = parser.parse_args()

    if not len(args.gpus) <= 1:
        raise Exception(
            "multi-GPU does not work anymore since we switched to "
            "Pytorch-Lightning. The reason is that the SSMs are implemented "
            "with time-first not batch-first. Although torch DataParallel can "
            "handle this (dim arg), pytorch lightning does not."
            "Will add support for this later through one of these options: "
            "1) Make PR to lightning that allows time-first. "
            "2) Re-write SSMs for batch-first. "
            "3) In lightning Wrapper, just transpose before SSMs, "
            "leaving SSM implementations as they are. (favorite solution)"
        )

    # random seeds
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

    model = make_model(config=config).to(dtype=getattr(torch, args.dtype))

    # ***** Optimizer *****

    trainer = Trainer(
        gpus=args.gpus,
        default_root_dir=log_paths.root,
        gradient_clip_val=config.grad_clip_norm,
        limit_val_batches=(500 // config.batch_size_val) + 1,
        max_epochs=config.n_epochs,
        # logger=pl.loggers.MLFlowLogger(),
    )

    trainer.fit(model)
    trainer.test(model)
