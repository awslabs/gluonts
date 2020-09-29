import os
import argparse
import numpy as np
import mxnet as mx
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import consts
from experiments.gluonts_univariate_datasets.config_arsgls import (
    make_model,
    make_experiment_config,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-root_log_path", type=str, default="/home/ubuntu/logs"
    )
    parser.add_argument("-dataset_name", type=str, default="wiki2000_nips")
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

    # random seeds
    seed = args.run_nr if args.run_nr is not None else 0
    mx.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # TODO: config -> hydra or something like that
    config = make_experiment_config(
        dataset_name=args.dataset_name, experiment_name=args.experiment_name,
    )

    model = make_model(config=config).to(dtype=getattr(torch, args.dtype))

    trainer = Trainer(
        gpus=args.gpus,
        default_root_dir=os.path.join(consts.log_dir, config.dataset_name),
        gradient_clip_val=config.grad_clip_norm,
        limit_val_batches=int(np.ceil((500 / config.batch_size_val))),
        max_epochs=config.n_epochs,
        checkpoint_callback=ModelCheckpoint(
            monitor="val_checkpoint_on",
            save_last=True,
        ),
    )

    trainer.fit(model)
    trainer.test(model)
