import argparse
import numpy as np
import mxnet as mx
import torch
from pytorch_lightning import Trainer

import consts
from utils.utils import prepare_logging
from experiments_new_will_replace.pendulum.config import (
    make_model,
    config,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-root_log_path", type=str, default="/home/ubuntu/logs"
    )
    # parser.add_argument(
    #     "-dataset_name", type=str, default="wiki-rolling_nips"
    # )
    # parser.add_argument("-experiment_name", type=str)
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

    # # TODO: config -> hydra or something like that
    # config = make_experiment_config(
    #     dataset_name=args.dataset_name, experiment_name=args.experiment_name,
    # )
    # TODO: Should not need anymore
    log_paths = prepare_logging(
        config=config, consts=consts, run_nr=args.run_nr,
    )

    model = make_model(config=config).to(dtype=getattr(torch, args.dtype))

    trainer = Trainer(
        gpus=args.gpus,
        default_root_dir=log_paths.root,
        gradient_clip_val=config.grad_clip_norm,
        # limit_val_batches=(500 // config.batch_size_val) + 1,
        max_epochs=config.n_epochs,
    )

    trainer.fit(model)
    trainer.test(model)
