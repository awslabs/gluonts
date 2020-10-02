import os
import consts
import argparse
import numpy as np
import mxnet as mx
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from experiments.pymunk.configs import (
    make_model,
    make_experiment_config,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-root_log_path", type=str, default="/home/ubuntu/logs"
    )
    parser.add_argument("-dataset_name", type=str, default="box")
    parser.add_argument("-experiment_name", type=str, default="arsgls")
    parser.add_argument("-distributed_backend", type=str, default="dp")
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
    args.gpus = None if len(args.gpus) == 0 else args.gpus

    if not (args.gpus is None or len(args.gpus) <= 1):
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

    model = make_model(config=config).to(dtype=getattr(torch, args.dtype))

    trainer = Trainer(
        gpus=args.gpus,
        default_root_dir=os.path.join(consts.log_dir, config.dataset_name),
        gradient_clip_val=config.grad_clip_norm,
        max_epochs=config.n_epochs,
        checkpoint_callback=ModelCheckpoint(
            monitor="val_checkpoint_on", save_last=True,
        ),
        distributed_backend=args.distributed_backend,
    )

    trainer.fit(model)

    ckpt_dir = trainer.checkpoint_callback.dirpath
    prefix = trainer.checkpoint_callback.prefix
    metrics_dir = os.path.join(trainer.logger.log_dir, "metrics")
    for name in ["last"]:
        ckpt_path = os.path.join(ckpt_dir, f"{prefix}{name}.ckpt")
        result = trainer.test(ckpt_path=ckpt_path)
        if (isinstance(result, list) and len(result)) == 1:
            result = result[0]
        print(result)
        np.savez(os.path.join(metrics_dir, f"{name}.npz"), result)
