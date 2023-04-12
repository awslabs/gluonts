# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import torch
import warnings
import os

warnings.filterwarnings("ignore")

from pts.model.deepvar import DeepVAREstimator
from pts import Trainer
from datasets import DATASETS, get_dataset
from pathlib import Path
import os
from utils import get_augmented_dataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "outdir", type=str, help="folder to save model and training log)"
)
parser.add_argument(
    "--epochs", type=int, default=50, help="number of training epochs"
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size for training"
)
parser.add_argument(
    "--prediction_length", type=int, default=24, help="prediction length"
)
parser.add_argument("--context_length", type=int, default=None)
parser.add_argument(
    "--rank", type=int, default=10, help="rank of multivariate Gaussian"
)
parser.add_argument(
    "--dataset", type=str, default="electricity", choices=DATASETS
)
parser.add_argument(
    "--num_samples", type=int, default=100, help="number of sample paths"
)
parser.add_argument("--max_target_dim", type=int, default=10)
parser.add_argument(
    "--gaussian", action="store_true", help="whether perform data augmentation"
)
parser.add_argument("--num_noises", type=int, default=100)
parser.add_argument("--sigma", type=float, default=0.1)
args = parser.parse_args()


def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    prediction_length = args.prediction_length
    context_length = (
        4 * prediction_length
        if not args.context_length
        else args.context_length
    )
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = get_dataset(args.dataset, args.max_target_dim)
    target_dim = ds.target_dim
    freq = ds.freq
    if args.gaussian:
        train_ds = get_augmented_dataset(
            ds, num_noises=args.num_noises, sigma=args.sigma
        )
    else:
        train_ds = ds.train_ds

    input_size = 1
    if args.dataset in ["electricity", "taxi", "traffic", "solar"]:
        input_size = 47
    elif args.dataset == "wiki":
        input_size = 45
    elif args.dataset == "exchange_rate":
        input_size = 31

    trainer = Trainer(epochs=args.epochs, batch_size=batch_size, device=device)
    estimator = DeepVAREstimator(
        freq=freq,
        rank=args.rank,
        num_parallel_samples=args.num_samples,
        prediction_length=prediction_length,
        input_size=input_size,
        trainer=trainer,
        target_dim=target_dim,
        context_length=context_length,
    )

    predictor = estimator.train(training_data=train_ds)
    predictor.serialize(Path(args.outdir))


if __name__ == "__main__":
    main()
