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
import pickle

import numpy as np
import torch

from pts.model.simple_feedforward import SimpleFeedForwardEstimator
from pts.model.deepar import DeepAREstimator
from pts.model.lstnet import LSTNetEstimator
from pts.model.n_beats import NBEATSEstimator

from .model.lstm import LSTMEstimator
from .model.ar import AREstimator
from .trainers.SGD import SGD
from .trainers.SCott import SCott
from .trainers.Adam import Adam
from .trainers.Adagrad import Adagrad
from .trainers.SAdam import SAdam
from .trainers.SAdagrad import SAdagrad
from .trainers.SCSG import SCSG


parser = argparse.ArgumentParser(
    description="Experiment on SCott and baselines"
)
# optimizer related arguments
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="total number of epochs to run (default: 200)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    metavar="N",
    help=(
        "batch size (default: 1, only for substantiating the theory, one can"
        " increase the batch size in practice)"
    ),
)
parser.add_argument(
    "--num_batches_per_epoch",
    type=int,
    default=50,
    metavar="N",
    help="number of batches per epoch (default: 50)",
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="SGD",
    help="optimizer used for training (default: SGD)",
)
parser.add_argument(
    "--nesterov",
    default=False,
    action="store_true",
    help="enable momentum for SGD (default: False)",
)
parser.add_argument(
    "--decreasing_step_size",
    default=False,
    action="store_true",
    help="enable decreasing_step_size (default: False)",
)
parser.add_argument(
    "--anchor_freq",
    type=int,
    default=20,
    help=(
        "frequency of updating the anchor gradient for control variate type"
        " optimizers (default: 20)"
    ),
)
parser.add_argument(
    "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.125,
    help="gamma variable used for the SCott",
)

# task (model and dataset) related arguments
parser.add_argument(
    "--prediction_length",
    type=int,
    default=1,
    metavar="N",
    help="prediction length (dimension of the prediction space, default: 1)",
)
parser.add_argument(
    "--context_length",
    type=int,
    default=8,
    metavar="N",
    help="context length (dimension of the context space, default: 8)",
)
parser.add_argument(
    "--input_size",
    type=int,
    default=1,
    metavar="N",
    help="input size of DeepAR (default: 1)",
)
parser.add_argument(
    "--freq",
    type=str,
    default="1H",
    help="frequency of the time series data (default: 1 hour)",
)
parser.add_argument(
    "--task_name",
    type=str,
    default="exchange_rate",
    help="name of the task to run (default: exchange rate)",
)
parser.add_argument(
    "--tensorboard_path",
    type=str,
    default="./runs/test",
    help="path to log the results from trainer",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="exchange_rate",
    help="dataset to use in the experiment (default: exchange_rate)",
)
parser.add_argument(
    "--num_duplicates",
    type=int,
    default=1,
    help="number of duplicates to use (for synthetic data experiment)",
)
parser.add_argument(
    "--num_strata",
    type=int,
    default=8,
    help="number of strata in the dataset (default: 8)",
)
parser.add_argument(
    "--data_file",
    type=str,
    default="None",
    help="name of the file dataset to use",
)

parser.add_argument(
    "--eval_model",
    default=True,
    action="store_true",
    help="enable validation during training (default: True)",
)

parser.add_argument(
    "--model", type=str, default="mlp", help="model to use (default: MLP)"
)
parser.add_argument(
    "--hidden_layer_size",
    type=int,
    default=100,
    metavar="S",
    help="hidden layer size for MLP model (default: 100)",
)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
num_hidden_dimensions = None

# get dataset
with open("./dataset/" + args.data_file + ".csv", "rb") as input:
    data_package = pickle.load(input)
    freq = args.freq
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get trainer
if args.optimizer == "SGD":
    trainer = SGD(
        epochs=args.epochs,
        task_name=args.task_name,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        decreasing_step_size=args.decreasing_step_size,
        nesterov=args.nesterov,
        eval_model=args.eval_model,
        tensorboard_path=args.tensorboard_path,
        device=device,
    )

if args.optimizer == "Adam":
    trainer = Adam(
        epochs=args.epochs,
        task_name=args.task_name,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        decreasing_step_size=args.decreasing_step_size,
        eval_model=args.eval_model,
        tensorboard_path=args.tensorboard_path,
        device=device,
    )

if args.optimizer == "Adagrad":
    trainer = Adagrad(
        epochs=args.epochs,
        task_name=args.task_name,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        decreasing_step_size=args.decreasing_step_size,
        eval_model=args.eval_model,
        tensorboard_path=args.tensorboard_path,
        device=device,
    )

if args.optimizer == "SCott":
    trainer = SCott(
        epochs=args.epochs,
        task_name=args.task_name,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
        num_strata=args.num_strata,
        learning_rate=args.lr,
        freq=args.anchor_freq,
        decreasing_step_size=args.decreasing_step_size,
        nesterov=args.nesterov,
        eval_model=args.eval_model,
        tensorboard_path=args.tensorboard_path,
        gamma=args.gamma,
        device=device,
    )

if args.optimizer == "SAdam":
    trainer = SAdam(
        epochs=args.epochs,
        task_name=args.task_name,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
        num_strata=args.num_strata,
        learning_rate=args.lr,
        freq=args.anchor_freq,
        decreasing_step_size=args.decreasing_step_size,
        eval_model=args.eval_model,
        tensorboard_path=args.tensorboard_path,
        gamma=args.gamma,
        device=device,
    )

if args.optimizer == "SAdagrad":
    trainer = SAdagrad(
        epochs=args.epochs,
        task_name=args.task_name,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
        num_strata=args.num_strata,
        learning_rate=args.lr,
        freq=args.anchor_freq,
        decreasing_step_size=args.decreasing_step_size,
        eval_model=args.eval_model,
        tensorboard_path=args.tensorboard_path,
        gamma=args.gamma,
        device=device,
    )

if args.optimizer == "SCSG":
    trainer = SCSG(
        epochs=args.epochs,
        task_name=args.task_name,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
        num_strata=args.num_strata,
        learning_rate=args.lr,
        freq=args.anchor_freq,
        decreasing_step_size=args.decreasing_step_size,
        nesterov=args.nesterov,
        eval_model=args.eval_model,
        tensorboard_path=args.tensorboard_path,
        device=device,
    )


# get model
if args.model == "mlp":
    estimator = SimpleFeedForwardEstimator(
        freq=freq,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        num_hidden_dimensions=num_hidden_dimensions,
        trainer=trainer,
    )

if args.model == "lstm":
    estimator = LSTMEstimator(
        freq=freq,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        input_size=args.input_size,
        hidden_layer_size=args.hidden_layer_size,
        trainer=trainer,
    )

if args.model == "linear":
    estimator = AREstimator(
        freq=freq,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        trainer=trainer,
    )

if args.model == "deepar":
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        input_size=42,
        cardinality=[70],
        use_feat_static_cat=True,
        trainer=trainer,
    )

if args.model == "nbeats":
    estimator = NBEATSEstimator(
        freq=freq,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        trainer=trainer,
    )

if args.model == "lstnet":
    estimator = LSTNetEstimator(
        skip_size=1,
        ar_window=1,
        num_series=1,
        channels=6,
        kernel_size=2,
        context_length=args.context_length,
        freq=freq,
        prediction_length=args.prediction_length,
        trainer=trainer,
    )

# get proper training entry
if args.optimizer in ["SGD", "Adam", "Adagrad", "SCSG", "BatchSGD"]:
    predictor = estimator.train(data_package=data_package)
else:
    predictor = estimator.stratified_train(data_package=data_package)
