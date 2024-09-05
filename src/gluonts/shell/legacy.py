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

import logging
import tarfile
import traceback
from dataclasses import dataclass
from distutils.util import strtobool
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, Union, Dict, List

import click
from pydantic import BaseModel
from toolz import valmap

from gluonts.dataset import Dataset
from gluonts.dataset.common import FileDataset
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.model import Predictor

from gluonts.mx import DeepAREstimator, Trainer
import gluonts.mx.distribution as dist

from . import sagemaker
from .train import run_train, run_test

logger = logging.getLogger(__name__)


class AutoIgnore(str, Enum):
    auto = "auto"
    ignore = "ignore"


Cardinality = Union[AutoIgnore, List[int], int]
NumDynamicFeat = Union[AutoIgnore, int]


class Likelihood(str, Enum):
    gaussian = "gaussian"
    beta = "beta"
    negative_binomial = "negative-binomial"
    student_t = "student-T"
    # deterministic_L1 = "deterministic-L1"

    def to_gluonts(self):
        return {
            "gaussian": dist.GaussianOutput(),
            "beta": dist.BetaOutput(),
            "negative-binomial": dist.NegativeBinomialOutput(),
            "student-T": dist.StudentTOutput(),
            # are these the same?
            # "deterministic-L1": dist.DeterministicOutput(),
        }[self]


@dataclass
class LazyStats:
    dataset: Dataset
    _stats = None

    def __getattr__(self, key):
        if self._stats is None:
            self._stats = calculate_dataset_statistics(self.dataset)

        return getattr(self._stats, key)


class DeepARHyperparameter(BaseModel):
    # required
    context_length: int
    epochs: int
    prediction_length: int
    time_freq: str

    cardinality: Cardinality = "auto"
    num_dynamic_feat: NumDynamicFeat = "auto"

    # 1 to 1 mapping
    dropout_rate: float = 0.1
    num_cells: int = 40
    num_layers: int = 2

    # trainer
    learning_rate: float = 1e-3
    mini_batch_size: int = 128
    num_batches_per_epoch: Optional[int] = 100

    # some handling required
    likelihood: Likelihood = Likelihood.student_t
    embedding_dimension: int = 10

    # TODO
    early_stopping_patience: Optional[int] = None

    # evaluation
    num_eval_samples: int = 100
    test_quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def derive_cardinality(self, stats):
        if self.cardinality == "ignore":
            return {
                "use_feat_static_cat": False,
                "cardinality": None,
            }

        if self.cardinality == "auto":
            cardinality = [len(cats) for cats in stats.feat_static_cat]
        elif isinstance(self.cardinality, int):
            cardinality = [self.cardinality]
        else:
            cardinality = self.cardinality

        return {
            "use_feat_static_cat": bool(cardinality),
            "cardinality": cardinality,
        }

    def derive_num_dynamic_feat(self, stats):
        if self.num_dynamic_feat == "ignore":
            return {
                "use_feat_dynamic_real": False,
            }

        if self.num_dynamic_feat == "auto":
            return {
                "use_feat_dynamic_real": stats.num_feat_dynamic_real > 0,
            }

        return {
            "use_feat_d ynamic_real": True,
        }

    def to_gluonts(self, dataset):
        stats = LazyStats(dataset)

        card = self.derive_cardinality(stats)

        if card["use_feat_static_cat"]:
            embedding_dimension = [self.embedding_dimension] * len(
                card["cardinality"]
            )
        else:
            embedding_dimension = None

        if self.num_batches_per_epoch is None:
            num_batches_per_epoch = max(
                200,
                int(math.ceil(len(dataset) / self.mini_batch_size)),
            )
        else:
            num_batches_per_epoch = self.num_batches_per_epoch

        estimator = DeepAREstimator(
            context_length=self.context_length,
            dropout_rate=self.dropout_rate,
            freq=self.time_freq,  # renamed
            num_cells=self.num_cells,
            num_layers=self.num_layers,
            prediction_length=self.prediction_length,
            trainer=Trainer(
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                num_batches_per_epoch=num_batches_per_epoch,
            ),
            distr_output=self.likelihood.to_gluonts(),
            embedding_dimension=embedding_dimension,
            **card,
            **self.derive_num_dynamic_feat(stats),
        )

        print(estimator)
        return estimator


class TrainEnv(sagemaker.TrainEnv):
    def __init__(self, *args, **kwargs):
        sagemaker.TrainEnv.__init__(self, *args, **kwargs)
        self.datasets = self._load()

    def _load(self) -> Dict[str, Union[Dataset, Predictor]]:
        datasets = {}

        if "model" in self.channels:
            path = self.channels.pop("model")
            with tarfile.open(path / "model.tar.gz") as targz:
                targz.extractall(path=path)

            datasets["model"] = Predictor.deserialize(path)

        file_dataset = partial(
            FileDataset,
            # we use time_freq in legacy mode
            freq=self.hyperparameters["time_freq"],
        )
        channels = valmap(file_dataset, self.channels)
        if self._listify_dataset():
            channels = valmap(list, datasets)

        datasets.update(channels)

        return datasets

    def _listify_dataset(self):
        return strtobool(self.hyperparameters.get("listify_dataset", "no"))


@click.group()
def cli() -> None:
    pass


# @cli.command(name="serve")
# @click.option(
#     "--data-path",
#     type=click.Path(),
#     envvar="SAGEMAKER_DATA_PATH",
#     default="/opt/ml",
#     help="The root path of all folders mounted by the SageMaker runtime.",
# )
# def serve_command(
#     data_path: str, forecaster: Optional[str], force_static: bool
# ) -> None:
#     from gluonts.shell import serve

#     env = ServeEnv(Path(data_path))
#     env.install_dynamic()

#     logger.info("Run 'serve' command")

#     if not force_static and forecaster is not None:
#         forecaster_type: Optional[Forecaster] = forecaster_type_by_name(
#             forecaster
#         )
#     else:
#         forecaster_type = None

#     gunicorn_app = serve.make_gunicorn_app(
#         env=env,
#         forecaster_type=forecaster_type,
#         settings=Settings(),
#     )
#     gunicorn_app.run()


@cli.command(name="train")
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    envvar="SAGEMAKER_DATA_PATH",
    default="/opt/ml",
    help="The root path of all folders mounted by the SageMaker runtime.",
)
def train_command(data_path: str) -> None:
    logger.info("Run 'train' command")

    try:
        env = TrainEnv(Path(data_path))
        train_dataset = env.datasets["train"]

        deepar_hp = DeepARHyperparameter.parse_obj(env.hyperparameters)
        deepar_estimator = deepar_hp.to_gluonts(train_dataset)

        predictor = run_train(
            deepar_estimator, train_dataset, env.hyperparameters, None, None
        )
        predictor.serialize(env.path.model)

        if "test" in env.datasets:
            run_test(env, predictor, env.datasets["test"], env.hyperparameters)

    except Exception as error:
        with open(
            sagemaker.TrainPaths(Path(data_path)).output / "failure", "w"
        ) as out_file:
            out_file.write(str(error))
            out_file.write("\n\n")
            out_file.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    import logging
    import os

    from gluonts.env import env

    if "TRAINING_JOB_NAME" in os.environ:
        env._push(use_tqdm=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )
    cli(prog_name=__package__)
