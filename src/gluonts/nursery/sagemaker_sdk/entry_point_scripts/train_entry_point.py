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


# Standard library imports
from pathlib import Path
import argparse
import os
import json
import logging

# Third-party imports

# First-party imports
from gluonts.core import serde
from gluonts.dataset import common
from gluonts.dataset.repository import datasets
from gluonts.evaluation import Evaluator, backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

# TODO: implement model_fn, input_fn, predict_fn, and output_fn !!
# TODO: segment script for readability


def train(arguments):
    """
    Generic train method that trains a specified estimator on a specified dataset.
    """

    logger.info("Downloading estimator config.")
    estimator_config = Path(arguments.estimator) / "estimator.json"
    with estimator_config.open() as config_file:
        estimator = serde.load_json(config_file.read())

    logger.info("Downloading dataset.")
    if arguments.s3_dataset is None:
        # load built in dataset
        dataset = datasets.get_dataset(arguments.dataset)
    else:
        # load custom dataset
        s3_dataset_dir = Path(arguments.s3_dataset)
        dataset = common.load_datasets(
            metadata=s3_dataset_dir,
            train=s3_dataset_dir / "train",
            test=s3_dataset_dir / "test",
        )

    logger.info("Starting model training.")
    predictor = estimator.train(dataset.train)
    forecast_it, ts_it = backtest.make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
        num_samples=int(arguments.num_samples),
    )

    logger.info("Starting model evaluation.")
    evaluator = Evaluator(quantiles=eval(arguments.quantiles))

    agg_metrics, item_metrics = evaluator(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    # required for metric tracking.
    for name, value in agg_metrics.items():
        logger.info(f"gluonts[metric-{name}]: {value}")

    # save the evaluation results
    metrics_output_dir = Path(arguments.output_data_dir)
    with open(metrics_output_dir / "agg_metrics.json", "w") as f:
        json.dump(agg_metrics, f)
    with open(metrics_output_dir / "item_metrics.csv", "w") as f:
        item_metrics.to_csv(f, index=False)

    # save the model
    model_output_dir = Path(arguments.model_dir)
    predictor.serialize(model_output_dir)


if __name__ == "__main__":
    # TODO switch to click
    parser = argparse.ArgumentParser()

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument(
        "--sm-hps", type=json.loads, default=os.environ["SM_HPS"]
    )

    # input data, output dir and model directories
    parser.add_argument(
        "--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--output-data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    parser.add_argument(
        "--input-dir", type=str, default=os.environ["SM_INPUT_DIR"]
    )

    parser.add_argument(
        "--estimator", type=str, default=os.environ["SM_CHANNEL_ESTIMATOR"]
    )
    # argument possibly not set
    parser.add_argument(
        "--s3-dataset",
        type=str,
        default=os.environ.get("SM_CHANNEL_S3_DATASET"),
    )
    parser.add_argument(
        "--dataset", type=str, default=os.environ["SM_HP_DATASET"]
    )
    parser.add_argument(
        "--num-samples", type=int, default=os.environ["SM_HP_NUM_SAMPLES"]
    )
    parser.add_argument(
        "--quantiles", type=str, default=os.environ["SM_HP_QUANTILES"]
    )

    args, _ = parser.parse_known_args()

    train(args)
