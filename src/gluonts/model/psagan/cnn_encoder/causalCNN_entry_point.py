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
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from gluonts.core import serde
from gluonts.dataset import common
from gluonts.dataset.repository import datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def train(arguments):

    logger.info("Downloading estimator config.")
    estimator_config = Path(arguments.estimator) / "estimator.json"
    print(estimator_config)
    with estimator_config.open() as config_file:
        config = config_file.read()
        print(config)
        estimator = serde.load_json(config)

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

    logger.info("Starting training of CausalCNNEncoder.")
    now = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    out_dir = Path(arguments.output_data_dir) / f"out_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)
    estimator.trainer.save_dir = out_dir
    save_model_dir = Path(arguments.model_dir) / f"model_{now}"
    save_model_dir.mkdir(parents=True, exist_ok=True)
    estimator.trainer.save_model_dir = save_model_dir
    estimator.freq = dataset.metadata.freq
    estimator.train(dataset.train)

    description = "Training with MinMax Scaler \n \
    Scaling the context and the negative samples independently\n \
    Changing the loss function so that we have a mean and not a sum for the right term. \n \
    Adding a new term to the loss, which is -logsigmoid(-positive_embedding, negative_embedding) \n \
    We do this to force the embedding of other part of other signal to be different \n \
    than the embedding of the positive signal. \n \
    We now use a dataloader that samples data exactly as described in the paper. However we limit the \n \
    the size of the sampled time series. "
    with open(out_dir / "configuration.txt", "a") as f:
        f.write("Estimator = " + repr(estimator) + "\n")
        f.write("Description = " + description + "\n")
        f.close()


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
