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
from pathlib import Path

import yaml

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
    print("dataset", arguments.dataset)
    print("s3 dataset", arguments.s3_dataset)
    TRAIN = "train"
    if "mv" in arguments.s3_dataset:
        TRAIN = "train-gan"
    if arguments.s3_dataset is None:
        # load built in dataset
        dataset = datasets.get_dataset(arguments.dataset)
    else:
        # load custom dataset

        s3_dataset_dir = Path(arguments.s3_dataset)
        print("s3 dataset dir", s3_dataset_dir)
        dataset = common.load_datasets(
            metadata=s3_dataset_dir / "metadata",
            train=s3_dataset_dir / TRAIN,
            test=s3_dataset_dir / "test",
        )

    def ds_to_ds(dataset_path):
        if "m4" in dataset_path:
            return "m4_hourly"
        elif "solar" in dataset_path:
            return "solar-energy"
        elif "electricity" in dataset_path:
            return "electricity"
        elif "exchange" in dataset_path:
            return "electricity_rate"
        elif "traffic" in dataset_path:
            return "traffic"

    logger.info("Starting training of the GAN.")

    hps = arguments.sm_hps
    model_id = hps["model_id"]
    out_dir = Path(arguments.output_data_dir) / f"synthetic_out_{model_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    estimator.trainer.save_plot_dir = out_dir
    save_model_dir = Path(arguments.model_dir) / f"synthetic_model_{model_id}"
    save_model_dir.mkdir(parents=True, exist_ok=True)
    estimator.trainer.save_model_dir = save_model_dir
    estimator.freq = dataset.metadata.freq
    configuration_dict = hps["config_dict"]
    configuration_dict["freq"] = estimator.freq

    predictor = estimator.train(dataset.train)

    save_predictor_dir = (
        out_dir / f"{estimator.trainer.use_loss}_{estimator.target_len}"
    )
    save_predictor_dir.mkdir(parents=True, exist_ok=True)
    predictor.serialize(save_predictor_dir)

    description = ""

    with open(out_dir / "configuration.txt", "a") as f:
        f.write("Dataset = " + ds_to_ds(arguments.dataset) + "\n")
        f.write("Estimator = " + repr(estimator) + "\n")
        f.write("Description = " + description + "\n")
        f.close()

    with open(save_predictor_dir / "data.yml", "w") as outfile:
        yaml.dump(configuration_dict, outfile, default_flow_style=False)
        outfile.close()

    with open(out_dir / "data.yml", "w") as outfile:
        yaml.dump(configuration_dict, outfile, default_flow_style=False)
        outfile.close()


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
