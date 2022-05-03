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
import os
from pathlib import Path
from typing import cast, Optional
import click
import mxnet as mx
import numpy as np
from gluonts.env import env
from tsbench.config import DATASET_REGISTRY, MODEL_REGISTRY, TrainConfig
from tsbench.config.dataset import get_dataset_config
from tsbench.config.model import get_model_config
from tsbench.constants import DEFAULT_DATA_PATH
from tsbench.evaluations.training import fit_estimator


@click.command()
# General Parameters
@click.option(
    "--dataset",
    type=click.Choice(DATASET_REGISTRY.keys()),
    required=True,
    help="The dataset to train the model on.",
)
@click.option(
    "--model",
    type=click.Choice(MODEL_REGISTRY.keys()),
    required=True,
    help="The model to fit on the data and to use for predictions.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help=(
        "The seed to use for reproducibility. Defaults to a random seed if not"
        " provided."
    ),
)
@click.option(
    "--data_path",
    type=click.Path(exists=True),
    default=(
        cast(str, os.getenv("SM_INPUT_DIR")) + "/data"
        if os.getenv("SM_INPUT_DIR") is not None
        else DEFAULT_DATA_PATH
    ),
    show_default=True,
    help="The directory where the input datasets are stored.",
)
@click.option(
    "--model_path",
    type=click.Path(),
    default=os.getenv("SM_MODEL_DIR") or Path.home() / "models",
    show_default=True,
    help=(
        "The directory where the trained model and the forecasts shall be"
        " stored."
    ),
)
# Options
@click.option(
    "--validate",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to use validation data.",
)
@click.option(
    "--use_tqdm",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to print progress of training and predictions.",
)
# Common hyperparameters
@click.option("--training_fraction", default=1.0, show_default=True)
@click.option("--num_learning_rate_decays", default=3, show_default=True)
@click.option("--learning_rate", default=1e-3, show_default=True)
@click.option("--context_length_multiple", default=1, show_default=True)
# Specific hyperparameters
@click.option("--deepar_num_layers", default=2, show_default=True)
@click.option("--deepar_num_cells", default=40, show_default=True)
@click.option("--mqcnn_num_filters", default=30, show_default=True)
@click.option("--mqcnn_kernel_size_first", default=7, show_default=True)
@click.option("--mqcnn_kernel_size_hidden", default=3, show_default=True)
@click.option("--mqcnn_kernel_size_last", default=3, show_default=True)
@click.option("--simple_feedforward_hidden_dim", default=40, show_default=True)
@click.option("--simple_feedforward_num_layers", default=2, show_default=True)
@click.option("--tft_hidden_dim", default=32, show_default=True)
@click.option("--tft_num_heads", default=4, show_default=True)
@click.option("--nbeats_num_stacks", default=30, show_default=True)
@click.option("--nbeats_num_blocks", default=1, show_default=True)
def main(
    dataset: str,
    model: str,
    seed: Optional[int],
    data_path: str,
    model_path: str,
    # Options
    validate: bool,
    use_tqdm: bool,
    # Common hyperparameters
    training_fraction: int,
    num_learning_rate_decays: int,
    learning_rate: float,
    context_length_multiple: int,
    # Model hyperparameters
    **kwargs: int,
) -> None:
    """
    Trains and evaluates a GluonTS model, logging all metrics and storing the
    generated forecasts on the test set (and, optionally, the validation set).
    """
    # Basic configuration
    env.use_tqdm = use_tqdm
    logging.basicConfig(level=logging.INFO)

    # Setup
    model_dir = Path(model_path)
    if seed is not None:
        np.random.seed(seed)
        mx.random.seed(seed)

    # Initialize data and model
    data = get_dataset_config(dataset, data_path)
    config = get_model_config(
        model,
        training_fraction=training_fraction,
        learning_rate=learning_rate,
        context_length_multiple=context_length_multiple,
        **{
            key[len(model) + 1 :]: value
            for key, value in kwargs.items()
            if key.startswith(model)
        },
    )
    logging.info("Using model configuration %s.", config)

    # Run training and evaluation
    logging.info("Fitting estimator...")
    fit_result = fit_estimator(
        config,
        data,
        num_learning_rate_decays=num_learning_rate_decays,
        validate=validate,
    )

    logging.info("Saving predictors...")
    fit_result.serialize_predictors(model_dir / "models")

    if validate and isinstance(config, TrainConfig):
        logging.info("Evaluating predictors on validation data...")
        fit_result.evaluate_predictors(
            data,
            data.data.val(),
            model_dir / "val_predictions",
            validation=True,
        )

    logging.info("Evaluating predictors on test data...")
    fit_result.evaluate_predictors(
        data, data.data.test(), model_dir / "predictions"
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()  # type: ignore
