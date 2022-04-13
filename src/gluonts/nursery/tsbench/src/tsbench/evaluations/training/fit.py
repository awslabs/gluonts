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

import tempfile
from pathlib import Path
from typing import cast, List
import numpy as np
from gluonts.model.predictor import Predictor
from mxnet.gluon import nn
from tsbench.config import DatasetConfig, ModelConfig, TrainConfig
from tsbench.gluonts.callbacks import (
    Callback,
    LearningRateScheduleCallback,
    ModelSaverCallback,
    ParameterCountCallback,
)
from .evaluate import FitResult
from .logging import log_metric


def fit_estimator(  # pylint: disable=too-many-statements
    config: ModelConfig,
    dataset: DatasetConfig,
    num_learning_rate_decays: int = 3,
    num_model_checkpoints: int = 5,
    validate: bool = False,
    verbose: bool = True,
) -> FitResult:
    """
    Fits the given estimator using the provided training dataset.

    Args:
        config: The configuration of the estimator to be fitted.
        dataset: The configuration of the dataset to be used for fitting.
        num_learning_rate_decays: The number of times the learning rate should be decayed.
        validate: Whether to use a validation dataset.
        choose_best: Whether the best model according to the validation loss within Hyperband
            intervals should be used.
        verbose: Whether to create multiple predictors and log associated information.

    Returns:
        The result from fitting, contains most notably the list of predictors fitted during
            training. Contains a single entry if the model is not trainable or `verbose` is set to
            false.
    """
    count_callback = ParameterCountCallback()
    callbacks: List[Callback] = [count_callback]

    milestones = []

    # We need to compute the full training time for the config on the dataset
    if isinstance(config, TrainConfig):
        training_time = config.training_fraction * dataset.max_training_time
    else:
        training_time = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        # If model is trainable, we need to create the callback
        saver_callback: ModelSaverCallback
        if isinstance(config, TrainConfig):
            hyperband_milestones = [
                training_time * (1 / 3) ** i
                for i in reversed(range(num_model_checkpoints))
            ]

            # Save at the first half of milestones as well as multiples of the middle milestone
            for i in range(len(hyperband_milestones) // 2):
                milestones += [hyperband_milestones[i]]

            pivot = hyperband_milestones[len(hyperband_milestones) // 2]
            milestones += np.arange(
                pivot, training_time + pivot / 2, pivot
            ).tolist()

            saver_callback = ModelSaverCallback(Path(tmp_dir), milestones)
            callbacks += [saver_callback]

            if num_learning_rate_decays > 0:
                learning_rate_callback = LearningRateScheduleCallback(
                    milestones=[
                        (training_time / (num_learning_rate_decays + 1)) * i
                        for i in range(1, num_learning_rate_decays + 1)
                    ],
                )
                callbacks += [learning_rate_callback]

        # Then, we can create the estimator
        meta = dataset.meta
        estimator = config.create_estimator(
            freq=meta.freq,
            prediction_length=cast(int, meta.prediction_length),
            time_features=dataset.has_time_features,
            training_time=training_time,
            validation_milestones=milestones if validate else [],
            callbacks=callbacks,
        )

        # Afterwards, we run the training. First, we need to optionally add validation data.
        train_kwargs = {}
        if isinstance(config, TrainConfig) and validate:
            train_kwargs["validation_data"] = dataset.data.val().gluonts()

        # Then, we can obtain the predictor
        train_data = dataset.data.train(validate).gluonts()
        predictor = estimator.train(train_data, **train_kwargs)

        # If the model is not trainable, we can return already, logging the recorded time
        if not isinstance(config, TrainConfig):
            if verbose:
                log_metric("num_model_parameters", 0)
                log_metric("num_gradient_updates", 0)
                log_metric("training_time", 0)
            return FitResult(config, [predictor], [0.0], 0)

        # Otherwise, we need to load all models that were stored by the callback
        predictors = []
        model_paths = []

        with tempfile.TemporaryDirectory() as model_dir:
            for i, params in enumerate(saver_callback.saved_parameters):  # type: ignore
                # Load the parameters
                saver_callback.network.load_parameters(params.absolute().as_posix())  # type: ignore

                # Create the predictor
                predictor = cast(TrainConfig, config).create_predictor(
                    estimator,
                    cast(nn.HybridBlock, saver_callback.network),  # type: ignore
                )
                # Serialize and deserialize to properly copy parameters
                path = Path(model_dir) / f"model_{i}"
                path.mkdir()

                predictor.serialize(path)
                model_paths.append(path)

                copied_predictor = Predictor.deserialize(model_paths[i])
                predictors.append(copied_predictor)

                # Log everything
                log_metric(
                    "num_model_parameters", count_callback.num_parameters
                )
                log_metric(
                    "num_gradient_updates",
                    saver_callback.num_gradient_updates[i],  # type: ignore
                )
                log_metric("training_time", saver_callback.training_times[i])  # type: ignore

        return FitResult(
            config,
            predictors,
            saver_callback.training_times,  # type: ignore
            count_callback.num_parameters,
        )
