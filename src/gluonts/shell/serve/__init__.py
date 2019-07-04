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
from typing import Optional, Type, Union

# Third-party imports
import flask
from gunicorn.app.base import BaseApplication
from pydantic import BaseSettings

# First-party imports
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.shell.sagemaker import SageMakerEnv

import logging
import multiprocessing

from .app import make_app

MB = 1024 * 1024


class Settings(BaseSettings):
    # see: https://pydantic-docs.helpmanual.io/#settings
    class Config:
        env_prefix = ''

    model_server_workers: Optional[int] = None
    max_content_length: int = 6 * MB
    sagemaker_batch: bool = False
    sagemaker_batch_strategy: str = 'SINGLE_RECORD'
    sagemaker_max_payload_in_mb: int = 6
    sagemaker_max_concurrent_transforms: int = 2 ** 32 - 1

    @property
    def number_of_workers(self) -> int:
        cpu_count = multiprocessing.cpu_count()

        if self.model_server_workers:
            logging.info(
                f'Using {self.model_server_workers} workers '
                '(set by MODEL_SERVER_WORKERS environment variable).'
            )
            return self.model_server_workers

        elif (
            self.sagemaker_batch
            and self.sagemaker_max_concurrent_transforms < cpu_count
        ):
            logging.info(
                f'Using {self.sagemaker_max_concurrent_transforms} workers '
                '(set by MaxConcurrentTransforms parameter in batch mode).'
            )
            return self.sagemaker_max_concurrent_transforms

        else:
            logging.info(f'Using {cpu_count} workers')
            return cpu_count


settings = Settings()

execution_params = {
    "MaxConcurrentTransforms": settings.number_of_workers,
    "BatchStrategy": settings.sagemaker_batch_strategy,
    "MaxPayloadInMB": settings.sagemaker_max_payload_in_mb,
}


class Application(BaseApplication):
    def __init__(self, app, config) -> None:
        self.application = app
        self.config = config
        BaseApplication.__init__(self)

    def load_config(self) -> None:
        for key, value in self.config.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key, value)

    def load(self) -> flask.Flask:
        return self.application

    def stop(self, *args, **kwargs):
        self.application.logger.info('Shutting down GluonTS scoring service')


def run_inference_server(
    env: SageMakerEnv,
    forecaster_type: Optional[Type[Union[Estimator, Predictor]]],
) -> None:
    if forecaster_type is not None:
        ctor = forecaster_type.from_hyperparameters

        def predictor_factory(request) -> Predictor:
            return ctor(**request['configuration'])

    else:
        predictor = Predictor.deserialize(env.path.model)

        def predictor_factory(request) -> Predictor:
            return predictor

    app = Application(
        app=make_app(predictor_factory, execution_params),
        config={
            "bind": "0.0.0.0:8080",
            "workers": settings.number_of_workers,
            "timeout": 100,
        },
    )

    app.run()
