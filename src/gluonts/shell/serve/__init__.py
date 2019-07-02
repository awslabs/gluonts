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
from typing import Optional

# Third-party imports
import flask
from gunicorn.app.base import BaseApplication
from pydantic import BaseSettings

# First-party imports
from gluonts.core.component import check_gpu_support

from gluonts.model.predictor import Predictor
from gluonts.shell.env import SageMakerEnv

from gluonts.shell.serve.app import make_app
from gluonts.shell.serve.util import number_of_workers


MB = 1024 * 1024


class Settings(BaseSettings):
    # see: https://pydantic-docs.helpmanual.io/#settings
    class Config:
        env_prefix = ''

    model_server_workers: Optional[int] = None
    max_content_length: int = 6 * MB
    sagemaker_batch: bool = False
    sagemaker_max_payload_in_mb: int = 6
    sagemaker_max_concurrent_transforms: int = 2 ** 32 - 1


settings = Settings()
workers = number_of_workers(settings)

execution_params = {
    "MaxConcurrentTransforms": workers,
    "BatchStrategy": "SINGLE_RECORD",
    "MaxPayloadInMB": settings.sagemaker_max_payload_in_mb,
}


class Application(BaseApplication):
    def __init__(self, app, config):
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


def online_forecaster(forecaster):
    from pydoc import locate
    from gluonts.core.component import from_hyperparameters

    forecaster_class = locate(forecaster)

    return lambda request: from_hyperparameters(
        forecaster_class, **request['configuration']
    )


def offline_forecaster(path):
    env = SageMakerEnv(path)
    predictor = Predictor.deserialize(env.path.model)

    return lambda request: predictor


def get_app(predictor_factory):
    return Application(
        app=make_app(predictor_factory, execution_params),
        config={"bind": "0.0.0.0:8080", "workers": workers, "timeout": 100},
    )
