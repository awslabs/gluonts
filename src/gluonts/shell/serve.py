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
import logging
import multiprocessing
import traceback
from ipaddress import IPv4Address
from typing import Optional, Tuple, Type, Union

# Third-party imports
from flask import Flask, Response, jsonify, request
from gunicorn.app.base import BaseApplication
from pydantic import BaseModel, BaseSettings

# First-party imports
import gluonts
from gluonts.core import fqname_for
from gluonts.core.component import check_gpu_support
from gluonts.dataset.common import ListDataset
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import Config as ForecastConfig
from gluonts.model.predictor import Predictor
from gluonts.shell.sagemaker import ServeEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S %z]",
)

logger = logging.getLogger("gluonts.serve")

MB = 1024 * 1024


class Settings(BaseSettings):
    # see: https://pydantic-docs.helpmanual.io/#settings
    class Config:
        env_prefix = ""

    model_server_workers: Optional[int] = None
    max_content_length: int = 6 * MB

    sagemaker_server_address: IPv4Address = IPv4Address("0.0.0.0")
    sagemaker_server_port: int = 8080
    sagemaker_server_timeout: int = 100

    sagemaker_batch: bool = False
    sagemaker_batch_strategy: str = "SINGLE_RECORD"

    sagemaker_max_payload_in_mb: int = 6
    sagemaker_max_concurrent_transforms: int = 2 ** 32 - 1

    @property
    def sagemaker_server_bind(self) -> str:
        return f"{self.sagemaker_server_address}:{self.sagemaker_server_port}"

    @property
    def number_of_workers(self) -> int:
        cpu_count = multiprocessing.cpu_count()

        if self.model_server_workers:
            logging.info(
                f"Using {self.model_server_workers} workers "
                f"(set by MODEL_SERVER_WORKERS environment variable)."
            )
            return self.model_server_workers

        elif (
            self.sagemaker_batch
            and self.sagemaker_max_concurrent_transforms < cpu_count
        ):
            logger.info(
                f"Using {self.sagemaker_max_concurrent_transforms} workers "
                f"(set by MaxConcurrentTransforms parameter in batch mode)."
            )
            return self.sagemaker_max_concurrent_transforms

        else:
            logger.info(f"Using {cpu_count} workers")
            return cpu_count


class InferenceRequest(BaseModel):
    instances: list
    configuration: ForecastConfig


class Application(BaseApplication):
    def __init__(self, app, config) -> None:
        self.app = app
        self.config = config
        BaseApplication.__init__(self)

    def load_config(self) -> None:
        for key, value in self.config.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key, value)

    def init(self, parser, opts, args):
        pass

    def load(self) -> Flask:
        return self.app

    def stop(self, *args, **kwargs):
        self.app.logger.info("Shutting down GluonTS scoring service")


def make_flask_app(predictor_factory, execution_params) -> Flask:
    app = Flask("GluonTS scoring service")

    @app.errorhandler(Exception)
    def handle_error(error) -> Tuple[str, int]:
        return traceback.format_exc(), 500

    @app.route("/ping")
    def ping() -> Response:
        app.logger.info("Responding to /ping request")
        return ""

    @app.route("/execution-parameters")
    def execution_parameters() -> Response:
        return jsonify(execution_params)

    @app.route("/invocations", methods=["POST"])
    def invocations() -> Response:
        predictor = predictor_factory(request.json)
        req = InferenceRequest.parse_obj(request.json)

        dataset = ListDataset(req.instances, predictor.freq)

        # create the forecasts
        forecasts = predictor.predict(
            dataset, num_eval_samples=req.configuration.num_eval_samples
        )

        return jsonify(
            predictions=[
                forecast.as_json_dict(req.configuration)
                for forecast in forecasts
            ]
        )

    return app


def run_inference_server(
    env: Optional[ServeEnv],
    forecaster_type: Optional[Type[Union[Estimator, Predictor]]],
) -> None:
    check_gpu_support()

    if forecaster_type is not None:
        logger.info(f"Using dynamic predictor factory")

        ctor = forecaster_type.from_hyperparameters

        forecaster_fq_name = fqname_for(forecaster_type)
        forecaster_version = forecaster_type.__version__

        def predictor_factory(request) -> Predictor:
            return ctor(**request["configuration"])

    else:
        logger.info(f"Using static predictor factory")

        assert env is not None
        predictor = Predictor.deserialize(env.path.model)

        forecaster_fq_name = fqname_for(type(predictor))
        forecaster_version = predictor.__version__

        def predictor_factory(request) -> Predictor:
            return predictor

    logger.info(f"Using gluonts v{gluonts.__version__}")
    logger.info(f"Using forecaster {forecaster_fq_name} v{forecaster_version}")

    settings = Settings()

    execution_params = {
        "MaxConcurrentTransforms": settings.number_of_workers,
        "BatchStrategy": settings.sagemaker_batch_strategy,
        "MaxPayloadInMB": settings.sagemaker_max_payload_in_mb,
    }

    flask_app = make_flask_app(predictor_factory, execution_params)

    gunicorn_app = Application(
        app=flask_app,
        config={
            "bind": settings.sagemaker_server_bind,
            "workers": settings.number_of_workers,
            "timeout": settings.sagemaker_server_timeout,
        },
    )

    gunicorn_app.run()
