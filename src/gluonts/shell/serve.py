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
import time
import traceback
from ipaddress import IPv4Address
from typing import Iterable, List, Optional, Tuple, Type, Union

# Third-party imports
import requests
import numpy as np
from flask import Flask, Response, jsonify, request
from gunicorn.app.base import BaseApplication
from pydantic import BaseModel, BaseSettings

# First-party imports
import gluonts
from gluonts.core import fqname_for
from gluonts.core.component import check_gpu_support, validated
from gluonts.dataset.common import DataEntry, ListDataset, serialize_data_entry
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


class ThrougputIter:
    def __init__(self, iterable):
        self.iter = iter(iterable)
        self.timings = []

    def __iter__(self):
        try:
            while True:
                start = time.time()
                element = next(self.iter)
                self.timings.append(time.time() - start)
                yield element
        except StopIteration:
            return None


def log_throughput(instances, timings):
    item_lengths = [len(item["target"]) for item in instances]

    total_time = sum(timings)
    avg_time = total_time / len(timings)
    logger.info(
        "Inference took "
        f"{total_time:.2f}s for {len(timings)} items, "
        f"{avg_time:.2f}s on average."
    )
    for idx, (duration, input_length) in enumerate(
        zip(timings, item_lengths), start=1
    ):
        logger.info(
            f"\t{idx} took -> {duration:.2f}s (len(target)=={input_length})."
        )

    # list(zip(timings, item_lengths)


def jsonify_floats(json_object):
    """
    Traverses through the JSON object and converts non JSON-spec compliant
    floats(nan, -inf, inf) to their string representations.

    Parameters
    ----------
    json_object
        JSON object
    """
    if isinstance(json_object, dict):
        return {k: jsonify_floats(v) for k, v in json_object.items()}
    elif isinstance(json_object, list):
        return [jsonify_floats(item) for item in json_object]
    elif isinstance(json_object, float):
        if np.isnan(json_object):
            return "NaN"
        elif np.isposinf(json_object):
            return "Infinity"
        elif np.isneginf(json_object):
            return "-Infinity"
        return json_object
    return json_object


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
        logger.info("Shutting down GluonTS scoring service")


def make_flask_app(predictor_factory, execution_params) -> Flask:
    app = Flask("GluonTS scoring service")

    @app.errorhandler(Exception)
    def handle_error(error) -> Tuple[str, int]:
        return traceback.format_exc(), 500

    @app.route("/ping")
    def ping() -> Response:
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
        forecasts = ThrougputIter(
            predictor.predict(
                dataset, num_eval_samples=req.configuration.num_eval_samples
            )
        )

        predictions = [
            forecast.as_json_dict(req.configuration) for forecast in forecasts
        ]

        log_throughput(req.instances, forecasts.timings)

        return jsonify(predictions=jsonify_floats(predictions))

    return app


def make_gunicorn_app(
    env: Optional[ServeEnv],
    forecaster_type: Optional[Type[Union[Estimator, Predictor]]],
    settings: Settings,
) -> Application:
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

    return gunicorn_app


class ServerFacade:
    """
    A convenience wrapper for sending requests and handling responses to
    an inference server located at the given address.
    """

    @validated()
    def __init__(self, base_address: str) -> None:
        self.base_address = base_address

    def url(self, path) -> str:
        return self.base_address + path

    def ping(self) -> bool:
        try:
            response = requests.get(url=self.url("/ping"))
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def execution_parameters(self) -> dict:
        response = requests.get(
            url=self.url("/execution-parameters"),
            headers={"Accept": "application/json"},
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code >= 400:
            raise RuntimeError(response.content.decode("utf-8"))
        else:
            raise RuntimeError(f"Unexpected {response.status_code} response")

    def invocations(
        self, data_entries: Iterable[DataEntry], configuration: dict
    ) -> List[dict]:
        instances = list(map(serialize_data_entry, data_entries))
        response = requests.post(
            url=self.url("/invocations"),
            json={"instances": instances, "configuration": configuration},
            headers={"Accept": "application/json"},
        )

        if response.status_code == 200:
            predictions = response.json()["predictions"]
            assert len(predictions) == len(instances)
            return predictions
        elif response.status_code >= 400:
            raise RuntimeError(response.content.decode("utf-8"))
        else:
            raise RuntimeError(f"Unexpected {response.status_code} response")
