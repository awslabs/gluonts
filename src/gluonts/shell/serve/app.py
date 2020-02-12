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
import json
import time
import traceback
from typing import Callable, Tuple, Iterable, List

from flask import Flask, Response, request, jsonify
from pydantic import BaseModel

from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import Config as ForecastConfig
from .util import jsonify_floats


logger = logging.getLogger("gluonts.serve")


class InferenceRequest(BaseModel):
    instances: list
    configuration: ForecastConfig


class ThrougputIter:
    def __init__(self, iterable: Iterable) -> None:
        self.iter = iter(iterable)
        self.timings: List[float] = []

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

    if timings:
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
    else:
        logger.info(
            "No items were provided for inference. No throughput to log."
        )


def get_base_app(execution_params):
    app = Flask("GluonTS scoring service")

    @app.errorhandler(Exception)
    def handle_error(error) -> Tuple[str, int]:
        return traceback.format_exc(), 500

    @app.route("/ping")
    def ping() -> str:
        return ""

    @app.route("/execution-parameters")
    def execution_parameters() -> Response:
        return jsonify(execution_params)

    return app


def handle_predictions(predictor, instances, configuration):
    # create the forecasts
    forecasts = ThrougputIter(
        predictor.predict(
            ListDataset(instances, predictor.freq),
            num_samples=configuration.num_samples,
        )
    )

    predictions = [
        forecast.as_json_dict(configuration) for forecast in forecasts
    ]

    log_throughput(instances, forecasts.timings)
    return predictions


def inference_invocations(predictor_factory) -> Callable[[], Response]:
    def invocations() -> Response:
        predictor = predictor_factory(request.json)
        req = InferenceRequest.parse_obj(request.json)

        predictions = handle_predictions(
            predictor, req.instances, req.configuration
        )
        return jsonify(predictions=jsonify_floats(predictions))

    return invocations


def batch_inference_invocations(
    predictor_factory, configuration
) -> Callable[[], Response]:
    DEBUG = configuration.dict().get("DEBUG")
    predictor = predictor_factory({"configuration": configuration.dict()})

    def invocations() -> Response:
        request_data = request.data.decode("utf8").strip()
        instances = list(map(json.loads, request_data.splitlines()))
        predictions = []

        # we have to take this as the initial start-time since the first
        # forecast is produced before the loop in predictor.predict
        start = time.time()

        forecast_iter = predictor.predict(
            ListDataset(instances, predictor.freq),
            num_samples=configuration.num_samples,
        )

        for forecast in forecast_iter:
            end = time.time()
            prediction = forecast.as_json_dict(configuration)

            if DEBUG:
                prediction["debug"] = {"timing": end - start}

            predictions.append(prediction)

            start = time.time()

        lines = list(map(json.dumps, map(jsonify_floats, predictions)))
        return Response("\n".join(lines), mimetype="application/jsonlines")

    return invocations


def make_app(predictor_factory, execution_params, batch_transform_config):
    app = get_base_app(execution_params)

    if batch_transform_config is not None:
        invocations_fn = batch_inference_invocations(
            predictor_factory, batch_transform_config
        )
    else:
        invocations_fn = inference_invocations(predictor_factory)

    app.route("/invocations", methods=["POST"])(invocations_fn)
    return app
