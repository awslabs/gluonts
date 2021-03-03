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
import json
import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
from queue import Empty as QueueEmpty
from typing import Callable, Iterable, List, NamedTuple, Tuple

from flask import Flask, Response, jsonify, request
from pydantic import BaseModel

from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import Config as ForecastConfig
from gluonts.shell.util import forecaster_type_by_name

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


def do(fn, args, queue):
    queue.put(fn(*args))


def with_timeout(fn, args, timeout):
    queue = mp.Queue()
    process = mp.Process(target=do, args=(fn, args, queue))
    process.start()

    try:
        return queue.get(True, timeout=timeout)
    except QueueEmpty:
        os.kill(process.pid, signal.SIGKILL)
        return None


def make_predictions(predictor, dataset, configuration):
    DEBUG = configuration.dict().get("DEBUG")

    # we have to take this as the initial start-time since the first
    # forecast is produced before the loop in predictor.predict
    start = time.time()

    predictions = []

    forecast_iter = predictor.predict(
        dataset,
        num_samples=configuration.num_samples,
    )

    for forecast in forecast_iter:
        end = time.time()
        prediction = forecast.as_json_dict(configuration)

        if DEBUG:
            prediction["debug"] = {"timing": end - start}

        predictions.append(prediction)

        start = time.time()

    return predictions


class ScoredInstanceStat(NamedTuple):
    amount: int
    duration: float


def batch_inference_invocations(
    predictor_factory, configuration, settings
) -> Callable[[], Response]:
    predictor = predictor_factory({"configuration": configuration.dict()})

    scored_instances: List[ScoredInstanceStat] = []
    last_scored = [time.time()]

    def log_scored(when):
        N = 60
        diff = when - last_scored[0]
        if diff > N:
            scored_amount = sum(info.amount for info in scored_instances)
            time_used = sum(info.duration for info in scored_instances)

            logger.info(
                f"Worker pid={os.getpid()}: scored {scored_amount} using on "
                f"avg {round(time_used / scored_amount, 1)} s/ts over the "
                f"last {round(diff)} seconds."
            )
            scored_instances.clear()
            last_scored[0] = time.time()

    def invocations() -> Response:
        request_data = request.data.decode("utf8").strip()

        # request_data can be empty, but .split() will produce a non-empty
        # list, which then means we try to decode an empty string, which
        # causes an error: `''.split() == ['']`
        if request_data:
            instances = list(map(json.loads, request_data.split("\n")))
        else:
            instances = []

        dataset = ListDataset(instances, predictor.freq)

        start_time = time.time()

        if settings.gluonts_batch_timeout > 0:
            predictions = with_timeout(
                make_predictions,
                args=(predictor, dataset, configuration),
                timeout=settings.gluonts_batch_timeout,
            )

            # predictions are None, when predictor timed out
            if predictions is None:
                logger.warning(f"predictor timed out for: {request_data}")
                FallbackPredictor = forecaster_type_by_name(
                    settings.gluonts_batch_fallback_predictor
                )
                fallback_predictor = FallbackPredictor(
                    freq=predictor.freq,
                    prediction_length=predictor.prediction_length,
                )

                predictions = make_predictions(
                    fallback_predictor, dataset, configuration
                )
        else:
            predictions = make_predictions(predictor, dataset, configuration)

        end_time = time.time()

        scored_instances.append(
            ScoredInstanceStat(
                amount=len(predictions), duration=end_time - start_time
            )
        )

        log_scored(when=end_time)

        for forward_field in settings.gluonts_forward_fields:
            for input_item, prediction in zip(dataset, predictions):
                prediction[forward_field] = input_item.get(forward_field)

        lines = list(map(json.dumps, map(jsonify_floats, predictions)))
        return Response("\n".join(lines), mimetype="application/jsonlines")

    def invocations_error_wrapper() -> Response:
        try:
            return invocations()
        except Exception as error:
            return Response(
                json.dumps({"error": traceback.format_exc()}),
                mimetype="application/jsonlines",
            )

    if settings.gluonts_batch_suppress_errors:
        return invocations_error_wrapper
    else:
        return invocations


def make_app(
    predictor_factory, execution_params, batch_transform_config, settings
):
    app = get_base_app(execution_params)

    if batch_transform_config is not None:
        invocations_fn = batch_inference_invocations(
            predictor_factory, batch_transform_config, settings
        )
    else:
        invocations_fn = inference_invocations(predictor_factory)

    app.route("/invocations", methods=["POST"])(invocations_fn)
    return app
